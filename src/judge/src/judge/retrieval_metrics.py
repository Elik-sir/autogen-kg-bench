"""LLM-as-a-judge retrieval metrics: Context Relevance and Evidence Recall.

Uses ``AsyncOpenAI`` with structured outputs (Pydantic) and parallel asyncio calls.

**Пакетный режим** (как ``extended_metrics``): каталог с ``simple.jsonl``, …
(см. :func:`judge.io.discover_benchmark_result_files`), например
``vector-rag/results``::

    uv run python -m judge.retrieval_metrics --input-dir ../benchmarks/vector-rag/results

По умолчанию JSON пишется в ``<input-dir>/retrieval_metrics/``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, TypeVar

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError
from tqdm.asyncio import tqdm

from .config import RagasJudgeSettings
from .extended_metrics import extract_claims, load_raw_items
from .io import (
    QUESTION_TYPE_ORDER,
    discover_benchmark_result_files,
    normalize_eval_record,
    try_load_per_type_metrics_json,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PLACEHOLDER_CONTEXT = "(No retrieval context was provided for this example.)"

T = TypeVar("T", bound=BaseModel)

# ---------------------------------------------------------------------------
# Pydantic schemas for structured LLM outputs
# ---------------------------------------------------------------------------


class ChunkRelevanceJudgment(BaseModel):
    """Per-chunk relevance R(c, Q, E) in [0, 1]."""

    score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "How relevant the chunk is to answering the question in light of the "
            "reference evidence. 1.0 = fully relevant and on-topic; 0.0 = irrelevant."
        ),
    )


class ClaimSupportJudgment(BaseModel):
    """Indicator for whether retrieved context supports a reference claim (S(c, C))."""

    supported: bool = Field(
        description=(
            "True if the retrieved context entails or clearly supports the claim; "
            "False if the claim is absent, contradicted, or cannot be verified from the text."
        ),
    )


# ---------------------------------------------------------------------------
# LLM helper (parse + JSON fallback)
# ---------------------------------------------------------------------------


def _strip_markdown_fenced_json(content: str) -> str:
    s = (content or "").strip()
    if not s.startswith("```"):
        return s
    lines = s.split("\n")
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _looks_like_json_schema_echo(obj: Any) -> bool:
    """Models sometimes echo JSON Schema back instead of a data instance."""
    if not isinstance(obj, dict):
        return False
    if "properties" in obj and obj.get("type") == "object":
        return "score" not in obj and "supported" not in obj
    return False


def _fallback_json_user_instruction(response_model: type[BaseModel]) -> str:
    """
    Explicit JSON shape for ``json_object`` fallback.

    Avoid pasting ``model_json_schema()`` into the prompt: many OpenAI-compatible
    models return the schema object instead of an instance matching it.
    """
    # Defined in ``generation_metrics`` — matched by name to avoid import cycles.
    if getattr(response_model, "__name__", "") == "FaithfulnessClaimsExtraction":
        return (
            "Reply with exactly one JSON object. Key \"claims\": a JSON array of strings, "
            "each string one short atomic factual assertion from the given answer. "
            'Example: {"claims": ["Revenue grew in 2024", "HQ is in Dallas"]}. '
            'Use {"claims": []} if there are no factual claims. No markdown, no other keys.'
        )
    if response_model is ChunkRelevanceJudgment:
        return (
            "Reply with exactly one JSON object and nothing else. "
            'It must have a single key "score" with a number between 0.0 and 1.0. '
            'Example: {"score": 0.73}. No markdown fences, no explanation, no other keys.'
        )
    if response_model is ClaimSupportJudgment:
        return (
            "Reply with exactly one JSON object and nothing else. "
            'It must have a single key "supported" with a boolean value. '
            'Example: {"supported": true}. No markdown fences, no explanation, no other keys.'
        )
    schema_hint = response_model.model_json_schema()
    return (
        "Reply with a single JSON object only (no markdown). "
        "It must be a *data* instance, not a JSON Schema. Match keys to this schema summary:\n"
        + json.dumps(schema_hint, ensure_ascii=False)
    )


def _parse_response_payload(raw: str, response_model: type[T]) -> T:
    cleaned = _strip_markdown_fenced_json(raw)
    try:
        data: Any = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            data = json.loads(cleaned[start : end + 1])
        else:
            raise
    if _looks_like_json_schema_echo(data):
        raise ValueError("LLM returned JSON Schema instead of a result object")
    return response_model.model_validate(data)


async def _llm_parse(
    client: AsyncOpenAI,
    *,
    model: str,
    max_tokens: int,
    system: str,
    user: str,
    response_model: type[T],
    extra_body: dict[str, Any] | None = None,
) -> T:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    create_kw: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "response_format": response_model,
    }
    if extra_body is not None:
        create_kw["extra_body"] = extra_body

    try:
        resp = await client.chat.completions.parse(**create_kw)
        parsed = resp.choices[0].message.parsed
        if parsed is not None:
            return parsed
    except Exception:
        pass

    fallback_messages = messages + [
        {"role": "user", "content": _fallback_json_user_instruction(response_model)},
    ]
    create_kw2: dict[str, Any] = {
        "model": model,
        "messages": fallback_messages,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    if extra_body is not None:
        create_kw2["extra_body"] = extra_body
    resp3 = await client.chat.completions.create(**create_kw2)
    raw3 = (resp3.choices[0].message.content or "").strip()
    try:
        return _parse_response_payload(raw3, response_model)
    except (json.JSONDecodeError, ValueError, ValidationError) as first_err:
        repair_messages = fallback_messages + [
            {
                "role": "user",
                "content": (
                    "Your previous reply was not valid. "
                    + _fallback_json_user_instruction(response_model)
                    + " Reply again with only that one JSON object."
                ),
            }
        ]
        create_kw3 = {**create_kw2, "messages": repair_messages}
        resp4 = await client.chat.completions.create(**create_kw3)
        raw4 = (resp4.choices[0].message.content or "").strip()
        try:
            return _parse_response_payload(raw4, response_model)
        except (json.JSONDecodeError, ValueError, ValidationError) as e:
            raise first_err from e


def _normalize_context_list(retrieved_context: str | list[str]) -> str:
    if isinstance(retrieved_context, str):
        return retrieved_context.strip()
    parts = [c.strip() for c in retrieved_context if (c or "").strip()]
    return "\n\n---\n\n".join(parts)


_CONTEXT_RELEVANCE_SYSTEM = """You are an expert evaluator for retrieval-augmented generation (RAG).

Your task: score a SINGLE retrieved text chunk for relevance to the user's question AND to the provided reference evidence snippets.

Scoring rules:
- Consider whether the chunk would help answer the question, aligned with what the evidence indicates is important.
- Penalize tangential, duplicate, or misleading content.
- Output ONLY the structured fields required by the schema (a numeric score between 0.0 and 1.0).

Interpretation:
- 1.0: the chunk is directly useful and consistent with the question and evidence focus.
- 0.0: the chunk is irrelevant, off-topic, or misleading for this question and evidence.
- Values between 0 and 1: partial relevance."""


_EVIDENCE_RECALL_SYSTEM = """You are an expert evaluator for retrieval-augmented generation (RAG).

Your task: decide whether a specific REFERENCE CLAIM is supported by the RETRIEVED CONTEXT alone.

Rules:
- Answer supported=true only if the context clearly entails or unambiguously supports the claim.
- If the claim is missing, only weakly implied, or contradicted, answer supported=false.
- Do not use outside knowledge; rely only on the retrieved context text.
- Output ONLY the structured fields required by the schema (boolean supported)."""


async def _score_one_chunk(
    client: AsyncOpenAI,
    *,
    model: str,
    max_tokens: int,
    chunk: str,
    question: str,
    evidence_block: str,
    extra_body: dict[str, Any] | None,
    sem: asyncio.Semaphore,
) -> float:
    async with sem:
        user = (
            f"USER QUESTION:\n{question.strip()}\n\n"
            f"REFERENCE EVIDENCE (ground truth snippets):\n{evidence_block}\n\n"
            f"RETRIEVED CHUNK TO SCORE:\n{chunk.strip()}\n\n"
            "Return the relevance score for this chunk only."
        )
        out = await _llm_parse(
            client,
            model=model,
            max_tokens=max_tokens,
            system=_CONTEXT_RELEVANCE_SYSTEM,
            user=user,
            response_model=ChunkRelevanceJudgment,
            extra_body=extra_body,
        )
        return float(out.score)


async def _support_one_claim(
    client: AsyncOpenAI,
    *,
    model: str,
    max_tokens: int,
    claim: str,
    context_text: str,
    extra_body: dict[str, Any] | None,
    sem: asyncio.Semaphore,
) -> float:
    async with sem:
        user = (
            f"REFERENCE CLAIM:\n{claim.strip()}\n\n"
            f"RETRIEVED CONTEXT:\n{context_text}\n\n"
            "Is the claim supported by the retrieved context? Return supported true/false."
        )
        out = await _llm_parse(
            client,
            model=model,
            max_tokens=max_tokens,
            system=_EVIDENCE_RECALL_SYSTEM,
            user=user,
            response_model=ClaimSupportJudgment,
            extra_body=extra_body,
        )
        return 1.0 if out.supported else 0.0


async def evaluate_context_relevance(
    contexts: list[str],
    question: str,
    evidence: list[str],
    *,
    client: AsyncOpenAI,
    model: str,
    max_tokens: int = 1024,
    max_concurrency: int = 8,
    extra_body: dict[str, Any] | None = None,
) -> float:
    """
    Context Relevance: ``(1 / |C|) * sum_{c in C} R(c, Q, E)``.

    Parameters
    ----------
    contexts
        Retrieved chunks ``C``.
    question
        User question ``Q``.
    evidence
        Reference evidence strings ``E``.
    client, model
        OpenAI API client and chat model id.
    max_tokens
        Max completion tokens per LLM call.
    max_concurrency
        Parallelism cap (semaphore).

    Returns
    -------
    float
        Mean relevance in ``[0, 1]``. Returns ``0.0`` if ``contexts`` is empty.
    """
    cleaned = [c.strip() for c in contexts if (c or "").strip()]
    if not cleaned:
        return 0.0
    ev_parts = [e.strip() for e in evidence if (e or "").strip()]
    evidence_block = "\n\n".join(ev_parts) if ev_parts else "(none provided)"

    sem = asyncio.Semaphore(max(1, max_concurrency))
    scores = await asyncio.gather(
        *(
            _score_one_chunk(
                client,
                model=model,
                max_tokens=max_tokens,
                chunk=c,
                question=question,
                evidence_block=evidence_block,
                extra_body=extra_body,
                sem=sem,
            )
            for c in cleaned
        )
    )
    return sum(scores) / len(scores)


async def evaluate_evidence_recall(
    reference_claims: list[str],
    retrieved_context: str | list[str],
    *,
    client: AsyncOpenAI,
    model: str,
    max_tokens: int = 1024,
    max_concurrency: int = 8,
    extra_body: dict[str, Any] | None = None,
) -> float:
    """
    Evidence Recall: ``(1 / |R|) * sum_{c in R} 1[S(c, C)]``.

    Parameters
    ----------
    reference_claims
        Claims that ought to be recoverable from retrieval ``R``.
    retrieved_context
        Full retrieved context ``C`` (single string or list of chunks joined in order).
    client, model
        OpenAI API client and chat model id.

    Returns
    -------
    float
        Fraction of claims supported by context in ``[0, 1]``.
        Returns ``0.0`` if ``reference_claims`` is empty (no claims to satisfy).
    """
    claims = [c.strip() for c in reference_claims if (c or "").strip()]
    if not claims:
        return 0.0
    context_text = _normalize_context_list(retrieved_context)
    if not context_text:
        return 0.0

    sem = asyncio.Semaphore(max(1, max_concurrency))
    indicators = await asyncio.gather(
        *(
            _support_one_claim(
                client,
                model=model,
                max_tokens=max_tokens,
                claim=cl,
                context_text=context_text,
                extra_body=extra_body,
                sem=sem,
            )
            for cl in claims
        )
    )
    return sum(indicators) / len(indicators)


class GraphRAGEvaluator:
    """
    Convenience wrapper around ``AsyncOpenAI`` for retrieval LLM-judge metrics.

    Can be constructed from explicit parameters or from :class:`RagasJudgeSettings`.
    """

    def __init__(
        self,
        *,
        client: AsyncOpenAI | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        max_concurrency: int = 8,
        settings: RagasJudgeSettings | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        openai_chat_extra_body: dict[str, Any] | None = None,
    ) -> None:
        if settings is not None:
            self._client = client or AsyncOpenAI(**settings.openai_client_kwargs())
            self._model = model or settings.llm_model
            self._max_tokens = (
                settings.llm_max_tokens if max_tokens is None else max_tokens
            )
            self._extra_body = (
                settings.openai_chat_extra_body
                if openai_chat_extra_body is None
                else openai_chat_extra_body
            )
        else:
            if client is None:
                key = api_key or os.environ.get("OPENAI_API_KEY")
                if not key:
                    raise ValueError("Provide client=, settings=, or api_key= / OPENAI_API_KEY.")
                kw: dict[str, Any] = {"api_key": key}
                if base_url:
                    kw["base_url"] = base_url.rstrip("/")
                if default_headers:
                    kw["default_headers"] = dict(default_headers)
                self._client = AsyncOpenAI(**kw)
            else:
                self._client = client
            if not model:
                raise ValueError("model= is required when settings= is not used.")
            self._model = model
            self._max_tokens = 1024 if max_tokens is None else max_tokens
            self._extra_body = openai_chat_extra_body

        self._max_concurrency = max_concurrency

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @property
    def model(self) -> str:
        return self._model

    async def evaluate_context_relevance(
        self,
        contexts: list[str],
        question: str,
        evidence: list[str],
    ) -> float:
        """Average per-chunk relevance R(c, Q, E)."""
        return await evaluate_context_relevance(
            contexts,
            question,
            evidence,
            client=self._client,
            model=self._model,
            max_tokens=self._max_tokens,
            max_concurrency=self._max_concurrency,
            extra_body=self._extra_body,
        )

    async def evaluate_evidence_recall(
        self,
        reference_claims: list[str],
        retrieved_context: str | list[str],
    ) -> float:
        """Fraction of reference claims supported by retrieved context."""
        return await evaluate_evidence_recall(
            reference_claims,
            retrieved_context,
            client=self._client,
            model=self._model,
            max_tokens=self._max_tokens,
            max_concurrency=self._max_concurrency,
            extra_body=self._extra_body,
        )


def _load_env() -> None:
    load_dotenv(_PROJECT_ROOT / ".env", override=False)


def _is_placeholder_contexts(contexts: list[str]) -> bool:
    """True if ``normalize_eval_record`` filled the RAGAS placeholder (no real retrieval)."""
    if not contexts:
        return True
    if len(contexts) == 1 and (contexts[0] or "").strip() == _PLACEHOLDER_CONTEXT:
        return True
    return False


def _ground_truth_to_evidence_and_claims(ground_truth: str | None) -> tuple[list[str], list[str]]:
    """
    Derive evidence snippets ``E`` and reference claims ``R`` from a single ground-truth string.

    If the row defines ``evidence`` / ``reference_claims`` (list[str]), the caller should
    prefer those; this helper is for typical benchmark rows with only ``ground_truth``.
    """
    if ground_truth is None:
        return [], []
    s = str(ground_truth).strip()
    if not s:
        return [], []
    if ";" in s and len(s) > 80:
        parts = [p.strip() for p in s.split(";") if len(p.strip()) > 15]
        if len(parts) >= 2:
            return parts, parts
    blocks = [b.strip() for b in s.split("\n\n") if b.strip()]
    if len(blocks) >= 2:
        return blocks, blocks
    return [s], [s]


def _row_evidence_and_claims(
    raw_row: dict[str, Any],
    *,
    ground_truth_str: str | None,
    atomic_claims: list[str] | None,
) -> tuple[list[str], list[str]]:
    ev_raw = raw_row.get("evidence")
    if isinstance(ev_raw, list) and ev_raw:
        evidence = [str(x).strip() for x in ev_raw if str(x).strip()]
    elif isinstance(ev_raw, str) and ev_raw.strip():
        evidence = [ev_raw.strip()]
    else:
        evidence, _ = _ground_truth_to_evidence_and_claims(ground_truth_str)

    rc_raw = raw_row.get("reference_claims")
    if isinstance(rc_raw, list) and rc_raw:
        claims = [str(x).strip() for x in rc_raw if str(x).strip()]
    elif atomic_claims is not None:
        claims = list(atomic_claims)
    else:
        _, claims = _ground_truth_to_evidence_and_claims(ground_truth_str)

    if not evidence and claims:
        evidence = list(claims)
    if not claims and evidence:
        claims = list(evidence)
    return evidence, claims


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def _safe_mean(values: list[float | None]) -> float | None:
    xs = [
        float(x)
        for x in values
        if x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))
    ]
    if not xs:
        return None
    return sum(xs) / len(xs)


def _mean_retrieval_from_rows(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    keys = ("context_relevance", "evidence_recall")
    vals: dict[str, list[float | None]] = {k: [] for k in keys}
    for r in rows:
        for k in keys:
            v = r.get(k)
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                vals[k].append(None)
            else:
                vals[k].append(float(v))
    return {k: _safe_mean(vals[k]) for k in keys}


async def _process_one_retrieval_row(
    client: AsyncOpenAI,
    *,
    model: str,
    max_tokens: int,
    extra_body: dict[str, Any] | None,
    per_metric_concurrency: int,
    row_sem: asyncio.Semaphore,
    raw_row: dict[str, Any],
    norm: dict[str, Any] | None,
    atomic_claims_from_gt: bool,
) -> dict[str, Any]:
    if not norm:
        return {
            **raw_row,
            "context_relevance": None,
            "evidence_recall": None,
            "retrieval_metrics_note": "invalid_row",
        }

    ctxs = list(norm.get("contexts") or [])
    if _is_placeholder_contexts(ctxs):
        return {
            **raw_row,
            "context_relevance": None,
            "evidence_recall": None,
            "retrieval_metrics_note": "missing_or_placeholder_contexts",
        }

    gt = norm.get("ground_truth")
    gt_s = str(gt).strip() if gt is not None else None

    async with row_sem:
        atomic: list[str] | None = None
        if atomic_claims_from_gt and gt_s:
            atomic = await extract_claims(
                client,
                model=model,
                max_tokens=max_tokens,
                text=gt_s,
            )
            if not atomic:
                atomic = None

        evidence, claims = _row_evidence_and_claims(
            raw_row,
            ground_truth_str=gt_s,
            atomic_claims=atomic,
        )
        if not evidence and not claims:
            return {
                **raw_row,
                "context_relevance": None,
                "evidence_recall": None,
                "retrieval_metrics_note": "no_ground_truth",
            }

        question = str(norm["question"])

        cr, er = await asyncio.gather(
            evaluate_context_relevance(
                ctxs,
                question,
                evidence,
                client=client,
                model=model,
                max_tokens=max_tokens,
                max_concurrency=per_metric_concurrency,
                extra_body=extra_body,
            ),
            evaluate_evidence_recall(
                claims,
                ctxs,
                client=client,
                model=model,
                max_tokens=max_tokens,
                max_concurrency=per_metric_concurrency,
                extra_body=extra_body,
            ),
        )

    out = {**raw_row, "context_relevance": cr, "evidence_recall": er}
    if atomic_claims_from_gt:
        out["reference_claims_used"] = claims
    return out


async def compute_retrieval_metrics_async(
    rows: list[dict[str, Any]],
    settings: RagasJudgeSettings,
    *,
    row_concurrency: int = 4,
    per_metric_concurrency: int = 8,
    atomic_claims_from_gt: bool = False,
    show_progress: bool = True,
) -> list[dict[str, Any]]:
    """
    Для каждой сырой строки бенчмарка (как из ``load_raw_items``) считает
    ``context_relevance`` и ``evidence_recall`` и мержит в исходный dict.

    ``evidence`` / ``reference_claims`` в строке, если есть, имеют приоритет;
    иначе эталон берётся из ``ground_truth`` после :func:`normalize_eval_record`.
    """
    if not rows:
        return []

    norms: list[dict[str, Any] | None] = []
    for r in rows:
        try:
            norms.append(normalize_eval_record(dict(r)))
        except ValueError:
            norms.append(None)

    client = AsyncOpenAI(**settings.openai_client_kwargs())
    row_sem = asyncio.Semaphore(max(1, row_concurrency))

    async def _one(i: int) -> dict[str, Any]:
        return await _process_one_retrieval_row(
            client,
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            extra_body=settings.openai_chat_extra_body,
            per_metric_concurrency=per_metric_concurrency,
            row_sem=row_sem,
            raw_row=dict(rows[i]),
            norm=norms[i],
            atomic_claims_from_gt=atomic_claims_from_gt,
        )

    tasks = [_one(i) for i in range(len(rows))]
    if show_progress:
        merged = await tqdm.gather(*tasks, desc="Retrieval LLM metrics")
    else:
        merged = await asyncio.gather(*tasks)
    return list(merged)


async def run_retrieval_input_dir(
    input_dir: Path,
    output_dir: Path,
    settings: RagasJudgeSettings,
    *,
    row_concurrency: int,
    per_metric_concurrency: int,
    atomic_claims_from_gt: bool,
    skip_existing: bool = False,
) -> None:
    files = discover_benchmark_result_files(input_dir)
    if not files:
        print(
            f"No per-type result files in {input_dir} (expected one of: "
            f"{', '.join(f'{t}.jsonl' for t in QUESTION_TYPE_ORDER)}).",
            file=sys.stderr,
        )
        raise SystemExit(1)
    output_dir.mkdir(parents=True, exist_ok=True)
    agg_parts: list[tuple[str, dict[str, float | None], int]] = []

    for question_type, in_path in files:
        raw = load_raw_items(in_path)
        if not raw:
            print(f"[skip] {question_type}: empty {in_path}", file=sys.stderr)
            continue
        out_file = output_dir / f"{question_type}.json"
        if skip_existing:
            cached = try_load_per_type_metrics_json(output_dir, question_type)
            if cached is not None:
                mean_row, n_cached = cached
                agg_parts.append((question_type, mean_row, n_cached))
                print(
                    f"[skip-existing] {question_type}: reused {out_file} (n={n_cached}); summary={mean_row}",
                    file=sys.stderr,
                )
                continue
        rows = await compute_retrieval_metrics_async(
            raw,
            settings,
            row_concurrency=row_concurrency,
            per_metric_concurrency=per_metric_concurrency,
            atomic_claims_from_gt=atomic_claims_from_gt,
            show_progress=True,
        )
        mean_row = _mean_retrieval_from_rows(rows)
        agg_parts.append((question_type, mean_row, len(rows)))
        payload = {
            "meta": {
                "question_type": question_type,
                "input_file": str(in_path.resolve()),
                "n_rows": len(rows),
                "llm_model": settings.llm_model,
                "atomic_claims_from_gt": atomic_claims_from_gt,
            },
            "summary": _json_safe(mean_row),
            "rows": _json_safe(rows),
        }
        out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"OK: {question_type} → {out_file} ({len(rows)} rows); summary={mean_row}")

    if not agg_parts:
        print("No non-empty type files evaluated.", file=sys.stderr)
        raise SystemExit(1)

    total_n = sum(n for _, _, n in agg_parts)
    agg: dict[str, float | None] = {}
    for k in ("context_relevance", "evidence_recall"):
        num = 0.0
        den = 0
        for _, sm, n in agg_parts:
            v = sm.get(k)
            if v is None:
                continue
            num += float(v) * n
            den += n
        agg[k] = (num / den) if den > 0 else None

    summary_path = output_dir / "_summary.json"
    summary_doc = {
        "meta": {
            "input_dir": str(input_dir.resolve()),
            "output_dir": str(output_dir.resolve()),
            "types": [t for t, _, _ in agg_parts],
            "llm_model": settings.llm_model,
            "atomic_claims_from_gt": atomic_claims_from_gt,
            "skip_existing": skip_existing,
        },
        "by_type": {t: {"n": n, "summary": s} for t, s, n in agg_parts},
        "aggregate_summary": _json_safe(agg),
    }
    summary_path.write_text(json.dumps(summary_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"OK: aggregate → {summary_path}")
    print("Overall averages (weighted by row count):", _json_safe(agg))


async def run_retrieval_single_input(
    input_path: Path,
    output_path: Path,
    settings: RagasJudgeSettings,
    *,
    row_concurrency: int,
    per_metric_concurrency: int,
    atomic_claims_from_gt: bool,
) -> None:
    raw = load_raw_items(input_path)
    if not raw:
        print("Input is empty.", file=sys.stderr)
        raise SystemExit(1)
    rows = await compute_retrieval_metrics_async(
        raw,
        settings,
        row_concurrency=row_concurrency,
        per_metric_concurrency=per_metric_concurrency,
        atomic_claims_from_gt=atomic_claims_from_gt,
        show_progress=True,
    )
    mean_row = _mean_retrieval_from_rows(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "input_file": str(input_path.resolve()),
            "n_rows": len(rows),
            "llm_model": settings.llm_model,
            "atomic_claims_from_gt": atomic_claims_from_gt,
        },
        "summary": _json_safe(mean_row),
        "rows": _json_safe(rows),
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"OK: wrote {output_path} ({len(rows)} rows)")
    print("Overall averages:", _json_safe(mean_row))


async def _demo_async() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Skip demo: OPENAI_API_KEY not set.")
        return

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    ev = GraphRAGEvaluator(api_key=api_key, model=model, max_concurrency=4)

    question = "What year did the Apollo 11 lunar landing occur?"
    evidence = [
        "Apollo 11 landed on the Moon on July 20, 1969.",
        "Neil Armstrong was the first person to walk on the Moon.",
    ]
    contexts = [
        "The Apollo program was NASA's third human spaceflight program.",
        "Apollo 11's lunar module Eagle touched down in the Sea of Tranquillity in July 1969.",
        "Gemini preceded Apollo in NASA's crewed spaceflight timeline.",
    ]

    cr = await ev.evaluate_context_relevance(contexts, question, evidence)
    print(f"Context relevance (mean): {cr:.4f}")

    reference_claims = [
        "Apollo 11 landed humans on the Moon.",
        "The landing happened in 1969.",
        "The mission was led by the Soviet space agency.",
    ]
    er = await ev.evaluate_evidence_recall(reference_claims, contexts)
    print(f"Evidence recall: {er:.4f}")


def main() -> int:
    _load_env()
    p = argparse.ArgumentParser(
        description="Retrieval LLM metrics (context relevance, evidence recall) for benchmark JSON/JSONL",
    )
    p.add_argument("--demo", action="store_true", help="Run a tiny in-memory example (needs OPENAI_API_KEY)")
    p.add_argument("--input", default="", help="Single .json / .jsonl (mutually exclusive with --input-dir)")
    p.add_argument(
        "--input-dir",
        default="",
        help="Directory with per-type files (e.g. vector-rag/results)",
    )
    p.add_argument("--output", default="", help="Output .json when using --input")
    p.add_argument(
        "--output-dir",
        default="",
        help="Output directory for --input-dir (default: <input-dir>/retrieval_metrics)",
    )
    p.add_argument("--api-key", default="", help="Or OPENAI_API_KEY in .env")
    p.add_argument("--base-url", default="")
    p.add_argument("--llm-model", default="")
    p.add_argument("--llm-max-tokens", type=int, default=0)
    p.add_argument("--row-concurrency", type=int, default=4, metavar="N")
    p.add_argument("--per-metric-concurrency", type=int, default=8, metavar="N")
    p.add_argument(
        "--atomic-claims-from-gt",
        action="store_true",
        help="Extract atomic claims from ground_truth via LLM for evidence recall (extra calls).",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="При --input-dir: не пересчитывать тип, если в output-dir уже есть <тип>.json.",
    )
    p.add_argument("--header", action="append", default=[], help='HTTP header "Name: value"')
    args = p.parse_args()

    if args.demo:
        asyncio.run(_demo_async())
        return 0

    has_in = bool(args.input.strip())
    has_dir = bool(args.input_dir.strip())
    if has_in == has_dir:
        print("Specify exactly one of: --input, --input-dir, or use --demo", file=sys.stderr)
        return 2
    if has_in and not args.output.strip():
        print("--output is required with --input", file=sys.stderr)
        return 2

    api_key = (args.api_key or os.environ.get("OPENAI_API_KEY", "")).strip()
    if not api_key:
        print("Need --api-key or OPENAI_API_KEY", file=sys.stderr)
        return 1

    headers: dict[str, str] | None = None
    if args.header:
        headers = {}
        for h in args.header:
            if ":" not in h:
                print(f"Bad --header: {h!r}", file=sys.stderr)
                return 2
            k, v = h.split(":", 1)
            headers[k.strip()] = v.strip()

    llm_model = (args.llm_model or os.environ.get("JUDGE_LLM_MODEL", "qwen/qwen3.6-plus")).strip()
    llm_max = args.llm_max_tokens or int(os.environ.get("JUDGE_LLM_MAX_TOKENS", "4096"))

    settings = RagasJudgeSettings(
        api_key=api_key,
        base_url=args.base_url.strip() or None,
        llm_model=llm_model,
        llm_max_tokens=llm_max,
        default_headers=headers,
    )

    async def _run() -> None:
        if has_dir:
            input_dir = Path(args.input_dir).expanduser().resolve()
            out_root = (args.output_dir or "").strip()
            output_dir = (
                Path(out_root).expanduser().resolve()
                if out_root
                else (input_dir / "retrieval_metrics").resolve()
            )
            await run_retrieval_input_dir(
                input_dir,
                output_dir,
                settings,
                row_concurrency=args.row_concurrency,
                per_metric_concurrency=args.per_metric_concurrency,
                atomic_claims_from_gt=bool(args.atomic_claims_from_gt),
                skip_existing=bool(args.skip_existing),
            )
        else:
            await run_retrieval_single_input(
                Path(args.input).expanduser().resolve(),
                Path(args.output).expanduser().resolve(),
                settings,
                row_concurrency=args.row_concurrency,
                per_metric_concurrency=args.per_metric_concurrency,
                atomic_claims_from_gt=bool(args.atomic_claims_from_gt),
            )

    asyncio.run(_run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
