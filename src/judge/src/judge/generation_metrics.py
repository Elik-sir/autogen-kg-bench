"""LLM-as-a-judge **generation** metrics: Faithfulness and Answer Relevance.

Faithfulness: :math:`|\\mathcal{V}|/|\\mathcal{A}|` — share of atomic claims from the
answer that are supported only by retrieved context.

Answer relevance: single judge score :math:`\\mathrm{Score}(A, Q) \\in [0,1]`.

Uses the same structured-output path as :mod:`judge.retrieval_metrics` (``_llm_parse``).

Batch mode::

    uv run python -m judge.generation_metrics --input-dir ../benchmarks/light-rag/results

Default output: ``<input-dir>/generation_metrics/``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm

from .config import RagasJudgeSettings
from .extended_metrics import load_raw_items
from .io import (
    QUESTION_TYPE_ORDER,
    discover_benchmark_result_files,
    normalize_eval_record,
    try_load_per_type_metrics_json,
)
from .retrieval_metrics import (
    ChunkRelevanceJudgment,
    ClaimSupportJudgment,
    _is_placeholder_contexts,
    _llm_parse,
    _normalize_context_list,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class FaithfulnessClaimsExtraction(BaseModel):
    """Structured output: atomic factual claims from the generated answer (set :math:`\\mathcal{A}`)."""

    claims: list[str] = Field(
        default_factory=list,
        description="Short atomic factual assertions stated in the answer (no duplicates).",
    )


_FAITHFULNESS_EXTRACT_SYSTEM = """You extract atomic factual claims from a RAG system's generated answer.

Rules:
- Each claim is one concise, verifiable factual assertion (who/what/when/where/how much).
- Split compound sentences into multiple claims when they state independent facts.
- Omit opinions, hedging without factual content, and duplicates.
- Do not invent claims; only what the answer explicitly asserts.
- The user question is given only for disambiguation (e.g. what "it" refers to).

If there are no factual claims, return an empty list."""


_FAITHFULNESS_VERIFY_SYSTEM = """You judge support between ONE atomic claim and RETRIEVED CONTEXT only.

The claim was taken from a model answer. Decide if the context alone entails or clearly supports the claim.

Rules:
- Use only the provided context text; no outside or prior knowledge.
- supported=true only if the context clearly backs the claim; weak implication or missing evidence → false.
- Contradiction → false.

Output only the structured fields required by the schema."""


_ANSWER_RELEVANCE_SYSTEM = """You score how well a generated answer responds to the user question.

Focus on usefulness and directness for this question (coverage of what was asked, appropriate scope, not evasive).

Do NOT score factual correctness against the world or against any context — that is a separate concern.

Return a single score from 0.0 (does not address the question / off-topic / useless) to 1.0 (fully addresses the question in a useful way)."""


def _load_env() -> None:
    load_dotenv(_PROJECT_ROOT / ".env", override=False)


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


def _mean_generation_from_rows(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    keys = ("faithfulness", "answer_relevance")
    vals: dict[str, list[float | None]] = {k: [] for k in keys}
    for r in rows:
        for k in keys:
            v = r.get(k)
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                vals[k].append(None)
            else:
                vals[k].append(float(v))
    return {k: _safe_mean(vals[k]) for k in keys}


async def _extract_atomic_claims_for_faithfulness(
    client: AsyncOpenAI,
    *,
    model: str,
    max_tokens: int,
    question: str,
    generated_answer: str,
    extra_body: dict[str, Any] | None,
) -> list[str]:
    text = (generated_answer or "").strip()
    if not text:
        return []
    user = (
        f"USER QUESTION (for disambiguation only):\n{question.strip()}\n\n"
        f"GENERATED ANSWER:\n{text}\n\n"
        "Return JSON with key 'claims' (list of strings), per instructions."
    )
    out = await _llm_parse(
        client,
        model=model,
        max_tokens=max_tokens,
        system=_FAITHFULNESS_EXTRACT_SYSTEM,
        user=user,
        response_model=FaithfulnessClaimsExtraction,
        extra_body=extra_body,
    )
    cleaned: list[str] = []
    for c in out.claims:
        s = str(c).strip()
        if s and s not in cleaned:
            cleaned.append(s)
    return cleaned


async def _verify_answer_claim_supported_by_context(
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
            f"RETRIEVED CONTEXT:\n{context_text.strip()}\n\n"
            f"CLAIM (from generated answer):\n{claim.strip()}\n\n"
            "Is this claim supported solely by the retrieved context?"
        )
        out = await _llm_parse(
            client,
            model=model,
            max_tokens=max_tokens,
            system=_FAITHFULNESS_VERIFY_SYSTEM,
            user=user,
            response_model=ClaimSupportJudgment,
            extra_body=extra_body,
        )
        return 1.0 if out.supported else 0.0


async def _faithfulness_ratio_for_claims(
    client: AsyncOpenAI,
    *,
    model: str,
    max_tokens: int,
    claims: list[str],
    context_text: str,
    max_concurrency: int,
    extra_body: dict[str, Any] | None,
) -> float:
    """Return the fraction of ``claims`` supported by ``context_text`` (:math:`|\\mathcal{V}|/|\\mathcal{A}|`)."""
    if not claims:
        return 0.0
    if not (context_text or "").strip():
        return 0.0
    sem = asyncio.Semaphore(max(1, max_concurrency))
    flags = await asyncio.gather(
        *(
            _verify_answer_claim_supported_by_context(
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
    return sum(flags) / len(flags)


async def evaluate_faithfulness(
    question: str,
    generated_answer: str,
    retrieved_context: str | list[str],
    *,
    client: AsyncOpenAI,
    model: str,
    max_tokens: int = 1024,
    max_concurrency: int = 8,
    extra_body: dict[str, Any] | None = None,
) -> float:
    r"""
    Faithfulness :math:`|\mathcal{V}|/|\mathcal{A}|`.

    Step 1: LLM extracts atomic claims :math:`\mathcal{A}` from ``generated_answer``.
    Step 2: For each claim, LLM checks if retrieved context alone supports it
    (members of :math:`\mathcal{V}`).

    Returns ``0.0`` if there are no claims (empty or non-factual answer).
    """
    claims = await _extract_atomic_claims_for_faithfulness(
        client,
        model=model,
        max_tokens=max_tokens,
        question=question,
        generated_answer=generated_answer,
        extra_body=extra_body,
    )
    context_text = _normalize_context_list(retrieved_context)
    return await _faithfulness_ratio_for_claims(
        client,
        model=model,
        max_tokens=max_tokens,
        claims=claims,
        context_text=context_text,
        max_concurrency=max_concurrency,
        extra_body=extra_body,
    )


async def evaluate_answer_relevance(
    question: str,
    generated_answer: str,
    *,
    client: AsyncOpenAI,
    model: str,
    max_tokens: int = 1024,
    extra_body: dict[str, Any] | None = None,
) -> float:
    r"""
    Answer relevance :math:`\mathrm{Score}(A, Q) \in [0,1]` — LLM judge only.

    Ignores factual alignment with external truth or context; measures whether the
    answer usefully addresses the question.
    """
    ans = (generated_answer or "").strip()
    if not ans:
        return 0.0
    user = (
        f"USER QUESTION:\n{question.strip()}\n\n"
        f"GENERATED ANSWER:\n{ans}\n\n"
        "Return your score for how well this answer addresses the question."
    )
    out = await _llm_parse(
        client,
        model=model,
        max_tokens=max_tokens,
        system=_ANSWER_RELEVANCE_SYSTEM,
        user=user,
        response_model=ChunkRelevanceJudgment,
        extra_body=extra_body,
    )
    return float(out.score)


class GraphRAGGenerationEvaluator:
    """Wrapper for faithfulness + answer-relevance (same pattern as :class:`GraphRAGEvaluator`)."""

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

    async def evaluate_faithfulness(
        self,
        question: str,
        generated_answer: str,
        retrieved_context: str | list[str],
    ) -> float:
        return await evaluate_faithfulness(
            question,
            generated_answer,
            retrieved_context,
            client=self._client,
            model=self._model,
            max_tokens=self._max_tokens,
            max_concurrency=self._max_concurrency,
            extra_body=self._extra_body,
        )

    async def evaluate_answer_relevance(self, question: str, generated_answer: str) -> float:
        return await evaluate_answer_relevance(
            question,
            generated_answer,
            client=self._client,
            model=self._model,
            max_tokens=self._max_tokens,
            extra_body=self._extra_body,
        )


async def _process_one_generation_row(
    client: AsyncOpenAI,
    *,
    model: str,
    max_tokens: int,
    extra_body: dict[str, Any] | None,
    per_metric_concurrency: int,
    row_sem: asyncio.Semaphore,
    raw_row: dict[str, Any],
    norm: dict[str, Any] | None,
    store_claims: bool,
) -> dict[str, Any]:
    if not norm:
        return {
            **raw_row,
            "faithfulness": None,
            "answer_relevance": None,
            "generation_metrics_note": "invalid_row",
        }

    question = str(norm["question"])
    answer = str(norm.get("answer") or "")
    ctxs = list(norm.get("contexts") or [])

    async with row_sem:
        ar = await evaluate_answer_relevance(
            question,
            answer,
            client=client,
            model=model,
            max_tokens=max_tokens,
            extra_body=extra_body,
        )

        if _is_placeholder_contexts(ctxs):
            out: dict[str, Any] = {
                **raw_row,
                "faithfulness": None,
                "answer_relevance": ar,
                "generation_metrics_note": "missing_or_placeholder_contexts",
            }
            return out

        ctx_text = _normalize_context_list(ctxs)
        if store_claims:
            claims = await _extract_atomic_claims_for_faithfulness(
                client,
                model=model,
                max_tokens=max_tokens,
                question=question,
                generated_answer=answer,
                extra_body=extra_body,
            )
            ff = await _faithfulness_ratio_for_claims(
                client,
                model=model,
                max_tokens=max_tokens,
                claims=claims,
                context_text=ctx_text,
                max_concurrency=per_metric_concurrency,
                extra_body=extra_body,
            )
            merged = {**raw_row, "faithfulness": ff, "answer_relevance": ar, "faithfulness_claims_used": claims}
        else:
            ff = await evaluate_faithfulness(
                question,
                answer,
                ctxs,
                client=client,
                model=model,
                max_tokens=max_tokens,
                max_concurrency=per_metric_concurrency,
                extra_body=extra_body,
            )
            merged = {**raw_row, "faithfulness": ff, "answer_relevance": ar}
        return merged


async def compute_generation_metrics_async(
    rows: list[dict[str, Any]],
    settings: RagasJudgeSettings,
    *,
    row_concurrency: int = 4,
    per_metric_concurrency: int = 8,
    store_claims: bool = False,
    show_progress: bool = True,
) -> list[dict[str, Any]]:
    """Merge ``faithfulness`` and ``answer_relevance`` into each raw benchmark row."""
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
        return await _process_one_generation_row(
            client,
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            extra_body=settings.openai_chat_extra_body,
            per_metric_concurrency=per_metric_concurrency,
            row_sem=row_sem,
            raw_row=dict(rows[i]),
            norm=norms[i],
            store_claims=store_claims,
        )

    tasks = [_one(i) for i in range(len(rows))]
    if show_progress:
        return list(await tqdm.gather(*tasks, desc="Generation LLM metrics"))
    return list(await asyncio.gather(*tasks))


async def run_generation_input_dir(
    input_dir: Path,
    output_dir: Path,
    settings: RagasJudgeSettings,
    *,
    row_concurrency: int,
    per_metric_concurrency: int,
    store_claims: bool,
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
        rows = await compute_generation_metrics_async(
            raw,
            settings,
            row_concurrency=row_concurrency,
            per_metric_concurrency=per_metric_concurrency,
            store_claims=store_claims,
            show_progress=True,
        )
        mean_row = _mean_generation_from_rows(rows)
        agg_parts.append((question_type, mean_row, len(rows)))
        payload = {
            "meta": {
                "question_type": question_type,
                "input_file": str(in_path.resolve()),
                "n_rows": len(rows),
                "llm_model": settings.llm_model,
                "store_claims": store_claims,
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
    for k in ("faithfulness", "answer_relevance"):
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
            "skip_existing": skip_existing,
        },
        "by_type": {t: {"n": n, "summary": s} for t, s, n in agg_parts},
        "aggregate_summary": _json_safe(agg),
    }
    summary_path.write_text(json.dumps(summary_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"OK: aggregate → {summary_path}")
    print("Overall averages (weighted by row count):", _json_safe(agg))


async def run_generation_single_input(
    input_path: Path,
    output_path: Path,
    settings: RagasJudgeSettings,
    *,
    row_concurrency: int,
    per_metric_concurrency: int,
    store_claims: bool,
) -> None:
    raw = load_raw_items(input_path)
    if not raw:
        print("Input is empty.", file=sys.stderr)
        raise SystemExit(1)
    rows = await compute_generation_metrics_async(
        raw,
        settings,
        row_concurrency=row_concurrency,
        per_metric_concurrency=per_metric_concurrency,
        store_claims=store_claims,
        show_progress=True,
    )
    mean_row = _mean_generation_from_rows(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "input_file": str(input_path.resolve()),
            "n_rows": len(rows),
            "llm_model": settings.llm_model,
            "store_claims": store_claims,
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
    ev = GraphRAGGenerationEvaluator(api_key=api_key, model=model, max_concurrency=4)
    q = "What is the capital of France?"
    ans = "Paris is the capital. The Eiffel Tower is in London."
    ctx = ["Paris is the capital city of France.", "The Eiffel Tower is located in Paris, France."]
    ff = await ev.evaluate_faithfulness(q, ans, ctx)
    ar = await ev.evaluate_answer_relevance(q, ans)
    print(f"Faithfulness: {ff:.4f}  Answer relevance: {ar:.4f}")


def main() -> int:
    _load_env()
    p = argparse.ArgumentParser(
        description="Generation LLM metrics (faithfulness, answer relevance) for benchmark JSON/JSONL",
    )
    p.add_argument("--demo", action="store_true")
    p.add_argument("--input", default="", help="Single .json / .jsonl")
    p.add_argument("--input-dir", default="", help="Directory with per-type files (e.g. light-rag/results)")
    p.add_argument("--output", default="", help="Output .json with --input")
    p.add_argument(
        "--output-dir",
        default="",
        help="Output directory for --input-dir (default: <input-dir>/generation_metrics)",
    )
    p.add_argument(
        "--store-claims",
        action="store_true",
        help="Include faithfulness_claims_used in each output row (same claim list as faithfulness; no duplicate extract).",
    )
    p.add_argument("--api-key", default="")
    p.add_argument("--base-url", default="")
    p.add_argument("--llm-model", default="")
    p.add_argument("--llm-max-tokens", type=int, default=0)
    p.add_argument("--row-concurrency", type=int, default=4, metavar="N")
    p.add_argument("--per-metric-concurrency", type=int, default=8, metavar="N")
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
        print("Specify exactly one of: --input, --input-dir, or --demo", file=sys.stderr)
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
                else (input_dir / "generation_metrics").resolve()
            )
            await run_generation_input_dir(
                input_dir,
                output_dir,
                settings,
                row_concurrency=args.row_concurrency,
                per_metric_concurrency=args.per_metric_concurrency,
                store_claims=bool(args.store_claims),
                skip_existing=bool(args.skip_existing),
            )
        else:
            await run_generation_single_input(
                Path(args.input).expanduser().resolve(),
                Path(args.output).expanduser().resolve(),
                settings,
                row_concurrency=args.row_concurrency,
                per_metric_concurrency=args.per_metric_concurrency,
                store_claims=bool(args.store_claims),
            )

    asyncio.run(_run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
