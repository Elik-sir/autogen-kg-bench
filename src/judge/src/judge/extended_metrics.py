# Requirements (add to pyproject or pip install):
#   rouge-score>=0.1.2
#   sentence-transformers>=3.0.0
#   openai>=2.32.0
#   pydantic>=2.0
#   numpy>=1.26.0
#   tqdm>=4.66.0
#   python-dotenv>=1.0.0
#
# Run from ``src/judge``::
#
#   uv sync
#   uv run python -m judge.extended_metrics --input-dir ../benchmarks/vector-rag/results --output-dir ../benchmarks/vector-rag/results/extended_metrics
#
#   uv run python -m judge.extended_metrics --input data/sample_ragas_input.json --output out_extended.json

"""ROUGE-L, semantic similarity, LLM-based FC / FS / Cov, and Answer Accuracy (AC)."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Any, TypeVar

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm

from .config import RagasJudgeSettings
from .io import (
    QUESTION_TYPE_ORDER,
    discover_benchmark_result_files,
    normalize_eval_record,
    try_load_per_type_metrics_json,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

T = TypeVar("T", bound=BaseModel)


def _load_env() -> None:
    load_dotenv(_PROJECT_ROOT / ".env", override=False)


class ClaimsExtraction(BaseModel):
    """Structured output: atomic factual claims."""

    claims: list[str] = Field(default_factory=list, description="Atomic factual claims in the text.")


class ClaimSupportRow(BaseModel):
    claim: str
    supported: bool


class AnswerSupportsBatch(BaseModel):
    """Per-claim support of answer claims against ground-truth context."""

    items: list[ClaimSupportRow] = Field(default_factory=list)


class EvidenceCoverageRow(BaseModel):
    evidence: str
    covered: bool


class GroundTruthCoveragesBatch(BaseModel):
    """Per-evidence coverage of ground-truth claims in the answer."""

    items: list[EvidenceCoverageRow] = Field(default_factory=list)


def _rouge_l_f1(reference: str, candidate: str) -> float | None:
    from rouge_score import rouge_scorer

    ref = (reference or "").strip()
    cand = (candidate or "").strip()
    if not ref and not cand:
        return 1.0
    if not ref or not cand:
        return 0.0
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return float(scorer.score(ref, cand)["rougeL"].fmeasure)


def _cosine_ss(emb_a: Any, emb_b: Any) -> float:
    import numpy as np

    a = np.asarray(emb_a, dtype=np.float64).ravel()
    b = np.asarray(emb_b, dtype=np.float64).ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _make_sentence_model(model_name: str) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


async def _llm_parse(
    client: AsyncOpenAI,
    *,
    model: str,
    max_tokens: int,
    system: str,
    user: str,
    response_model: type[T],
) -> T:
    """Structured output: prefer native parse, fall back to JSON schema / json_object."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        resp = await client.chat.completions.parse(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            response_format=response_model,
        )
        parsed = resp.choices[0].message.parsed
        if parsed is not None:
            return parsed
    except Exception:
        pass

    schema_hint = response_model.model_json_schema()
    resp3 = await client.chat.completions.create(
        model=model,
        messages=messages
        + [
            {
                "role": "user",
                "content": (
                    "Reply with a single JSON object only (no markdown), matching this JSON Schema:\n"
                    + json.dumps(schema_hint, ensure_ascii=False)
                ),
            }
        ],
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    raw3 = (resp3.choices[0].message.content or "").strip()
    return response_model.model_validate_json(raw3)


async def extract_claims(
    client: AsyncOpenAI,
    *,
    model: str,
    max_tokens: int,
    text: str,
) -> list[str]:
    if not (text or "").strip():
        return []
    system = (
        "You extract atomic factual claims from text. "
        "Each claim is one verifiable fact (no duplicates). "
        "If there are no factual claims, return an empty list."
    )
    user = f"Text:\n\n{text.strip()}\n\nReturn JSON with key 'claims' (list of strings)."
    out = await _llm_parse(
        client,
        model=model,
        max_tokens=max_tokens,
        system=system,
        user=user,
        response_model=ClaimsExtraction,
    )
    cleaned: list[str] = []
    for c in out.claims:
        s = str(c).strip()
        if s and s not in cleaned:
            cleaned.append(s)
    return cleaned


async def verify_answer_claims_against_context(
    client: AsyncOpenAI,
    *,
    model: str,
    max_tokens: int,
    claims: list[str],
    context: str,
) -> list[bool]:
    """S(c, C): each answer claim supported by context (ground_truth)."""
    if not claims:
        return []
    system = (
        "For each claim, decide if it is fully supported by the given context. "
        "Supported means the context entails or clearly states the claim; "
        "partial speculation counts as not supported."
    )
    lines = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
    user = (
        f"Context:\n\n{context.strip()}\n\n"
        f"Claims:\n{lines}\n\n"
        "Return JSON object with key 'items': list of {{\"claim\": str, \"supported\": bool}} "
        "in the same order as the claims (claim text must match)."
    )
    out = await _llm_parse(
        client,
        model=model,
        max_tokens=max_tokens,
        system=system,
        user=user,
        response_model=AnswerSupportsBatch,
    )
    boo: list[bool] = []
    for i, c in enumerate(claims):
        if i < len(out.items) and out.items[i].claim.strip() == c.strip():
            boo.append(bool(out.items[i].supported))
        else:
            # order mismatch: match by claim text
            matched = next((x.supported for x in out.items if x.claim.strip() == c.strip()), False)
            boo.append(bool(matched))
    while len(boo) < len(claims):
        boo.append(False)
    return boo[: len(claims)]


async def verify_ground_truth_evidences_in_answer(
    client: AsyncOpenAI,
    *,
    model: str,
    max_tokens: int,
    evidences: list[str],
    answer: str,
) -> list[bool]:
    """M(e, G): each ground-truth evidence appears or is conveyed in the answer."""
    if not evidences:
        return []
    system = (
        "For each reference evidence (atomic fact from the reference), decide if the answer "
        "covers or conveys that fact (paraphrase allowed). "
        "If the answer is silent or contradicts it, covered is false."
    )
    lines = "\n".join(f"{i + 1}. {e}" for i, e in enumerate(evidences))
    user = (
        f"Answer:\n\n{answer.strip()}\n\n"
        f"Reference evidences:\n{lines}\n\n"
        "Return JSON with key 'items': list of {{\"evidence\": str, \"covered\": bool}} "
        "in the same order."
    )
    out = await _llm_parse(
        client,
        model=model,
        max_tokens=max_tokens,
        system=system,
        user=user,
        response_model=GroundTruthCoveragesBatch,
    )
    boo: list[bool] = []
    for i, e in enumerate(evidences):
        if i < len(out.items) and out.items[i].evidence.strip() == e.strip():
            boo.append(bool(out.items[i].covered))
        else:
            matched = next((x.covered for x in out.items if x.evidence.strip() == e.strip()), False)
            boo.append(bool(matched))
    while len(boo) < len(evidences):
        boo.append(False)
    return boo[: len(evidences)]


def _fc_from_counts(tp: int, fp: int, fn: int) -> float | None:
    d = tp + fp + fn
    if d <= 0:
        return None
    return 2.0 * tp / d


def _safe_mean(values: list[float | None]) -> float | None:
    xs = [float(x) for x in values if x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))]
    if not xs:
        return None
    return sum(xs) / len(xs)


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def load_raw_jsonl_items(path: Path) -> list[dict[str, Any]]:
    """Load JSONL benchmark rows (skip ``kind: summary``), preserve original keys."""
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            continue
        if row.get("kind") == "summary":
            continue
        if row.get("kind") == "item" or "question" in row:
            out.append(dict(row))
    return out


def load_raw_json_items(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "items" in raw:
        raw = raw["items"]
    if not isinstance(raw, list):
        raise ValueError("JSON root must be a list or object with 'items'")
    return [dict(x) for x in raw if isinstance(x, dict)]


def load_raw_items(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(str(p))
    if p.suffix.lower() == ".jsonl":
        return load_raw_jsonl_items(p)
    return load_raw_json_items(p)


async def _process_one_row(
    client: AsyncOpenAI,
    *,
    model: str,
    max_tokens: int,
    sem: asyncio.Semaphore,
    row_norm: dict[str, Any],
    rouge_l: float | None,
    ss: float | None,
    alpha: float,
) -> dict[str, float | None]:
    """LLM metrics for a single row (rouge/ss precomputed)."""
    answer = str(row_norm.get("answer") or "")
    gt = row_norm.get("ground_truth")
    gt_s = str(gt).strip() if gt is not None else ""

    metrics: dict[str, float | None] = {
        "rouge_l": rouge_l,
        "ss": ss,
        "fc": None,
        "ac": None,
        "fs": None,
        "cov": None,
    }

    if not gt_s:
        return metrics

    async with sem:
        claims_a, claims_g = await asyncio.gather(
            extract_claims(client, model=model, max_tokens=max_tokens, text=answer),
            extract_claims(client, model=model, max_tokens=max_tokens, text=gt_s),
        )
        sup_flags, cov_flags = await asyncio.gather(
            verify_answer_claims_against_context(
                client,
                model=model,
                max_tokens=max_tokens,
                claims=claims_a,
                context=gt_s,
            ),
            verify_ground_truth_evidences_in_answer(
                client,
                model=model,
                max_tokens=max_tokens,
                evidences=claims_g,
                answer=answer,
            ),
        )

    tp = sum(1 for s in sup_flags if s)
    fp = sum(1 for s in sup_flags if not s)
    fn = sum(1 for c in cov_flags if not c)
    fc = _fc_from_counts(tp, fp, fn)

    if claims_a:
        fs = sum(1 for s in sup_flags if s) / len(claims_a)
    else:
        fs = None

    if claims_g:
        cov = sum(1 for c in cov_flags if c) / len(claims_g)
    else:
        cov = None

    metrics["fc"] = fc
    metrics["fs"] = fs
    metrics["cov"] = cov

    if fc is not None and ss is not None:
        metrics["ac"] = alpha * fc + (1.0 - alpha) * ss
    elif fc is not None:
        metrics["ac"] = fc
    elif ss is not None:
        metrics["ac"] = ss
    else:
        metrics["ac"] = None

    return metrics


async def compute_extended_metrics_async(
    rows: list[dict[str, Any]],
    settings: RagasJudgeSettings,
    *,
    sentence_transformer_model: str = "all-MiniLM-L6-v2",
    alpha: float = 0.5,
    llm_concurrency: int = 8,
    show_progress: bool = True,
) -> list[dict[str, Any]]:
    """Return each input row merged with ``rouge_l``, ``ss``, ``fc``, ``ac``, ``fs``, ``cov``."""
    if not rows:
        return []

    norms: list[dict[str, Any] | None] = []
    for r in rows:
        try:
            norms.append(normalize_eval_record(dict(r)))
        except ValueError:
            norms.append(None)

    client = AsyncOpenAI(**settings.openai_client_kwargs())
    sem = asyncio.Semaphore(max(1, llm_concurrency))

    # ROUGE + embeddings (sync, batched)
    scorer_model = _make_sentence_model(sentence_transformer_model)
    texts_a: list[str] = []
    texts_g: list[str] = []
    rouge_vals: list[float | None] = []
    for n in norms:
        if not n:
            texts_a.append("")
            texts_g.append("")
            rouge_vals.append(None)
            continue
        a = str(n.get("answer") or "")
        g = n.get("ground_truth")
        g_s = str(g).strip() if g is not None else ""
        texts_a.append(a)
        texts_g.append(g_s)
        rouge_vals.append(_rouge_l_f1(g_s, a) if g_s else None)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        emb_a = scorer_model.encode(texts_a, convert_to_numpy=True, show_progress_bar=False)
        emb_g = scorer_model.encode(texts_g, convert_to_numpy=True, show_progress_bar=False)

    ss_vals: list[float | None] = []
    for i in range(len(rows)):
        if not texts_g[i].strip():
            ss_vals.append(None)
        else:
            ss_vals.append(_cosine_ss(emb_a[i], emb_g[i]))

    async def _one(i: int) -> dict[str, float | None]:
        n = norms[i]
        if not n:
            return {
                "rouge_l": rouge_vals[i],
                "ss": ss_vals[i],
                "fc": None,
                "ac": None,
                "fs": None,
                "cov": None,
            }
        return await _process_one_row(
            client,
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            sem=sem,
            row_norm=n,
            rouge_l=rouge_vals[i],
            ss=ss_vals[i],
            alpha=alpha,
        )

    tasks = [_one(i) for i in range(len(rows))]

    if show_progress:
        llm_metrics = await tqdm.gather(*tasks, desc="LLM metrics")
    else:
        llm_metrics = await asyncio.gather(*tasks)

    out_rows: list[dict[str, Any]] = []
    for raw, m in zip(rows, llm_metrics, strict=True):
        merged = {**raw, **m}
        out_rows.append(merged)
    return out_rows


def _mean_from_row_metrics(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    keys = ("rouge_l", "ss", "fc", "ac", "fs", "cov")
    vals: dict[str, list[float | None]] = {k: [] for k in keys}
    for r in rows:
        for k in keys:
            v = r.get(k)
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                vals[k].append(None)
            else:
                vals[k].append(float(v))
    return {k: _safe_mean(vals[k]) for k in keys}


async def run_input_dir(
    input_dir: Path,
    output_dir: Path,
    settings: RagasJudgeSettings,
    *,
    sentence_transformer_model: str,
    alpha: float,
    llm_concurrency: int,
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
        rows = await compute_extended_metrics_async(
            raw,
            settings,
            sentence_transformer_model=sentence_transformer_model,
            alpha=alpha,
            llm_concurrency=llm_concurrency,
        )
        mean_row = _mean_from_row_metrics(rows)
        agg_parts.append((question_type, mean_row, len(rows)))
        payload = {
            "meta": {
                "question_type": question_type,
                "input_file": str(in_path.resolve()),
                "n_rows": len(rows),
                "llm_model": settings.llm_model,
                "sentence_transformer_model": sentence_transformer_model,
                "alpha": alpha,
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
    for k in ("rouge_l", "ss", "fc", "ac", "fs", "cov"):
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
            "sentence_transformer_model": sentence_transformer_model,
            "alpha": alpha,
            "llm_model": settings.llm_model,
            "skip_existing": skip_existing,
        },
        "by_type": {t: {"n": n, "summary": s} for t, s, n in agg_parts},
        "aggregate_summary": _json_safe(agg),
    }
    summary_path.write_text(json.dumps(summary_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"OK: aggregate → {summary_path}")
    print("Overall averages (weighted by row count):", _json_safe(agg))


async def run_single_input(
    input_path: Path,
    output_path: Path,
    settings: RagasJudgeSettings,
    *,
    sentence_transformer_model: str,
    alpha: float,
    llm_concurrency: int,
) -> None:
    raw = load_raw_items(input_path)
    if not raw:
        print("Input is empty.", file=sys.stderr)
        raise SystemExit(1)
    rows = await compute_extended_metrics_async(
        raw,
        settings,
        sentence_transformer_model=sentence_transformer_model,
        alpha=alpha,
        llm_concurrency=llm_concurrency,
    )
    mean_row = _mean_from_row_metrics(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "input_file": str(input_path.resolve()),
            "n_rows": len(rows),
            "llm_model": settings.llm_model,
            "sentence_transformer_model": sentence_transformer_model,
            "alpha": alpha,
        },
        "summary": _json_safe(mean_row),
        "rows": _json_safe(rows),
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"OK: wrote {output_path} ({len(rows)} rows)")
    print("Overall averages:", _json_safe(mean_row))


def main() -> int:
    _load_env()
    p = argparse.ArgumentParser(description="Extended RAG metrics (ROUGE-L, SS, FC, AC, FS, Cov)")
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
        help="Output directory for --input-dir (default: <input-dir>/extended_metrics)",
    )
    p.add_argument("--api-key", default="", help="Or OPENAI_API_KEY in .env")
    p.add_argument("--base-url", default="")
    p.add_argument("--llm-model", default="")
    p.add_argument("--llm-max-tokens", type=int, default=0)
    p.add_argument("--sentence-transformer-model", default="all-MiniLM-L6-v2")
    p.add_argument("--alpha", type=float, default=0.5, help="AC = alpha*FC + (1-alpha)*SS")
    p.add_argument("--llm-concurrency", type=int, default=8, metavar="N")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="При --input-dir: не пересчитывать тип, если в output-dir уже есть <тип>.json.",
    )
    p.add_argument("--header", action="append", default=[], help='HTTP header "Name: value"')
    args = p.parse_args()

    has_in = bool(args.input.strip())
    has_dir = bool(args.input_dir.strip())
    if has_in == has_dir:
        print("Specify exactly one of: --input or --input-dir", file=sys.stderr)
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
            output_dir = Path(out_root).expanduser().resolve() if out_root else (input_dir / "extended_metrics").resolve()
            await run_input_dir(
                input_dir,
                output_dir,
                settings,
                sentence_transformer_model=args.sentence_transformer_model,
                alpha=float(args.alpha),
                llm_concurrency=args.llm_concurrency,
                skip_existing=bool(args.skip_existing),
            )
        else:
            await run_single_input(
                Path(args.input).expanduser().resolve(),
                Path(args.output).expanduser().resolve(),
                settings,
                sentence_transformer_model=args.sentence_transformer_model,
                alpha=float(args.alpha),
                llm_concurrency=args.llm_concurrency,
            )

    asyncio.run(_run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
