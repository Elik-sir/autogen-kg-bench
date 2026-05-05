from __future__ import annotations

from typing import Any

from utils.hippo_extract import extract_answer_from_result, extract_contexts_from_rag_qa  # noqa: WPS433
from utils.metrics import recall_overlap  # noqa: WPS433


def run_benchmark_chunk(rag: Any, chunk: list[dict]) -> list[dict]:
    """Синхронный прогон списка вопросов через ``rag.rag_qa``."""
    rows: list[dict] = []
    n = len(chunk)
    qa_top_k = int(getattr(rag.global_config, "qa_top_k", 5))
    for i, it in enumerate(chunk, 1):
        q = str(it.get("question", ""))
        ground_truth = it.get("ground_truth")
        complexity = str(it.get("complexity", "") or "")
        is_subgraph_deep = complexity == "subgraph-deep-analytics"
        benchmark_answer = str(it.get("answer") or "").strip()
        reference = (
            benchmark_answer
            if is_subgraph_deep
            else (str(ground_truth).strip() if ground_truth is not None else "")
        )
        try:
            qa_raw = rag.rag_qa(queries=[q])
            answer = extract_answer_from_result(qa_raw)
            contexts = extract_contexts_from_rag_qa(qa_raw, qa_top_k)
        except Exception as e:  # noqa: BLE001
            answer = f"[error] {e}"
            contexts = []

        row = {
            "index": i,
            "complexity": complexity,
            "scoring_reference": "answer" if is_subgraph_deep else "ground_truth",
            "recall_on_ground_truth_tokens": round(recall_overlap(reference, str(answer)), 4),
            "question": q,
            "ground_truth": ground_truth,
            "answer": answer,
            "contexts": contexts,
        }
        if is_subgraph_deep:
            row["ideal_for_scoring"] = benchmark_answer
        rows.append(row)
        sc = row["recall_on_ground_truth_tokens"]
        ref_tag = row["scoring_reference"]
        print(
            f"  [{i}/{n}] recall@{ref_tag}={sc:.3f}  {q[:70]}…"
            if len(q) > 70
            else f"  [{i}/{n}] recall@{ref_tag}={sc:.3f}  {q}"
        )
    return rows
