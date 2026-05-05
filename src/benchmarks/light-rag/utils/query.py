from __future__ import annotations

import asyncio

import settings  # noqa: WPS433

from utils.metrics import recall_overlap  # noqa: WPS433 — локальный пакет light-rag/utils


def contexts_from_aquery_llm(full: dict) -> list[str]:
    """Тексты чанков из `LightRAG.aquery_llm` (`data.chunks[].content`) для RAGAS."""
    data = full.get("data")
    if not isinstance(data, dict):
        return []
    chunks = data.get("chunks") or []
    if not isinstance(chunks, list):
        return []
    out: list[str] = []
    for ch in chunks:
        if not isinstance(ch, dict):
            continue
        raw = ch.get("content")
        if raw is None:
            continue
        s = str(raw).strip()
        if s:
            out.append(s)
    return out


def answer_from_aquery_llm(full: dict) -> str:
    lr = full.get("llm_response") or {}
    if lr.get("is_streaming"):
        return ""
    c = lr.get("content")
    return "" if c is None else str(c)


def benchmark_query_param(mode: str):
    """Параметры запроса: те же лимиты, что режут контекст для ответа и для `contexts`."""
    from lightrag import QueryParam  # noqa: WPS433

    return QueryParam(
        mode=mode,
        stream=False,
        top_k=int(settings.QUERY_TOP_K),
        chunk_top_k=int(settings.QUERY_CHUNK_TOP_K),
        max_entity_tokens=int(settings.QUERY_MAX_ENTITY_TOKENS),
        max_relation_tokens=int(settings.QUERY_MAX_RELATION_TOKENS),
        max_total_tokens=int(settings.QUERY_MAX_TOTAL_TOKENS),
    )


async def run_single_benchmark_query(
    rag: object,
    qp: object,
    semaphore: asyncio.Semaphore,
    print_lock: asyncio.Lock,
    *,
    index: int,
    item: dict,
    n_items: int,
) -> dict:
    """Один вопрос: `aquery_llm` под семафором, печать прогресса под lock."""
    q = item.get("question", "")
    ground_truth = item.get("ground_truth")
    complexity = item.get("complexity", "")
    is_subgraph_deep = complexity == "subgraph-deep-analytics"
    benchmark_answer = str(item.get("answer") or "").strip()
    reference = (
        benchmark_answer
        if is_subgraph_deep
        else (str(ground_truth).strip() if ground_truth is not None else "")
    )

    rag_contexts: list[str] = []
    rag_answer = ""
    async with semaphore:
        try:
            full = await rag.aquery_llm(q, param=qp)
            rag_answer = answer_from_aquery_llm(full)
            rag_contexts = contexts_from_aquery_llm(full)
        except Exception as e:  # noqa: BLE001
            rag_answer = f"[error] {e}"

    rdict = {
        "index": index,
        "complexity": complexity,
        "scoring_reference": "answer" if is_subgraph_deep else "ground_truth",
        "recall_on_ground_truth_tokens": round(
            recall_overlap(reference, str(rag_answer)), 4
        ),
        "question": q,
        "ground_truth": ground_truth,
        "answer": rag_answer,
        "contexts": rag_contexts,
    }
    if is_subgraph_deep:
        rdict["ideal_for_scoring"] = benchmark_answer

    sc = rdict["recall_on_ground_truth_tokens"]
    ref_tag = rdict["scoring_reference"]
    line = (
        f"  [{index}/{n_items}] recall@{ref_tag}={sc:.3f}  {q[:70]}…"
        if len(q) > 70
        else f"  [{index}/{n_items}] recall@{ref_tag}={sc:.3f}  {q}"
    )
    async with print_lock:
        print(line, flush=True)

    return rdict
