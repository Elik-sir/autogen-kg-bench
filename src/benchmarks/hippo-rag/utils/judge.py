from __future__ import annotations


def ideal_for_llm_judge(row: dict) -> str:
    """Эталон для LLM-as-judge: как в бенчмарке (ideal_for_scoring vs ground_truth)."""
    if row.get("scoring_reference") == "answer":
        s = row.get("ideal_for_scoring")
        return str(s).strip() if s is not None else ""
    gt = row.get("ground_truth")
    if gt is None:
        return ""
    return str(gt).strip()


# Совместимость с вызовами вида ``from bench_utils import _ideal_for_llm_judge``
_ideal_for_llm_judge = ideal_for_llm_judge
