from __future__ import annotations


def print_question_progress(
    rdict: dict,
    type_label: str | None,
    total: int,
    *,
    completed: int | None = None,
) -> None:
    q = str(rdict.get("question", ""))
    i = int(rdict.get("index", 0))
    sc = float(rdict.get("recall_on_ground_truth_tokens", 0.0))
    tag = f"[{type_label}] " if type_label else ""
    prog = f"{completed}/{total}" if completed is not None else f"{i}/{total}"
    if len(q) > 70:
        print(f"  {tag}[{prog}] idx={i} recall={sc:.3f}  {q[:70]}…")
    else:
        print(f"  {tag}[{prog}] idx={i} recall={sc:.3f}  {q}")
