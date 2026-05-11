from __future__ import annotations

import json
import os
import re

from benchmark_generator.answer_builder import build_answer_from_context
from benchmark_generator.dedup import is_near_duplicate_question, normalize_question_text
from benchmark_generator.query_filters import append_limit_if_missing, should_skip_low_signal_result
from benchmark_generator.utils.benchmark_validation import (
    is_trivial_self_return,
    result_to_ground_truth,
)

GROUND_TRUTH_ROW_LIMIT = max(1, int(os.getenv("BENCHMARK_GROUND_TRUTH_LIMIT", "10")))


def _build_deterministic_answer_for_multi_hop(item: dict) -> str:
    ground_truth = str(item.get("ground_truth", "")).strip()
    if not ground_truth:
        return ""
    values = [part.strip() for part in ground_truth.split(";") if part.strip()]
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    if not deduped:
        return ""
    return "; ".join(deduped)


def _normalize_text_for_leak_check(text: str) -> str:
    normalized = str(text or "").strip().lower()
    normalized = re.sub(r"[^\w\s]", " ", normalized, flags=re.UNICODE)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _candidate_answer_fragments(text: str) -> list[str]:
    raw = str(text or "").strip()
    if not raw:
        return []
    parts = re.split(r"[;\n|]+", raw)
    out: list[str] = []
    for part in parts:
        cleaned = str(part).strip()
        if not cleaned:
            continue
        normalized = _normalize_text_for_leak_check(cleaned)
        if len(normalized) < 4:
            continue
        if normalized.isdigit():
            continue
        out.append(normalized)
    return out


def _question_contains_answer_leak(question: str, ground_truth: str, answer: str) -> bool:
    q_norm = _normalize_text_for_leak_check(question)
    if not q_norm:
        return False
    candidates = [
        *_candidate_answer_fragments(ground_truth),
        *_candidate_answer_fragments(answer),
    ]
    # Preserve order while deduplicating.
    seen: set[str] = set()
    unique_candidates: list[str] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)
    for candidate in unique_candidates:
        # Слишком длинные фрагменты (описания компаний) часто шумные; проверяем только
        # короткие/средние сущности, которые обычно и являются "утечкой ответа".
        if len(candidate) > 80:
            continue
        if candidate in q_norm:
            return True
    return False


def _compact_result_for_prompt(result_rows: list[dict], max_rows: int = 5) -> str:
    safe_rows = result_rows[: max(1, max_rows)]
    try:
        return json.dumps(safe_rows, ensure_ascii=False, default=str, indent=2)
    except Exception:
        return str(safe_rows)


def _build_question_from_cypher_and_context(
    *,
    llm,
    complexity: str,
    cypher_query: str,
    params: dict,
    result_rows: list[dict],
    ground_truth: str,
    anchor_info: dict | None,
    target_meta: dict | None,
    fallback_question: str,
) -> str:
    if not cypher_query or not result_rows:
        return str(fallback_question or "").strip()

    anchor_info = anchor_info if isinstance(anchor_info, dict) else {}
    target_meta = target_meta if isinstance(target_meta, dict) else {}
    params_text = json.dumps(params or {}, ensure_ascii=False, default=str)
    result_context = _compact_result_for_prompt(result_rows)
    system_prompt = (
        "You rewrite graph query tasks into natural business questions. "
        "Use only the provided Cypher and database result context. "
        "Do not invent facts, and do not mention graph/Cypher terminology."
    )
    user_prompt = f"""
Complexity: {complexity}
Anchor info: {json.dumps(anchor_info, ensure_ascii=False, default=str)}
Target meta: {json.dumps(target_meta, ensure_ascii=False, default=str)}
Cypher:
{cypher_query}

Cypher params:
{params_text}

Neo4j result context:
{result_context}

Ground truth value(s):
{ground_truth}

Task:
Write exactly one natural-sounding business question in English.
Rules:
1) One sentence, human-readable, no technical wording.
2) Do NOT include the exact answer value from ground truth in the question.
3) Keep the question specific enough to be answered by the provided query context.
4) No markdown, no explanation, only the question text.
"""
    response = llm.generate_response(system_prompt, user_prompt)
    question = str(response or "").strip()
    question = re.sub(r"^['\"`]+|['\"`]+$", "", question).strip()
    if "\n" in question:
        question = question.splitlines()[0].strip()
    if question and not question.endswith("?"):
        question = question.rstrip(".") + "?"
    return question or str(fallback_question or "").strip()


def validate_generated_items(
    *,
    db,
    llm,
    generated_items,
    seen_exact_questions=None,
    seen_normalized_questions=None,
    output_file=None,
    existing_benchmark=None,
    row_limit: int = GROUND_TRUTH_ROW_LIMIT,
):
    """Выполняет Cypher в базе. Если есть результат -> сохраняем в бенчмарк."""
    print("Валидация запросов в Neo4j...")
    benchmark_dataset = []
    seen_exact_questions = seen_exact_questions if seen_exact_questions is not None else set()
    seen_normalized_questions = (
        seen_normalized_questions if seen_normalized_questions is not None else []
    )
    prefix = existing_benchmark if existing_benchmark is not None else []

    for item in generated_items:
        cypher_query = item.get("cypher", "")
        question = str(item.get("question", "")).strip()
        params = item.get("params")
        if not isinstance(params, dict):
            params = {}
            item["params"] = params
        has_precomputed_context = bool(item.get("ground_truth"))
        debug_only_cypher = bool(item.get("debug_only_cypher"))
        if not isinstance(item.get("provenance"), dict):
            item["provenance"] = {"source": "legacy", "template_id": "unknown"}

        try:
            if cypher_query and is_trivial_self_return(cypher_query):
                print(f"[ПРОПУСК] Тривиальный запрос (WHERE/RETURN одного поля): {question}")
                continue
            if cypher_query and re.search(r"\belementId\s*\(", cypher_query, flags=re.IGNORECASE):
                print(f"[ПРОПУСК] Cypher использует elementId(...): {question}")
                continue

            result = []
            if cypher_query:
                if not debug_only_cypher:
                    limited_cypher = append_limit_if_missing(
                        cypher_query,
                        row_limit=row_limit,
                    )
                    if limited_cypher != cypher_query:
                        item["cypher"] = limited_cypher
                        cypher_query = limited_cypher
                # Для subgraph-deep-analytics это debug-запрос, не источник ground_truth.
                result = db.run_query(cypher_query, params)
                if not result and not (debug_only_cypher and has_precomputed_context):
                    print(f"[ПРОПУСК] Запрос вернул 0 строк: {question}")
                    continue
                if has_precomputed_context:
                    pass
                elif should_skip_low_signal_result(cypher_query, result):
                    print(f"[ПРОПУСК] Low-signal результат (None/0): {question}")
                    continue

            # Если ground_truth уже подготовлен заранее, используем его.
            if not has_precomputed_context:
                item["ground_truth"] = result_to_ground_truth(question, result)
                if not str(item.get("ground_truth", "")).strip():
                    print(f"[ПРОПУСК] Пустой ground_truth после преобразования: {question}")
                    continue

            # Генерируем человекочитаемый вопрос по Cypher + фактическому контексту Neo4j.
            if not debug_only_cypher:
                question = _build_question_from_cypher_and_context(
                    llm=llm,
                    complexity=str(item.get("complexity", "")),
                    cypher_query=cypher_query,
                    params=params,
                    result_rows=result,
                    ground_truth=str(item.get("ground_truth", "")),
                    anchor_info=item.get("anchor_info"),
                    target_meta=item.get("target_meta"),
                    fallback_question=question,
                )
                item["question"] = question

            normalized_question = normalize_question_text(question)
            if not normalized_question:
                print("[ПРОПУСК] Пустой вопрос после нормализации.")
                continue
            if normalized_question in seen_exact_questions:
                print(f"[ПРОПУСК] Дубликат вопроса (exact): {question}")
                continue
            if is_near_duplicate_question(question, seen_normalized_questions):
                print(f"[ПРОПУСК] Дубликат вопроса (near): {question}")
                continue

            complexity = str(item.get("complexity", "")).strip().lower()
            if complexity.startswith("multi-hop-"):
                deterministic_answer = _build_deterministic_answer_for_multi_hop(item)
                item["answer"] = deterministic_answer or str(item.get("ground_truth", "")).strip()
            else:
                item["answer"] = build_answer_from_context(
                    llm=llm,
                    question=question,
                    ground_truth=str(item.get("ground_truth", "")),
                    fallback=str(item.get("answer", "")),
                )
            if _question_contains_answer_leak(
                question=question,
                ground_truth=str(item.get("ground_truth", "")),
                answer=str(item.get("answer", "")),
            ):
                print(f"[ПРОПУСК] Утечка ответа в тексте вопроса: {question}")
                continue
            benchmark_dataset.append(item)
            seen_exact_questions.add(normalized_question)
            seen_normalized_questions.append(normalized_question)
            print(f"[УСПЕХ] Добавлен вопрос ({item['complexity']}): {question}")
            if output_file is not None:
                snapshot = [*prefix, *benchmark_dataset]
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(snapshot, f, ensure_ascii=False, indent=2)

        except Exception as e:
            # Если синтаксическая ошибка в Cypher - бракуем
            print(f"[ОШИБКА SYNTAX] {e} | Query: {cypher_query}")

    return benchmark_dataset
