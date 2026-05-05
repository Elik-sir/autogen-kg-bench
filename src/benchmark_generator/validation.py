from __future__ import annotations

import json
import os

from benchmark_generator.answer_builder import build_answer_from_context
from benchmark_generator.dedup import is_near_duplicate_question, normalize_question_text
from benchmark_generator.query_filters import append_limit_if_missing, should_skip_low_signal_result
from benchmark_generator.utils.benchmark_validation import (
    is_trivial_self_return,
    result_to_ground_truth,
)

GROUND_TRUTH_ROW_LIMIT = max(1, int(os.getenv("BENCHMARK_GROUND_TRUTH_LIMIT", "10")))


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
        question = item.get("question", "")
        params = item.get("params")
        has_precomputed_context = bool(item.get("ground_truth"))
        debug_only_cypher = bool(item.get("debug_only_cypher"))

        try:
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

            if cypher_query and is_trivial_self_return(cypher_query):
                print(f"[ПРОПУСК] Тривиальный запрос (WHERE/RETURN одного поля): {question}")
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
            item["answer"] = build_answer_from_context(
                llm=llm,
                question=question,
                ground_truth=str(item.get("ground_truth", "")),
                fallback=str(item.get("answer", "")),
            )
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
