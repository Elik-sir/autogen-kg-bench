from __future__ import annotations

import json

from benchmark_generator.utils.schema_context import get_samples, get_schema


def run_generation_pipeline(
    *,
    question_engine,
    validate_fn,
    db,
    target_size=5,
    output_file="graphrag_benchmark.json",
    sample_entities_per_type=10,
    per_type_targets=None,
):
    """Генерирует бенчмарк по типам по очереди: simple -> multi-hop-X -> aggregation -> cross-branch -> subgraph."""
    schema = get_schema(db)
    data_samples = get_samples(db, per_label_limit=sample_entities_per_type)
    final_benchmark = []
    seen_exact_questions = set()
    seen_normalized_questions = []

    generation_plan = [
        (
            "simple",
            lambda n, existing_questions=None: question_engine.generate_simple_pairs(
                schema, data_samples, num_questions=n, existing_questions=existing_questions
            ),
            1,
        ),
        (
            "multi-hop-2",
            lambda n, existing_questions=None: question_engine.generate_multi_hop_x_pairs(
                schema, hop_count=2, num_questions=n, existing_questions=existing_questions
            ),
            3,
        ),
        (
            "multi-hop-3",
            lambda n, existing_questions=None: question_engine.generate_multi_hop_x_pairs(
                schema, hop_count=3, num_questions=n, existing_questions=existing_questions
            ),
            3,
        ),
        (
            "multi-hop-4",
            lambda n, existing_questions=None: question_engine.generate_multi_hop_x_pairs(
                schema, hop_count=4, num_questions=n, existing_questions=existing_questions
            ),
            3,
        ),
        (
            "aggregation",
            lambda n, existing_questions=None: question_engine.generate_aggregation_pairs(
                schema, data_samples, num_questions=n, existing_questions=existing_questions
            ),
            7,
        ),
        (
            "cross-branch",
            lambda n, existing_questions=None: question_engine.generate_cross_branch_pairs(
                schema, data_samples, num_questions=n, existing_questions=existing_questions
            ),
            1,
        ),
        (
            "subgraph-deep-analytics",
            lambda n, existing_questions=None: question_engine.generate_subgraph_deep_analytics_pairs(
                schema, num_questions=n, existing_questions=existing_questions
            ),
            5,
        ),
        # (
        #   "same-type-common",
        #   lambda n, existing_questions=None: question_engine.generate_same_type_common_pairs(
        #       schema, data_samples, num_questions=n, existing_questions=existing_questions
        #   ),
        #   2
        # ),
    ]

    if per_type_targets is None:
        base = target_size // len(generation_plan)
        remainder = target_size % len(generation_plan)
        per_type_targets = {
            type_name: base + (1 if i < remainder else 0)
            for i, (type_name, _, _) in enumerate(generation_plan)
        }

    # Поддерживаем только известные типы; если цель не задана - 0.
    per_type_targets = {
        type_name: int(max(0, per_type_targets.get(type_name, 0)))
        for type_name, _, _ in generation_plan
    }

    print("\nПлан генерации по типам:")
    for type_name, _, _ in generation_plan:
        print(f"- {type_name}: {per_type_targets[type_name]}")

    for type_name, generator_fn, batch_size in generation_plan:
        target_for_type = per_type_targets[type_name]
        if target_for_type <= 0:
            continue

        print(f"\n=== Этап: {type_name} (цель {target_for_type}) ===")
        collected_for_type = 0
        attempts = 0
        max_attempts = max(target_for_type * 8, 20)

        while collected_for_type < target_for_type and attempts < max_attempts:
            attempts += 1
            remaining = target_for_type - collected_for_type
            request_n = min(batch_size, remaining)
            existing_questions = [
                str(item.get("question", "")).strip()
                for item in final_benchmark
                if item.get("complexity") == type_name and str(item.get("question", "")).strip()
            ]
            generated_items = generator_fn(request_n, existing_questions=existing_questions)
            if isinstance(generated_items, dict):
                generated_items = [generated_items]

            valid_items = validate_fn(
                generated_items=generated_items or [],
                seen_exact_questions=seen_exact_questions,
                seen_normalized_questions=seen_normalized_questions,
                output_file=output_file,
                existing_benchmark=final_benchmark,
            )
            final_benchmark.extend(valid_items)
            added_for_type = sum(
                1 for item in valid_items if item.get("complexity") == type_name
            )
            collected_for_type += added_for_type
            print(
                f"[ПРОГРЕСС] {type_name}: +{added_for_type}, "
                f"итого {collected_for_type}/{target_for_type} (попытка {attempts}/{max_attempts})"
            )

        if collected_for_type < target_for_type:
            print(
                f"[ПРЕДУПРЕЖДЕНИЕ] Тип {type_name}: собрано {collected_for_type}/{target_for_type}. "
                "Лимит попыток исчерпан."
            )

    # Сохраняем в файл
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_benchmark, f, ensure_ascii=False, indent=2)

    print(f"\nГотово! Бенчмарк на {len(final_benchmark)} вопросов сохранен в {output_file}")
    return final_benchmark
