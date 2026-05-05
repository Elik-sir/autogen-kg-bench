from __future__ import annotations

import concurrent.futures
import json

from benchmark_generator.dedup import is_near_duplicate_question, normalize_question_text
from benchmark_generator.utils.schema_context import get_samples, get_schema


def _run_type_generation_worker(
    *,
    type_name,
    generator_fn,
    validate_fn,
    output_file,
    target_for_type,
    batch_size,
):
    collected_for_type = 0
    attempts = 0
    max_attempts = max(target_for_type * 8, 20)
    local_benchmark = []
    local_seen_exact = set()
    local_seen_normalized = []

    print(f"\n=== Этап: {type_name} (цель {target_for_type}) ===")
    while collected_for_type < target_for_type and attempts < max_attempts:
        attempts += 1
        remaining = target_for_type - collected_for_type
        request_n = min(batch_size, remaining)
        existing_questions = [
            str(item.get("question", "")).strip()
            for item in local_benchmark
            if item.get("complexity") == type_name and str(item.get("question", "")).strip()
        ]
        generated_items = generator_fn(request_n, existing_questions=existing_questions)
        if isinstance(generated_items, dict):
            generated_items = [generated_items]

        valid_items = validate_fn(
            generated_items=generated_items or [],
            seen_exact_questions=local_seen_exact,
            seen_normalized_questions=local_seen_normalized,
            output_file=None,
            existing_benchmark=local_benchmark,
        )
        local_benchmark.extend(valid_items)
        added_for_type = sum(1 for item in valid_items if item.get("complexity") == type_name)
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

    return type_name, local_benchmark


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

    jobs = []
    generator_fn_by_type = {}
    for type_name, generator_fn, batch_size in generation_plan:
        generator_fn_by_type[type_name] = generator_fn
        target_for_type = per_type_targets[type_name]
        if target_for_type <= 0:
            continue
        jobs.append((type_name, generator_fn, batch_size, target_for_type))

    # Каждый тип вопросов генерируется и валидируется в отдельном потоке.
    # После этого результаты объединяются с глобальной дедупликацией.
    type_results: dict[str, list] = {}
    max_workers = max(1, min(len(jobs), 8))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _run_type_generation_worker,
                type_name=type_name,
                generator_fn=generator_fn,
                validate_fn=validate_fn,
                output_file=output_file,
                target_for_type=target_for_type,
                batch_size=batch_size,
            )
            for type_name, generator_fn, batch_size, target_for_type in jobs
        ]
        for future in concurrent.futures.as_completed(futures):
            type_name, collected_items = future.result()
            type_results[type_name] = collected_items

    def _count_for_type(items, type_name: str) -> int:
        return sum(1 for item in (items or []) if str(item.get("complexity", "")).strip() == type_name)

    multihop_shortfall = 0
    for short_type in ("multi-hop-3", "multi-hop-4"):
        target = int(per_type_targets.get(short_type, 0))
        collected = _count_for_type(type_results.get(short_type, []), short_type)
        if collected < target:
            multihop_shortfall += target - collected
    if multihop_shortfall > 0 and int(per_type_targets.get("multi-hop-2", 0)) > 0:
        print(
            f"[FALLBACK] Недобор multi-hop-3/4 = {multihop_shortfall}. "
            "Пытаемся компенсировать вопросами multi-hop-2."
        )
        mh2_generator = generator_fn_by_type.get("multi-hop-2")
        if mh2_generator is not None:
            mh2_items = list(type_results.get("multi-hop-2", []))
            seen_exact = {
                normalize_question_text(str(item.get("question", "")).strip())
                for item in mh2_items
                if str(item.get("question", "")).strip()
            }
            seen_normalized = [q for q in seen_exact if q]
            added = 0
            attempts = 0
            max_attempts = max(multihop_shortfall * 6, 14)
            while added < multihop_shortfall and attempts < max_attempts:
                attempts += 1
                remaining = multihop_shortfall - added
                request_n = min(3, remaining)
                existing_questions = [
                    str(item.get("question", "")).strip()
                    for item in mh2_items
                    if str(item.get("question", "")).strip()
                ]
                generated = mh2_generator(request_n, existing_questions=existing_questions)
                if isinstance(generated, dict):
                    generated = [generated]
                valid = validate_fn(
                    generated_items=generated or [],
                    seen_exact_questions=seen_exact,
                    seen_normalized_questions=seen_normalized,
                    output_file=None,
                    existing_benchmark=mh2_items,
                )
                mh2_items.extend(valid)
                added_now = _count_for_type(valid, "multi-hop-2")
                added += added_now
                print(
                    f"[FALLBACK-ПРОГРЕСС] multi-hop-2: +{added_now}, "
                    f"итого {added}/{multihop_shortfall} (попытка {attempts}/{max_attempts})"
                )
            type_results["multi-hop-2"] = mh2_items

    seen_exact_questions = set()
    seen_normalized_questions = []
    for type_name, _, _ in generation_plan:
        for item in type_results.get(type_name, []):
            question = str(item.get("question", "")).strip()
            normalized = normalize_question_text(question)
            if not normalized:
                continue
            if normalized in seen_exact_questions:
                continue
            if is_near_duplicate_question(question, seen_normalized_questions):
                continue
            seen_exact_questions.add(normalized)
            seen_normalized_questions.append(normalized)
            final_benchmark.append(item)

    # Сохраняем в файл
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_benchmark, f, ensure_ascii=False, indent=2)

    print(f"\nГотово! Бенчмарк на {len(final_benchmark)} вопросов сохранен в {output_file}")
    return final_benchmark
