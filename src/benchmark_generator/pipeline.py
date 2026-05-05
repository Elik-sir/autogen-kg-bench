from __future__ import annotations

import concurrent.futures
import json
import os
import tempfile
import threading
from pathlib import Path

from benchmark_generator.dedup import is_near_duplicate_question, normalize_question_text
from benchmark_generator.utils.schema_context import get_samples, get_schema


def _wip_checkpoint_path(output_file: str, type_name: str) -> str:
    """Отдельный файл на тип — безопасно при параллельной генерации."""
    p = Path(output_file)
    return str(p.parent / f"{p.stem}.wip.{type_name}{p.suffix}")


def _save_benchmark_json(path: str, data: list) -> None:
    """Атомарная запись JSON (меньше шансов получить битый файл при обрыве записи)."""
    path = os.fspath(path)
    parent = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _run_type_generation_parallel(
    *,
    type_name,
    generator_fn,
    validate_fn,
    output_file,
    target_for_type,
    batch_size,
):
    """Несколько потоков генерируют батчи параллельно; валидация и общее состояние — под одним lock."""
    collected_for_type = 0
    attempts = 0
    max_attempts = max(target_for_type * 8, 20)
    local_benchmark: list = []
    local_seen_exact: set = set()
    local_seen_normalized: list = []
    lock = threading.Lock()
    wip = _wip_checkpoint_path(output_file, type_name)

    max_workers = max(1, min(8, target_for_type))

    print(f"\n=== Этап: {type_name} (цель {target_for_type}, потоков {max_workers}) ===")

    def worker() -> None:
        nonlocal collected_for_type, attempts
        while True:
            with lock:
                if collected_for_type >= target_for_type or attempts >= max_attempts:
                    return
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

            with lock:
                if collected_for_type >= target_for_type:
                    continue
                valid_items = validate_fn(
                    generated_items=generated_items or [],
                    seen_exact_questions=local_seen_exact,
                    seen_normalized_questions=local_seen_normalized,
                    output_file=wip,
                    existing_benchmark=local_benchmark,
                )
                local_benchmark.extend(valid_items)
                added_for_type = sum(1 for item in valid_items if item.get("complexity") == type_name)
                collected_for_type += added_for_type
                print(
                    f"[ПРОГРЕСС] {type_name}: +{added_for_type}, "
                    f"итого {collected_for_type}/{target_for_type} (попытка {attempts}/{max_attempts})"
                )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker) for _ in range(max_workers)]
        for future in concurrent.futures.as_completed(futures):
            future.result()

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
    """Генерирует бенчмарк: типы строго по очереди (simple → multi-hop-… → …); внутри типа — параллельные батчи."""
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
    for type_name, generator_fn, batch_size in generation_plan:
        target_for_type = per_type_targets[type_name]
        if target_for_type <= 0:
            continue
        jobs.append((type_name, generator_fn, batch_size, target_for_type))

    # Типы идут по очереди (simple → multi-hop → …). Внутри типа — параллельные батчи.
    type_results: dict[str, list] = {}
    for type_name, generator_fn, batch_size, target_for_type in jobs:
        _, collected_items = _run_type_generation_parallel(
            type_name=type_name,
            generator_fn=generator_fn,
            validate_fn=validate_fn,
            output_file=output_file,
            target_for_type=target_for_type,
            batch_size=batch_size,
        )
        type_results[type_name] = collected_items

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
            _save_benchmark_json(output_file, final_benchmark)

    if not final_benchmark:
        _save_benchmark_json(output_file, final_benchmark)

    print(f"\nГотово! Бенчмарк на {len(final_benchmark)} вопросов сохранен в {output_file}")
    return final_benchmark
