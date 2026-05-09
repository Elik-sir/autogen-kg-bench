"""CLI: ``uv run python -m judge`` из каталога проекта ``src/judge``."""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .config import RagasJudgeSettings
from .io import (
    QUESTION_TYPE_ORDER,
    discover_benchmark_result_files,
    load_eval_records,
    try_load_per_type_metrics_json,
)
from .ragas_eval import (
    aggregate_summaries_weighted,
    evaluate_with_ragas,
    result_summary_mean,
    save_evaluation_json,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_env() -> None:
    """Подхватить ``.env`` из корня uv-проекта (рядом с pyproject.toml)."""
    load_dotenv(_PROJECT_ROOT / ".env", override=False)


def _default_gpt_oss_reasoning_effort() -> str:
    """Значение по умолчанию для ``--reasoning-effort`` (или ``JUDGE_GPT_OSS_REASONING_EFFORT``)."""
    v = (os.environ.get("JUDGE_GPT_OSS_REASONING_EFFORT") or "low").strip().lower()
    return v if v in ("none", "low", "medium", "high") else "low"


def main() -> int:
    _load_env()

    p = argparse.ArgumentParser(description="RAGAS evaluation over JSON/JSONL (OpenAI-compatible API)")
    p.add_argument(
        "--input",
        type=str,
        default="",
        help="Один файл .json / .jsonl (взаимно исключается с --input-dir)",
    )
    p.add_argument(
        "--input-dir",
        type=str,
        default="",
        help=(
            "Каталог с результатами бенчмарка по типам (например ../benchmarks/vector-rag/results): "
            "последовательно оцениваются simple.jsonl, multi-hop-2.jsonl, … см. judge.io.QUESTION_TYPE_ORDER"
        ),
    )
    p.add_argument(
        "--output",
        type=str,
        default="",
        help="При --input: путь к итоговому .json. При --input-dir не используется (см. --output-dir).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="При --input-dir: каталог для <тип>.json и _summary.json (по умолчанию: <input-dir>/ragas).",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="При --input-dir: не пересчитывать тип, если в output-dir уже есть <тип>.json.",
    )
    p.add_argument("--api-key", default="", help="Или переменная OPENAI_API_KEY из .env / окружения")
    p.add_argument(
        "--base-url",
        default="",
        help="Необязательный base_url (или переменная OPENAI_BASE_URL / OPENROUTER_BASE_URL)",
    )
    p.add_argument("--llm-model", default="openai/gpt-oss-120b")
    p.add_argument(
        "--llm-max-tokens",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Лимит токенов ответа для RAGAS/Instructor (0 = из JUDGE_LLM_MAX_TOKENS или 16384). "
            "У gpt-oss часть бюджета уходит в reasoning — при обрезке JSON ломается (context_precision / context_recall). "
            "Для gpt-oss без своего extra_body автоматически добавляется reasoning.effort (OpenRouter; см. --reasoning-effort)."
        ),
    )
    p.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high"),
        default=_default_gpt_oss_reasoning_effort(),
        metavar="LEVEL",
        help=(
            "Только для моделей gpt-oss: кладётся в extra_body.reasoning.effort (OpenRouter), "
            "если в --chat-extra-body-json ещё нет ключа reasoning. По умолчанию low — часть маршрутов "
            "отклоняет effort=none («Reasoning is mandatory … cannot be disabled»). "
            "Переопределение: переменная JUDGE_GPT_OSS_REASONING_EFFORT."
        ),
    )
    p.add_argument("--embedding-model", default="text-embedding-3-small")
    p.add_argument(
        "--metrics",
        choices=("default", "faithfulness_only"),
        default="default",
        help=(
            "Набор метрик RAGAS: default = faithfulness + answer_relevancy + context_recall + "
            "context_precision + nv_context_relevance (Context Relevancy); "
            "faithfulness_only = faithfulness + три контекстные метрики (без answer_relevancy / эмбеддингов). "
            "Контекстные метрики нуждаются в реальных contexts в JSONL; precision/recall — ещё и в ground_truth."
        ),
    )
    p.add_argument(
        "--answer-relevancy-strictness",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Число LLM-генераций для answer_relevancy (RAGAS strictness). "
            "Многие провайдеры отдают только 1 при n>1 — тогда 1 убирает предупреждение «returned 1 generations…»."
        ),
    )
    p.add_argument(
        "--ragas-max-workers",
        type=int,
        default=4,
        metavar="N",
        help="Параллельность RAGAS evaluate (RunConfig.max_workers; по умолчанию 4).",
    )
    p.add_argument(
        "--run-max-retries",
        type=int,
        default=10,
        metavar="N",
        help="RunConfig.max_retries в RAGAS (повторы при сбоях LLM; по умолчанию 10).",
    )
    p.add_argument(
        "--header",
        action="append",
        default=[],
        help='Доп. HTTP-заголовок, формат "Name: value" (повторяемо)',
    )
    p.add_argument(
        "--openrouter-provider-only",
        default="google-vertex",
        metavar="SLUG",
        help=(
            "OpenRouter: ограничить провайдеров (кладётся в extra_body). "
            "Пример: google-vertex. Альтернатива: переменная OPENROUTER_PROVIDER_ONLY."
        ),
    )
    p.add_argument(
        "--chat-extra-body-json",
        default="",
        metavar="JSON",
        help=(
            "Сырой JSON для extra_body у chat completions (OpenRouter: provider, transforms и т.д.). "
            "Сливается с --openrouter-provider-only (ключ provider перезапишется флагом, если оба заданы)."
        ),
    )
    args = p.parse_args()

    has_input = bool((args.input or "").strip())
    has_input_dir = bool((args.input_dir or "").strip())
    if has_input == has_input_dir:
        print("Укажите ровно одно из: --input <файл> или --input-dir <каталог>", file=sys.stderr)
        return 2
    if has_input and not (args.output or "").strip():
        print("При --input нужен --output <файл.json>", file=sys.stderr)
        return 2
    if has_input and (args.output_dir or "").strip():
        print("При --input не используйте --output-dir (только --output).", file=sys.stderr)
        return 2

    api_key = (args.api_key or os.environ.get("OPENAI_API_KEY", "")).strip()
    if not api_key:
        print("Нужен --api-key или OPENAI_API_KEY в .env / окружении", file=sys.stderr)
        return 1

    base_url = (args.base_url or "").strip()
    if not base_url:
        base_url = (os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENROUTER_BASE_URL") or "").strip()

    openrouter_provider = (args.openrouter_provider_only or "").strip()
    if not openrouter_provider:
        openrouter_provider = (os.environ.get("OPENROUTER_PROVIDER_ONLY") or "").strip()

    extra_body: dict[str, Any] = {}
    raw_extra = (args.chat_extra_body_json or "").strip()
    if raw_extra:
        try:
            parsed = json.loads(raw_extra)
        except json.JSONDecodeError as e:
            print(f"Некорректный --chat-extra-body-json: {e}", file=sys.stderr)
            return 2
        if not isinstance(parsed, dict):
            print("--chat-extra-body-json должен быть JSON-объектом", file=sys.stderr)
            return 2
        extra_body.update(parsed)
    if openrouter_provider:
        extra_body["provider"] = {"only": [openrouter_provider]}

    # gpt-oss: reasoning съедает max_tokens, в content остаётся обрезанный JSON → InstructorRetryException.
    # OpenRouter: https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
    # effort=none отключает reasoning — часть endpoint'ов возвращает 400 «cannot be disabled».
    if "gpt-oss" in args.llm_model.lower() and "reasoning" not in extra_body:
        extra_body["reasoning"] = {"effort": args.reasoning_effort}

    llm_max_tokens = int(args.llm_max_tokens or 0)
    if llm_max_tokens <= 0:
        llm_max_tokens = int(os.environ.get("JUDGE_LLM_MAX_TOKENS", "16384"))

    headers: dict[str, str] | None = None
    if args.header:
        headers = {}
        for h in args.header:
            if ":" not in h:
                print(f"Некорректный --header: {h!r}", file=sys.stderr)
                return 2
            k, v = h.split(":", 1)
            headers[k.strip()] = v.strip()

    if args.metrics == "default" and "gpt-oss" in args.llm_model.lower():
        print(
            "Предупреждение: answer_relevancy ожидает строгий JSON в content; у gpt-oss при включённом "
            "reasoning ответ часто ломает парсинг. Включено: extra_body.reasoning.effort="
            f"{args.reasoning_effort} (если не задано в --chat-extra-body-json) и увеличенный лимит токенов. "
            "Варианты: --metrics faithfulness_only, другая --llm-model, или --reasoning-effort none на "
            "маршрутах, где это допустимо.",
            file=sys.stderr,
        )

    settings = RagasJudgeSettings(
        api_key=api_key,
        base_url=base_url or None,
        llm_model=args.llm_model,
        llm_max_tokens=llm_max_tokens,
        embedding_model=args.embedding_model,
        default_headers=headers,
        openai_chat_extra_body=extra_body or None,
        answer_relevancy_strictness=args.answer_relevancy_strictness,
        ragas_max_workers=args.ragas_max_workers,
        ragas_run_max_retries=args.run_max_retries,
    )

    common_meta = {
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "metrics": args.metrics,
        "answer_relevancy_strictness": settings.answer_relevancy_strictness,
        "llm_max_tokens": settings.llm_max_tokens,
        "ragas_max_workers": settings.ragas_max_workers,
        "ragas_run_max_retries": settings.ragas_run_max_retries,
        "base_url": settings.base_url,
        "openai_chat_extra_body": settings.openai_chat_extra_body,
    }
    if "gpt-oss" in args.llm_model.lower():
        common_meta["gpt_oss_reasoning_effort"] = args.reasoning_effort

    if has_input_dir:
        input_dir = Path(args.input_dir).expanduser().resolve()
        try:
            files = discover_benchmark_result_files(input_dir)
        except (OSError, NotADirectoryError) as e:
            print(str(e), file=sys.stderr)
            return 1
        if not files:
            print(
                f"В {input_dir} не найдено ни одного файла "
                f"«<тип>.jsonl» / «<тип>.json» для типов: {', '.join(QUESTION_TYPE_ORDER)}.",
                file=sys.stderr,
            )
            return 1
        out_root = (args.output_dir or "").strip()
        output_dir = (
            Path(out_root).expanduser().resolve()
            if out_root
            else (input_dir / "ragas").resolve()
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        by_type_payload: dict[str, Any] = {}
        agg_parts: list[tuple[str, dict[str, float | None], int]] = []

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            for question_type, in_path in files:
                records = load_eval_records(in_path)
                if not records:
                    print(f"[пропуск] {question_type}: пустой файл {in_path}", file=sys.stderr)
                    continue
                out_file = output_dir / f"{question_type}.json"
                if args.skip_existing:
                    cached = try_load_per_type_metrics_json(output_dir, question_type)
                    if cached is not None:
                        sm, n_rows = cached
                        agg_parts.append((question_type, sm, n_rows))
                        by_type_payload[question_type] = {
                            "n_rows": n_rows,
                            "summary": sm,
                            "input_file": str(in_path),
                            "output_file": str(out_file),
                        }
                        print(
                            f"[skip-existing] {question_type}: reused {out_file} (n={n_rows}); summary={sm}",
                            file=sys.stderr,
                        )
                        continue
                result = evaluate_with_ragas(
                    records,
                    settings,
                    metrics_preset=args.metrics,
                )
                save_evaluation_json(
                    result,
                    out_file,
                    source_records=records,
                    extra_meta={
                        **common_meta,
                        "question_type": question_type,
                        "input": os.path.abspath(str(in_path)),
                        "input_dir": str(input_dir),
                    },
                )
                sm = result_summary_mean(result)
                agg_parts.append((question_type, sm, len(records)))
                by_type_payload[question_type] = {
                    "n_rows": len(records),
                    "summary": sm,
                    "input_file": str(in_path),
                    "output_file": str(out_file),
                }
                print(f"OK: {question_type} → {out_file} ({len(records)} rows); summary={result!r}")

        if not agg_parts:
            print("Не оценён ни один тип: все найденные файлы оказались пустыми.", file=sys.stderr)
            return 1

        agg = aggregate_summaries_weighted(agg_parts)
        summary_doc = {
            "meta": {
                **common_meta,
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "types_evaluated": [t for t, _, _ in agg_parts],
                "skip_existing": bool(args.skip_existing),
            },
            "by_type": by_type_payload,
            "aggregate_summary": agg,
        }
        summary_path = output_dir / "_summary.json"
        summary_path.write_text(
            json.dumps(summary_doc, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"OK: сводка → {summary_path}")
        return 0

    records = load_eval_records(args.input)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = evaluate_with_ragas(
            records,
            settings,
            metrics_preset=args.metrics,
        )

    save_evaluation_json(
        result,
        args.output,
        source_records=records,
        extra_meta={
            "input": os.path.abspath(args.input),
            **common_meta,
        },
    )
    print(f"OK: wrote {args.output} ({len(records)} rows); summary={result!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
