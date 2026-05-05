"""
Ручная проверка LLM-as-judge (accuracy) для HippoRAG — без индексации и RAG.

По умолчанию читает **benchmark_data.jsonl** из этого каталога (или путь из
settings.OUTPUT_FILE): строки `kind: item`, эталон как в бенчмарке
(`ideal_for_scoring` при scoring_reference=answer, иначе ground_truth).
Поле `llm_accuracy_correct` из файла используется как expect — сравнение
с повторным вызовом судьи (удобно ловить нестабильность).

Запуск из каталога hippo-rag (как main.py):
  (PowerShell)
  $env:PYTHONPATH = (Resolve-Path ..\\..).Path
  uv run python test_llm_judge.py

  только расхождения с сохранённым llm_accuracy_correct:
  uv run python test_llm_judge.py --only-mismatches

  первые 3 вопроса (экономия API):
  uv run python test_llm_judge.py --limit 3

  синтетические мини-кейсы вместо jsonl:
  uv run python test_llm_judge.py --demo

  свой JSON с кейсами (массив объектов или {\"cases\": [...]}):
  uv run python test_llm_judge.py --cases path/to/cases.json

  явный jsonl:
  uv run python test_llm_judge.py --data other.jsonl

  только CDE (без бинарного accuracy — один вызов LLM на строку):
  uv run python test_llm_judge.py --cde-only

  только accuracy без CDE:
  uv run python test_llm_judge.py --no-cde
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parent.parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import settings  # noqa: E402
from bench_utils import _ideal_for_llm_judge  # noqa: E402
from openai import OpenAI  # noqa: E402
from utils.eval import (  # noqa: E402
    ACCURACY_SYSTEM,
    ACCURACY_USER_TEMPLATE,
    CDE_JUDGE_SYSTEM,
    CDE_JUDGE_USER_TEMPLATE,
    cde_recall_block,
    extract_json_object,
    openai_client_for_judge,
    resolve_judge_model,
)


def _call_judge_raw(
    client: OpenAI,
    model: str,
    question: str,
    ideal: str,
    answer: str,
) -> tuple[bool, str, dict[str, Any]]:
    user = ACCURACY_USER_TEMPLATE.format(
        question=question or "",
        ideal=ideal or "",
        answer=answer or "",
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": ACCURACY_SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
    )
    content = (resp.choices[0].message.content or "").strip()
    data = extract_json_object(content)
    if "correct" not in data:
        raise ValueError(f"judge JSON missing 'correct': {content!r}")
    v = data["correct"]
    if not isinstance(v, bool):
        raise ValueError(f"'correct' must be bool, got {type(v).__name__}: {content!r}")
    return v, content, data


def _call_cde_raw(
    client: OpenAI,
    model: str,
    question: str,
    ideal: str,
    answer: str,
    cde_row: dict[str, Any] | None,
) -> tuple[tuple[float, float, float], str, dict[str, Any]]:
    """Возвращает ((c, d, e), raw_content, parsed_json)."""
    rb = cde_recall_block(cde_row)
    user = CDE_JUDGE_USER_TEMPLATE.format(
        recall_block=rb,
        question=question or "",
        ideal=ideal or "",
        answer=answer or "",
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": CDE_JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
    )
    content = (resp.choices[0].message.content or "").strip()
    data = extract_json_object(content)
    c = int(data["comprehensiveness"])
    d = int(data["diversity"])
    e = int(data["empowerment"])
    for name, val in (("comprehensiveness", c), ("diversity", d), ("empowerment", e)):
        if not 1 <= val <= 5:
            raise ValueError(f"{name} out of range: {val}: {content!r}")
    return (float(c), float(d), float(e)), content, data


def default_cases() -> list[dict[str, Any]]:
    """Кейсы: expect_correct — «как должно быть» для регрессии; None = только вывод."""
    return [
        {
            "name": "точное совпадение",
            "question": "Столица Франции?",
            "ideal": "Париж",
            "answer": "Париж",
            "expect_correct": True,
        },
        {
            "name": "перефраз, суть та же",
            "question": "Столица Франции?",
            "ideal": "Париж",
            "answer": "Столицей является город Париж.",
            "expect_correct": True,
        },
        {
            "name": "число как в эталоне",
            "question": "Сколько?",
            "ideal": "7.7",
            "answer": "7.7",
            "expect_correct": True,
        },
        {
            "name": "отказ при непустом эталоне",
            "question": "Столица Франции?",
            "ideal": "Париж",
            "answer": "В предоставленном контексте нет информации для ответа.",
            "expect_correct": False,
        },
        {
            "name": "ошибочный факт",
            "question": "Столица Франции?",
            "ideal": "Париж",
            "answer": "Лондон",
            "expect_correct": False,
        },
        {
            "name": "да/нет как число в эталоне",
            "question": "Есть ли связь?",
            "ideal": "1",
            "answer": "Да, связь есть.",
            "expect_correct": True,
        },
        {
            "name": "частичный список vs полный эталон (строго — на усмотрение судьи)",
            "question": "Перечисли элементы.",
            "ideal": "A; B; C",
            "answer": "A и B.",
            "expect_correct": None,
        },
        {
            "name": "пустой ответ",
            "question": "Что?",
            "ideal": "X",
            "answer": "",
            "expect_correct": False,
        },
    ]


def load_cases(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and "cases" in raw:
        c = raw["cases"]
        if isinstance(c, list):
            return c
    raise ValueError("Ожидается JSON-массив кейсов или объект {\"cases\": [...]}")


def _default_benchmark_jsonl_path() -> Path:
    """Как вывод main.py: OUTPUT_FILE относительно каталога hippo-rag."""
    hip = Path(__file__).resolve().parent
    out = (getattr(settings, "OUTPUT_FILE", None) or "").strip() or "benchmark_data.jsonl"
    p = Path(out).expanduser()
    return p if p.is_absolute() else (hip / p).resolve()


def load_benchmark_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    """
    Строки JSONL с kind=item; эталон и expect как в прогоне бенчмарка.
    Возвращает (кейсы для судьи, число пропущенных строк без эталона).
    """
    if not path.is_file():
        raise FileNotFoundError(f"Нет файла: {path}")
    cases: list[dict[str, Any]] = []
    skipped_no_ideal = 0
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSONL строка {line_no}: {e}") from e
        if row.get("kind") != "item":
            continue
        ideal = _ideal_for_llm_judge(row)
        if not ideal:
            skipped_no_ideal += 1
            continue
        idx = row.get("index", len(cases) + 1)
        exp = row.get("llm_accuracy_correct")
        if exp is not None and not isinstance(exp, bool):
            exp = None
        cases.append(
            {
                "name": f"#{idx}",
                "question": str(row.get("question", "")),
                "ideal": ideal,
                "answer": str(row.get("answer", "")),
                "expect_correct": exp,
                "_scoring_reference": row.get("scoring_reference"),
                "_saved_recall": row.get("recall_on_ground_truth_tokens"),
                "_cde_row": {
                    "recall_on_ground_truth_tokens": row.get("recall_on_ground_truth_tokens"),
                },
            }
        )
    return cases, skipped_no_ideal


def main() -> int:
    p = argparse.ArgumentParser(description="Проверка LLM judge (accuracy) для HippoRAG")
    p.add_argument(
        "--data",
        type=Path,
        default=None,
        help="JSONL результатов (по умолчанию benchmark_data.jsonl из settings.OUTPUT_FILE)",
    )
    p.add_argument(
        "--cases",
        type=Path,
        default=None,
        help="JSON с ручными кейсами (массив или {\"cases\": [...]}), вместо --data",
    )
    p.add_argument(
        "--demo",
        action="store_true",
        help="Синтетические мини-кейсы вместо benchmark_data.jsonl",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Максимум N строк (0 = без лимита)",
    )
    p.add_argument(
        "--only-mismatches",
        action="store_true",
        help="Печатать только кейсы, где expect задан и не совпал с вердиктом accuracy",
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--no-cde",
        action="store_true",
        help="Не вызывать CDE-судью (только accuracy)",
    )
    g.add_argument(
        "--cde-only",
        action="store_true",
        help="Только CDE-судья (как light-rag/cde_metrics.py), без accuracy",
    )
    args = p.parse_args()

    if args.demo and (args.cases or args.data):
        print("Нельзя совмещать --demo с --cases/--data", file=sys.stderr)
        return 2
    if args.cases and args.data:
        print("Укажите либо --cases, либо --data, не оба", file=sys.stderr)
        return 2
    if args.only_mismatches and args.cde_only:
        print(
            "Предупреждение: при --cde-only флаг --only-mismatches не используется (нет accuracy).",
            file=sys.stderr,
        )

    if args.demo:
        cases = default_cases()
        skipped_no_ideal = 0
        data_path: Path | None = None
    elif args.cases:
        cases = load_cases(args.cases)
        skipped_no_ideal = 0
        data_path = None
    else:
        data_path = args.data if args.data else _default_benchmark_jsonl_path()
        try:
            cases, skipped_no_ideal = load_benchmark_jsonl(data_path)
        except FileNotFoundError as e:
            print(f"{e}\nПодсказка: прогоните main.py или укажите --data / --demo", file=sys.stderr)
            return 1

    if args.limit and args.limit > 0:
        cases = cases[: args.limit]

    run_accuracy = not args.cde_only
    run_cde = not args.no_cde

    model = resolve_judge_model(settings)
    print(f"OPENAI_API_BASE: {getattr(settings, 'OPENAI_API_BASE', '')}")
    print(f"judge model: {model}")
    print(f"METRICS_JUDGE_MODEL (settings): {getattr(settings, 'METRICS_JUDGE_MODEL', '')!r}")
    if data_path is not None:
        print(f"data: {data_path}")
    if args.demo:
        print("режим: --demo (синтетика)")
    elif args.cases:
        print(f"режим: --cases {args.cases}")
    print(f"кейсов к судье: {len(cases)}", end="")
    if skipped_no_ideal:
        print(f" (пропущено без эталона в jsonl: {skipped_no_ideal})")
    else:
        print()
    modes = []
    if run_accuracy:
        modes.append("accuracy")
    if run_cde:
        modes.append("CDE")
    print(f"режимы LLM: {', '.join(modes) or '(ничего)'}")
    print()

    try:
        client = openai_client_for_judge(settings)
    except Exception as e:  # noqa: BLE001
        print(f"Клиент не создан: {e}", file=sys.stderr)
        return 1

    mismatches = 0
    errors = 0
    delay = float(getattr(settings, "METRICS_API_DELAY_SEC", 0.0) or 0.0)

    sum_c = sum_d = sum_e = 0.0
    n_cde = 0

    for i, row in enumerate(cases, 1):
        name = str(row.get("name", f"case_{i}"))
        q = str(row.get("question", ""))
        ideal = str(row.get("ideal", row.get("ground_truth", "")))
        answer = str(row.get("answer", ""))
        expect = row.get("expect_correct")
        if expect is not None and not isinstance(expect, bool):
            print(f"[{i}] {name}: expect_correct должен быть bool или отсутствовать", file=sys.stderr)
            errors += 1
            continue

        ok: bool | None = None
        raw = ""
        data: dict[str, Any] = {}
        acc_err: str | None = None
        cde_scores: tuple[float, float, float] | None = None
        cde_raw = ""
        cde_data: dict[str, Any] = {}
        cde_err: str | None = None

        if run_accuracy:
            try:
                ok, raw, data = _call_judge_raw(client, model, q, ideal, answer)
            except Exception as ex:  # noqa: BLE001
                acc_err = str(ex)
                errors += 1
            if delay > 0 and run_accuracy:
                time.sleep(delay)

        if run_cde:
            cde_row = row.get("_cde_row")
            if cde_row is None and row.get("recall_on_ground_truth_tokens") is not None:
                cde_row = {"recall_on_ground_truth_tokens": row.get("recall_on_ground_truth_tokens")}
            try:
                cde_scores, cde_raw, cde_data = _call_cde_raw(client, model, q, ideal, answer, cde_row)
                sum_c += cde_scores[0]
                sum_d += cde_scores[1]
                sum_e += cde_scores[2]
                n_cde += 1
            except Exception as ex:  # noqa: BLE001
                cde_err = str(ex)
                errors += 1
            if delay > 0 and run_cde:
                time.sleep(delay)

        mismatch = bool(run_accuracy and expect is not None and ok is not None and ok != expect)
        if mismatch:
            mismatches += 1

        if args.only_mismatches and args.cde_only:
            pass
        elif args.only_mismatches and not mismatch and not acc_err and not cde_err:
            continue

        print(f"--- [{i}] {name} ---")
        sr = row.get("_scoring_reference")
        if sr is not None:
            print(f"scoring_reference: {sr}")
        rcl = row.get("_saved_recall")
        if rcl is not None:
            print(f"saved recall_on_ground_truth_tokens: {rcl}")
        if run_accuracy:
            if acc_err:
                print(f"accuracy ERROR: {acc_err}")
            elif ok is not None:
                print(f"judge_correct: {ok}")
                if expect is not None:
                    mark = "OK" if ok == expect else "MISMATCH"
                    print(f"expect_correct: {expect}  [{mark}]")
                print(f"accuracy raw: {raw!r}")
                print(f"accuracy parsed: {data}")
        if run_cde:
            if cde_err:
                print(f"CDE ERROR: {cde_err}")
            elif cde_scores is not None:
                print(
                    f"CDE: comprehensiveness={cde_scores[0]:.0f}, diversity={cde_scores[1]:.0f}, "
                    f"empowerment={cde_scores[2]:.0f}"
                )
                print(f"CDE raw: {cde_raw!r}")
                print(f"CDE parsed: {cde_data}")
        print(f"Q: {q[:200]}{'…' if len(q) > 200 else ''}")
        print(f"ideal: {ideal[:300]}{'…' if len(ideal) > 300 else ''}")
        print(f"answer: {answer[:500]}{'…' if len(answer) > 500 else ''}\n")

    print("--- итог ---")
    print(f"errors: {errors}, mismatches accuracy (vs expect): {mismatches}")
    if run_cde and n_cde:
        print(
            "mean CDE: "
            f"c={sum_c / n_cde:.3f}, d={sum_d / n_cde:.3f}, e={sum_e / n_cde:.3f} (n={n_cde})"
        )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
