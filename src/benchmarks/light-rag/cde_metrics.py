"""
Метрики Comprehensiveness, Diversity, Empowerment (CDE) по файлу `benchmark_data.jsonl`.

Протокол: LLM-судья оценивает ответ (answer) в контексте вопроса (question) и эталона
(ground_truth), в духе оценок качества GraphRAG / query-focused summarization.

Запуск (из папки light-rag, с PYTHONPATH=../..):
  python cde_metrics.py
  python cde_metrics.py --input benchmark_data.jsonl --output benchmark_cde.json

Модель судьи: settings.METRICS_JUDGE_MODEL или settings.LLM_MODEL, ключ — как в main.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

_LIGHT_RAG = Path(__file__).resolve().parent

import settings  # noqa: E402


@dataclass
class CdeScores:
    comprehensiveness: float
    diversity: float
    empowerment: float

    def as_dict(self) -> dict[str, float]:
        return {
            "comprehensiveness": self.comprehensiveness,
            "diversity": self.diversity,
            "empowerment": self.empowerment,
        }


JUDGE_SYSTEM = """Ты оценщик ответов RAG в бенчмарке. Твоя задача — судить по соответствию **эталону** и вопросу, а не по «вежливости» или гладкости формулировки.

КРИТИЧЕСКИЕ ПРАВИЛА (имеют приоритет над общими формулировками):
1) **Эталон (ground truth)** — ожидаемый факт для этой строки бенчмарка. Если эталон НЕ пустой и содержит конкретику (число, имя, перечень, факт «да/нет» в виде числа и т.п.), а ответ системы **не передаёт этот факт** (в т.ч. говорит «нет данных», «невозможно ответить», «в контексте не упоминается», «недостаточно информации») — это **провал извлечения/ответа**, а не успех. В таких случаях:
   - comprehensiveness: 1 (иногда 2, если передан хотя бы частично релевантный факт, но не эталон).
   - empowerment: 1 (иногда 2 по той же логике).
   - diversity: 1–2, если ответ — по сути один тезис об отсутствии информации; не завышай за объём текста.
2) Эталон вроде **«0»**, **«1»**, **«7.7»** — это полноценные ответы. Если система **не** даёт то же значение/смысл (а отказывается) — п.1, оценки низкие. Не путай «честный отказ» с хорошим ответом: для бенчмарка отказ при заданном эталоне = ошибка.
3) **Не** завышай баллы за структуру, список ссылок, вежливость, развёрнутость, если суть из эталона не донесена.
4) Высокие 4–5 по comprehensiveness/empowerment только если ожидаемое содержание эталона **видно** в ответе (с учётом перефраза и чисел/имён).
5) diversity: 4–5 только если в ответе **реально** несколько различимых, уместных аспектов; короткий или длинный повтор «мы не можем» — низкая diversity.

Шкалы 1–5 — целые. Верни только JSON."""

JUDGE_USER_TEMPLATE = """{recall_block}Вопрос:
{question}

Эталон (ожидаемый ответ в бенчмарке; может быть одним числом или краткой строкой):
{ideal}

Ответ системы:
{answer}

Оцени (целые 1–5):
- **comprehensiveness**: насколько ответ **по содержанию** покрывает ожидаемое эталоном и вопросом (сопоставь с эталоном явно). Отказ/«нет в контексте» при непустом эталоне — низкий балл (см. правила system).
- **diversity**: разнообразие **уместных** независимых смыслов; не путай с длиной отказа.
- **empowerment**: насколько пользователь получил **ожидаемый факт** (или прямой эквивалент), а не уход от ответа.

JSON одной строкой, без пояснений:
{{"comprehensiveness": <int 1-5>, "diversity": <int 1-5>, "empowerment": <int 1-5>}}"""


def _recall_block(row: dict[str, Any]) -> str:
    v = row.get("recall_on_ground_truth_tokens")
    if v is None:
        return ""
    try:
        r = float(v)
    except (TypeError, ValueError):
        return ""
    return (
        f"Служебно: recall относительно эталона (токенное пересечение, 0..1) = {r:.4f}. "
        f"Если 0.0 и эталон непустой, ответ с высокой вероятностью **не** воспроизводит ожидаемый факт; "
        f"оцени строго, особенно если ответ = отказ/«нет данных».\n\n"
    )


def _judge_model() -> str:
    m = (getattr(settings, "METRICS_JUDGE_MODEL", None) or "").strip()
    return m or settings.LLM_MODEL


def _openai_client() -> OpenAI:
    key = settings.OPENROUTER_API_KEY or ""
    if not key.strip():
        raise RuntimeError("Нужен OPENROUTER_API_KEY в settings.py или .env")
    kwargs: dict[str, Any] = {
        "base_url": settings.OPENAI_API_BASE,
        "api_key": key,
    }
    if settings.OPENROUTER_HTTP_REFERER:
        kwargs["default_headers"] = {
            "HTTP-Referer": settings.OPENROUTER_HTTP_REFERER,
            "X-Title": settings.OPENROUTER_APP_TITLE,
        }
    return OpenAI(**kwargs)


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if m:
        text = m.group(1)
    m2 = re.search(r"\{[\s\S]*\}", text)
    if m2 and not text.startswith("{"):
        text = m2.group(0)
    return json.loads(text)


def score_answer(
    client: OpenAI,
    model: str,
    question: str,
    ideal: str,
    answer: str,
    row: dict[str, Any] | None = None,
) -> CdeScores:
    rb = _recall_block(row) if row else ""
    user = JUDGE_USER_TEMPLATE.format(
        recall_block=rb,
        question=question or "",
        ideal=ideal or "",
        answer=answer or "",
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
    )
    content = (resp.choices[0].message.content or "").strip()
    data = _extract_json_object(content)
    c = int(data["comprehensiveness"])
    d = int(data["diversity"])
    e = int(data["empowerment"])
    for name, v in (("comprehensiveness", c), ("diversity", d), ("empowerment", e)):
        if not 1 <= v <= 5:
            raise ValueError(f"{name} out of range: {v}")
    return CdeScores(
        comprehensiveness=float(c),
        diversity=float(d),
        empowerment=float(e),
    )


def load_items_from_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("kind") == "summary":
            continue
        if row.get("kind") == "item" or (
            "question" in row and "answer" in row and "index" in row
        ):
            rows.append(row)
    return rows


def run_cde(
    input_path: Path,
    output_path: Path,
    delay_sec: float | None = None,
) -> dict[str, Any]:
    if delay_sec is None:
        delay_sec = float(getattr(settings, "METRICS_API_DELAY_SEC", 0.0))
    model = _judge_model()
    client = _openai_client()
    items = load_items_from_jsonl(input_path)
    if not items:
        raise ValueError(f"Нет записей kind=item в {input_path}")

    per_index: list[dict[str, Any]] = []
    for row in items:
        idx = row.get("index", len(per_index) + 1)
        q = str(row.get("question", ""))
        ideal = str(row.get("ground_truth"))
        ans = str(row.get("answer", ""))
        try:
            s = score_answer(client, model, q, ideal, ans, row)
            entry = {
                "index": idx,
                "comprehensiveness": s.comprehensiveness,
                "diversity": s.diversity,
                "empowerment": s.empowerment,
                "error": None,
            }
        except Exception as ex:  # noqa: BLE001
            entry = {
                "index": idx,
                "comprehensiveness": None,
                "diversity": None,
                "empowerment": None,
                "error": str(ex),
            }
        per_index.append(entry)
        print(f"  [CDE] index {idx} → {entry.get('comprehensiveness', '?')} / "
              f"{entry.get('diversity', '?')} / {entry.get('empowerment', '?')}", flush=True)
        if delay_sec > 0:
            time.sleep(delay_sec)

    valid = [p for p in per_index if p.get("error") is None and p.get("comprehensiveness") is not None]
    n = len(valid)
    if n == 0:
        means = None
    else:
        means = {
            "comprehensiveness": sum(p["comprehensiveness"] for p in valid) / n,
            "diversity": sum(p["diversity"] for p in valid) / n,
            "empowerment": sum(p["empowerment"] for p in valid) / n,
        }

    report = {
        "source_file": str(input_path.resolve()),
        "judge_model": model,
        "n_scored": n,
        "n_errors": len(per_index) - n,
        "mean": means,
        "items": per_index,
    }
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CDE-метрики по benchmark_data.jsonl")
    p.add_argument(
        "--input",
        type=Path,
        default=_LIGHT_RAG / "benchmark_data.jsonl",
        help="JSONL с kind=item (question, ground_truth, answer)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=_LIGHT_RAG / "benchmark_cde_metrics.json",
        help="Куда записать агрегат и оценки по индексам",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=None,
        help="Пауза между запросами (сек); по умолчанию METRICS_API_DELAY_SEC в settings",
    )
    return p.parse_args()


def main() -> int:
    args = _parse()
    if not args.input.is_file():
        print(f"Файл не найден: {args.input}", file=sys.stderr)
        return 1
    try:
        report = run_cde(args.input, args.output, delay_sec=args.delay)
    except Exception as e:  # noqa: BLE001
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1
    print(json.dumps({"mean": report["mean"], "n_scored": report["n_scored"]}, ensure_ascii=False))
    print(f"Сохранено: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
