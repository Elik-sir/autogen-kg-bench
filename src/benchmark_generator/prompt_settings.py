from __future__ import annotations

import os

ENGLISH_BENCHMARK_TEXT_RULE = (
    "LANGUAGE (mandatory): All natural-language content you produce for the benchmark must be in English: "
    "especially the `question` field. For complexity `subgraph-deep-analytics`, also write `answer` and every "
    "string in `analysis_focus` in English. Keep proper names, tickers, and literals exactly as they appear in the sample data."
)

BASE_SYSTEM_PROMPT = (
    "You are a Data Scientist. Your task is to build a benchmark for evaluating GraphRAG systems. "
    "You are given the Neo4j graph schema and SAMPLE rows from the database. "
    "You must return a strictly valid JSON array of objects. No markdown, no text before or after. "
    "Generate only questions that admit an unambiguous answer from the graph data. "
    + ENGLISH_BENCHMARK_TEXT_RULE
)

USEFUL_ENTITY_KEYS = {
    "name",
    "title",
    "description",
    "summary",
    "headline",
    "sector",
    "industry",
    "country",
    "region",
    "category",
    "status",
    "date",
    "year",
    "amount",
    "value",
    "revenue",
    "profit",
    "risk",
    "sentiment",
    "score",
    "impact",
    "ticker",
}

MAX_SCHEMA_CHARS = max(2_000, int(os.getenv("BENCHMARK_MAX_SCHEMA_CHARS", "12000")))
MAX_SAMPLES_CHARS = max(2_000, int(os.getenv("BENCHMARK_MAX_SAMPLES_CHARS", "24000")))
MAX_SCHEMA_PROPS_PER_TYPE = max(
    5, int(os.getenv("BENCHMARK_MAX_SCHEMA_PROPS_PER_TYPE", "40"))
)
MAX_SAMPLE_LABELS = max(1, int(os.getenv("BENCHMARK_MAX_SAMPLE_LABELS", "30")))
MAX_SAMPLES_PER_LABEL = max(1, int(os.getenv("BENCHMARK_MAX_SAMPLES_PER_LABEL", "3")))
MAX_PROPS_PER_SAMPLE = max(3, int(os.getenv("BENCHMARK_MAX_PROPS_PER_SAMPLE", "12")))
MAX_VALUE_CHARS = max(20, int(os.getenv("BENCHMARK_MAX_VALUE_CHARS", "180")))

ANCHORS_PER_LABEL_LIMIT = max(1, int(os.getenv("BENCHMARK_ANCHORS_PER_LABEL_LIMIT", "10")))
MAX_PATHS_PER_ANCHOR = max(1, int(os.getenv("BENCHMARK_MAX_PATHS_PER_ANCHOR", "20")))
