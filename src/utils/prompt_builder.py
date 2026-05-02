import json
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


def _truncate_text(text: str, max_chars: int) -> str:
    text = str(text or "")
    if len(text) <= max_chars:
        return text
    keep_head = max_chars // 2
    keep_tail = max_chars - keep_head
    return (
        f"{text[:keep_head]}\n...[TRUNCATED {len(text) - max_chars} chars]...\n{text[-keep_tail:]}"
    )


def _trim_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return _truncate_text(value, MAX_VALUE_CHARS) if isinstance(value, str) else value
    if isinstance(value, list):
        return [_trim_value(v) for v in value[:5]]
    if isinstance(value, tuple):
        return [_trim_value(v) for v in value[:5]]
    if isinstance(value, dict):
        out = {}
        for i, (k, v) in enumerate(value.items()):
            if i >= MAX_PROPS_PER_SAMPLE:
                break
            out[str(k)] = _trim_value(v)
        return out
    # Neo4j temporal/spatial and other driver-specific values are not JSON-serializable.
    return _truncate_text(str(value), MAX_VALUE_CHARS)


def _compact_schema(schema) -> str:
    if not isinstance(schema, dict):
        return _truncate_text(str(schema), MAX_SCHEMA_CHARS)

    nodes = {}
    rels = {}
    for type_name, entry in schema.items():
        if not isinstance(entry, dict):
            continue
        type_kind = str(entry.get("type", "")).lower()
        properties = entry.get("properties") if isinstance(entry.get("properties"), dict) else {}
        prop_names = sorted(str(p) for p in properties.keys())[:MAX_SCHEMA_PROPS_PER_TYPE]
        record = {"properties": prop_names}
        if type_kind == "node":
            nodes[str(type_name)] = record
        elif type_kind == "relationship":
            rels[str(type_name)] = record

    compact = {
        "node_labels": dict(sorted(nodes.items())),
        "relationship_types": dict(sorted(rels.items())),
    }
    rendered = json.dumps(compact, ensure_ascii=False, indent=2)
    return _truncate_text(rendered, MAX_SCHEMA_CHARS)


def _is_useful_sample_key(key: str) -> bool:
    lowered = key.lower()
    if lowered in USEFUL_ENTITY_KEYS:
        return True
    return lowered.endswith("id") or lowered in {"id", "uuid", "symbol"}


def _compact_samples(data_samples) -> str:
    if not isinstance(data_samples, dict):
        return _truncate_text(str(data_samples), MAX_SAMPLES_CHARS)

    labels = sorted(data_samples.keys())[:MAX_SAMPLE_LABELS]
    compact = {}
    for label in labels:
        rows = data_samples.get(label, [])
        if not isinstance(rows, list):
            continue
        compact_rows = []
        for row in rows[:MAX_SAMPLES_PER_LABEL]:
            if not isinstance(row, dict):
                continue
            useful = {}
            for k, v in row.items():
                key_str = str(k)
                if _is_useful_sample_key(key_str):
                    useful[key_str] = _trim_value(v)
                if len(useful) >= MAX_PROPS_PER_SAMPLE:
                    break
            if not useful:
                for i, (k, v) in enumerate(row.items()):
                    if i >= MAX_PROPS_PER_SAMPLE:
                        break
                    useful[str(k)] = _trim_value(v)
            compact_rows.append(useful)
        compact[str(label)] = compact_rows

    rendered = json.dumps(compact, ensure_ascii=False, indent=2)
    return _truncate_text(rendered, MAX_SAMPLES_CHARS)


def _existing_questions_prompt(existing_questions):
    questions = [str(q).strip() for q in (existing_questions or []) if str(q).strip()]
    if not questions:
        return ""
    formatted = "\n".join(f"- {q}" for q in questions[-200:])
    return f"""

3. ALREADY GENERATED QUESTIONS (DO NOT REPEAT):
{formatted}

=== ANTI-DUPLICATION RULES (MANDATORY) ===
- Do not copy any question from the list verbatim.
- Do not produce a close paraphrase of an existing question.
- If a candidate would be too similar in meaning, pick another entity, metric, or angle.
"""


def _base_user_prompt(schema, data_samples, existing_questions=None):
    compact_schema = _compact_schema(schema)
    compact_samples = _compact_samples(data_samples)
    return f"""
You are a Senior Neo4j Architect and GraphRAG evaluation expert.
Your task is to produce a high-quality "gold standard" dataset for measuring knowledge retrieval from the graph.

=== INPUT ===
1. GRAPH SCHEMA (labels, relationships, properties):
{compact_schema}

2. SAMPLE DATA:
{compact_samples}
{_existing_questions_prompt(existing_questions)}

=== GENERAL RULES (MANDATORY) ===
1. Use only labels, relationship types, and properties that exist in the schema.
2. Use real values from the samples in WHERE clauses and patterns so queries are not empty.
3. Phrase each question naturally in English, as a business analyst would.
4. In RETURN, use concrete fields, not bare nodes.
5. Anchor to specific entities via identifying fields (name/title/ticker/id); avoid purely categorical filters.
6. {ENGLISH_BENCHMARK_TEXT_RULE}
"""


def _output_format_prompt(complexity):
    return f"""
=== OUTPUT FORMAT ===
Return only a valid JSON array:
[
  {{
    "complexity": "{complexity}",
    "question": "Clear English benchmark question text",
    "cypher": "MATCH ... RETURN ..."
  }}
]
"""


def build_simple_prompts(schema, data_samples, count, existing_questions=None):
    user_prompt = (
        _base_user_prompt(schema, data_samples, existing_questions=existing_questions)
        + f"""
=== TASK TYPE: SIMPLE ===
Generate {count} questions of type "simple":
- read attributes of one node or its direct neighbors (1 hop),
- at least one concrete entity filter.
"""
        + _output_format_prompt("simple")
    )
    return BASE_SYSTEM_PROMPT, user_prompt


def build_multi_hop_prompts(schema, data_samples, count, existing_questions=None):
    user_prompt = (
        _base_user_prompt(schema, data_samples, existing_questions=existing_questions)
        + f"""
=== TASK TYPE: MULTI-HOP ===
Generate {count} questions of type "multi-hop":
- paths of 2–4 hops across different node types,
- focus on non-obvious links or dependencies.

=== EXTRA RULES FOR NON-EMPTY RESULTS (MANDATORY) ===
1. Each question must tie to at least one anchor (name/title/ticker) that CLEARLY appears in the SAMPLE DATA.
2. Do not use rare or exotic WHERE values that do not appear in the SAMPLE DATA.
3. Avoid overly tight filter combinations (city + industry + resource + keyword) in one query.
4. Before finalizing Cypher, run an internal self-check:
   - is there at least one concrete anchor from the samples;
   - could the filter set yield an empty intersection;
   - can the query be made less brittle without losing the multi-hop intent.
5. If the query is aggregate (COUNT/SUM/AVG/MIN/MAX), phrase it so the result is informative (not null and not a trivial zero).
6. Prefer patterns where at least one hop is supported by the samples (entities and relationships appear in the provided data).
"""
        + _output_format_prompt("multi-hop")
    )
    return BASE_SYSTEM_PROMPT, user_prompt


def build_aggregation_prompts(schema, data_samples, count, existing_questions=None):
    user_prompt = (
        _base_user_prompt(schema, data_samples, existing_questions=existing_questions)
        + f"""
=== TASK TYPE: AGGREGATION ===
Generate {count} questions of type "aggregation":
- use COUNT, MAX, MIN, AVG, ORDER BY, or LIMIT,
- the wording should be analytical (rankings, comparisons, trends).
"""
        + _output_format_prompt("aggregation")
    )
    return BASE_SYSTEM_PROMPT, user_prompt


def build_cross_branch_prompts(schema, data_samples, count, existing_questions=None):
    user_prompt = (
        _base_user_prompt(schema, data_samples, existing_questions=existing_questions)
        + f"""
=== TASK TYPE: CROSS-BRANCH (SUMMARIZATION / ANALYTICS) ===
Generate {count} questions of type "cross-branch" using this recipe:
1) Pick a central anchor node.
2) Build branch A from the anchor to Entity_A.
3) Build an independent branch B from the anchor to Entity_B.
4) In the question text, avoid naming Entity_A/Entity_B directly (entity masking), but Cypher must retrieve them explicitly.

Success criterion: answering requires combining context from both chains Anchor→Entity_A and Anchor→Entity_B.
"""
        + _output_format_prompt("cross-branch")
    )
    return BASE_SYSTEM_PROMPT, user_prompt


def build_same_type_common_prompts(schema, data_samples, pair_context: dict, existing_questions=None):
    """
    One case per call: pair_context from same_type_common_context.find_same_type_common_contexts.
    """
    ctx = pair_context if isinstance(pair_context, dict) else {}
    lbl = ctx.get("node_label", "?")
    pa = ctx.get("props_a") or {}
    pb = ctx.get("props_b") or {}
    cl = ctx.get("common_labels") or []
    cc = ctx.get("common_props") or {}
    hop_a = ctx.get("hop1_text_a", "")
    hop_b = ctx.get("hop1_text_b", "")
    da = ctx.get("dist_a")
    db = ctx.get("dist_b")
    pha = ctx.get("path_hint_a")
    phb = ctx.get("path_hint_b")

    path_lines = []
    if da is not None and db is not None:
        path_lines.append(
            f"Lengths of shortest paths A→common and B→common (in hops): {da} and {db} (each at most 3)."
        )
    if pha:
        path_lines.append(f"Example shortest path A→common: {pha}")
    if phb:
        path_lines.append(f"Example shortest path B→common: {phb}")
    path_block = "\n".join(path_lines) if path_lines else ""

    case_block = f"""
CASE (node label: {lbl})
Node A (no direct edge to B): {pa}
Node B (no direct edge to A): {pb}
1-hop neighborhood of A:
{hop_a}
1-hop neighborhood of B:
{hop_b}
{path_block}

For Cypher synthesis only (do not repeat verbatim in the question): common entity labels {cl}, key fields {cc}.
Question style: what connects / what is shared / common contextual element for A and B, without naming values from {cc} explicitly.
"""

    user_prompt = (
        _base_user_prompt(schema, data_samples, existing_questions=existing_questions)
        + """
=== TASK TYPE: SAME-TYPE-COMMON ===
Generate exactly 1 question for the case below.

Logic:
- A and B share one label and have no edge between them.
- Some entity is reachable from both A and B along chains of 1–3 hops (including only 2–3 steps with no shared direct neighbor).
- Local 1-hop lists and path hints are for context; the question should target that shared entity.

Requirements:
1) In the question, refer to A and B using fields from the case (name/title/ticker, etc.).
2) Do not name the common entity in the question—the answer must come from the query.
3) Cypher must unambiguously return that common entity (RETURN clear node fields). Fixed-length paths or *1..3 are allowed if consistent with the schema. Only labels and relationship types from the schema.
"""
        + case_block
        + _output_format_prompt("same-type-common")
    )
    return BASE_SYSTEM_PROMPT, user_prompt


def build_subgraph_deep_analytics_prompts(schema, subgraph_contexts, count, existing_questions=None):
    system_prompt = (
        "You are an analyst. From the business context, formulate difficult questions "
        "that test the ability to spot hidden dependencies. "
        "Do not mention graphs, nodes, edges, relationships, hops, Cypher, or schema. "
        + ENGLISH_BENCHMARK_TEXT_RULE
    )

    def _is_useful_key(key: str) -> bool:
        lowered = key.lower()
        if "embedding" in lowered:
            return False
        if lowered in {"vector", "vectors"}:
            return False
        return lowered in USEFUL_ENTITY_KEYS

    def _pick_useful_props(props):
        if not isinstance(props, dict):
            return {}
        out = {}
        for k, v in props.items():
            if not _is_useful_key(str(k)):
                continue
            if isinstance(v, (str, int, float)) and v not in ("", None):
                out[k] = v
        return out

    contexts_text = ""
    for idx, ctx in enumerate(subgraph_contexts, 1):
        useful_lines = []
        if isinstance(ctx, dict):
            anchor = _pick_useful_props(ctx.get("anchor_props", {}))
            if anchor:
                useful_lines.append(f"anchor: {anchor}")
            if ctx.get("useful_context"):
                useful_lines.append(str(ctx.get("useful_context")))
        else:
            useful_lines.append(str(ctx))
        contexts_text += f"\nCONTEXT {idx}:\n" + "\n".join(useful_lines) + "\n"

    single_block = len(subgraph_contexts) == 1 and count == 1
    if single_block:
        intro = """Below is one business-context fragment about a company.
Generate exactly one pair: a complex analytical question and a reference answer.
The question and answer must rely only on this fragment: every fact in the answer must follow from the signals shown (no outside knowledge).
All of the question, answer, and analysis_focus strings must be in English."""
    else:
        intro = f"""Below are business-context fragments about companies.
Generate {count} pairs: each pair is a complex analytical question and a reference answer.
Critical for indexing: pair i in the JSON array must use ONLY CONTEXT i—do not mix facts across companies or move signals between blocks.
All questions, answers, and analysis_focus strings must be in English."""

    user_prompt = f"""
{intro}
The question should be useful to an analyst and may summarize facts from news or articles.
You may mentally omit some nodes or links from the provided context so the question targets finding a specific node or relationship.
Critical:
1) Questions should read like strategy / risk / market / operations inquiries.
2) Questions must require synthesizing several signals, not a single fact.
3) Do not use data-structure jargon (graph, node, edge, relationship, path, hop, Cypher).
4) A question may be open-ended but must be verifiable against this context.
5) The answer must be short, precise, and grounded only in this context (no fabrication).
6) Use only useful entity attributes (e.g. name, title, description); ignore technical fields like embeddings/vectors.
7) Do not repeat or closely paraphrase questions already generated.

ALREADY GENERATED QUESTIONS (DO NOT REPEAT):
{chr(10).join(f"- {q}" for q in (existing_questions or [])[-200:]) if existing_questions else "- (none yet)"}

CONTEXTS:
{contexts_text}

Return only a JSON array of exactly {count} objects:
[
  {{
    "complexity": "subgraph-deep-analytics",
    "question": "Complex analytical question in English",
    "answer": "Short reference answer in English",
    "analysis_focus": ["signal 1", "signal 2", "signal 3"]
  }}
]
"""
    return system_prompt, user_prompt
