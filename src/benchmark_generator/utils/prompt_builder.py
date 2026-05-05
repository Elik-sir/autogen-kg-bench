from __future__ import annotations

import json
import random

from benchmark_generator.prompt_settings import (
    BASE_SYSTEM_PROMPT,
    ENGLISH_BENCHMARK_TEXT_RULE,
    MAX_PROPS_PER_SAMPLE,
    MAX_SAMPLE_LABELS,
    MAX_SAMPLES_CHARS,
    MAX_SAMPLES_PER_LABEL,
    MAX_SCHEMA_CHARS,
    MAX_SCHEMA_PROPS_PER_TYPE,
    MAX_VALUE_CHARS,
    USEFUL_ENTITY_KEYS,
)


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


def _build_multi_hop_x_prompts(
    schema,
    local_context: dict,
    *,
    count: int,
    hop_count: int,
    complexity: str,
    existing_questions=None,
):
    local_ontology = ""
    if isinstance(local_context, dict):
        local_ontology = str(local_context.get("local_ontology") or "")
    local_ontology = _truncate_text(local_ontology, MAX_SAMPLES_CHARS)
    anchor_label = (
        str(local_context.get("anchor_label", "")).strip()
        if isinstance(local_context, dict)
        else ""
    )
    anchor_id = (
        str(local_context.get("anchor_element_id", "")).strip()
        if isinstance(local_context, dict)
        else ""
    )
    paths_found = int(local_context.get("paths_found") or 0) if isinstance(local_context, dict) else 0

    user_prompt = (
        _base_user_prompt(schema, {}, existing_questions=existing_questions)
        + f"""
=== TASK TYPE: {complexity.upper()} ===
Generate {count} question(s) of type "{complexity}".

Anchor label: {anchor_label or "(unknown)"}
Anchor element_id: {anchor_id or "(unknown)"}
Extracted paths count: {paths_found}
Required reasoning hops: exactly {hop_count}

=== LOCAL ONTOLOGY (ONLY TRUSTED CONTEXT) ===
{local_ontology or "(empty)"}

=== STRICT RULES (MANDATORY) ===
1. Use ONLY the local ontology and path examples above.
2. Do NOT invent entities, labels, relationship types, or properties.
3. The generated question must require EXACTLY {hop_count} logical steps/hops.
4. Cypher must reflect exactly {hop_count} hops between the start anchor and target entity.
5. Do not collapse the task into a shorter query with fewer hops.
6. Use concrete anchor/filter values that exist in the provided local ontology.
7. Return concrete fields in RETURN (not full nodes).
8. If you cannot satisfy the rules, output an empty JSON array [].
"""
        + _output_format_prompt(complexity)
    )
    return BASE_SYSTEM_PROMPT, user_prompt


def build_multi_hop_2_prompts(schema, local_context, count, existing_questions=None):
    return _build_multi_hop_x_prompts(
        schema,
        local_context,
        count=count,
        hop_count=2,
        complexity="multi-hop-2",
        existing_questions=existing_questions,
    )


def build_multi_hop_3_prompts(schema, local_context, count, existing_questions=None):
    return _build_multi_hop_x_prompts(
        schema,
        local_context,
        count=count,
        hop_count=3,
        complexity="multi-hop-3",
        existing_questions=existing_questions,
    )


def build_multi_hop_4_prompts(schema, local_context, count, existing_questions=None):
    return _build_multi_hop_x_prompts(
        schema,
        local_context,
        count=count,
        hop_count=4,
        complexity="multi-hop-4",
        existing_questions=existing_questions,
    )


def build_multi_hop_prompts(schema, data_samples, count, existing_questions=None):
    # Backward-compatible alias; prefer build_multi_hop_2/3/4_prompts.
    return _build_multi_hop_x_prompts(
        schema,
        data_samples if isinstance(data_samples, dict) else {},
        count=count,
        hop_count=2,
        complexity="multi-hop-2",
        existing_questions=existing_questions,
    )


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

Success criterion: answering requires combining context from both chains Anchor->Entity_A and Anchor->Entity_B.
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
            f"Lengths of shortest paths A->common and B->common (in hops): {da} and {db} (each at most 3)."
        )
    if pha:
        path_lines.append(f"Example shortest path A->common: {pha}")
    if phb:
        path_lines.append(f"Example shortest path B->common: {phb}")
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
- Some entity is reachable from both A and B along chains of 1-3 hops (including only 2-3 steps with no shared direct neighbor).
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
        "You are Senior Graph Data Analyst specializing in GraphRAG global-search evaluation. "
        "Your goal is to generate a hard analytical question that cannot be solved by simple fact lookup. "
        "The question must require understanding of subgraph structure, dense connections, and multi-entity synthesis. "
        + ENGLISH_BENCHMARK_TEXT_RULE
    )

    categories = [
        "Structural Hubs & Bottlenecks",
        "Pattern Recognition & Commonalities",
        "Impact/Cascading Analysis",
        "Holistic Summarization",
    ]
    contexts_text_parts: list[str] = []
    for idx, ctx in enumerate(subgraph_contexts, 1):
        category = random.choice(categories)
        if isinstance(ctx, dict):
            anchor = ctx.get("anchor_props") or {}
            useful = str(ctx.get("useful_context") or "").strip()
            topology = str(ctx.get("subgraph_context") or "").strip()
            topology_metrics = ctx.get("topology_metrics") if isinstance(ctx.get("topology_metrics"), dict) else {}
            metrics_json = _truncate_text(json.dumps(topology_metrics, ensure_ascii=False), 2000)
            block = (
                f"CONTEXT {idx}:\n"
                f"- chosen_category: {category}\n"
                f"- anchor_props: {_truncate_text(json.dumps(anchor, ensure_ascii=False), 1200)}\n"
                f"- topology_metrics: {metrics_json}\n"
                f"- analytics_brief:\n{_truncate_text(useful, 7000)}\n"
                f"- subgraph_details:\n{_truncate_text(topology, 10000)}"
            )
            contexts_text_parts.append(block)
        else:
            contexts_text_parts.append(
                f"CONTEXT {idx}:\n- chosen_category: {category}\n- analytics_brief: {_truncate_text(str(ctx), 9000)}"
            )
    contexts_text = "\n\n".join(contexts_text_parts).strip()

    user_prompt = f"""
You must generate {count} item(s) for complexity "subgraph-deep-analytics".

GOAL:
- Create advanced benchmark questions for GraphRAG-style global search and topology-aware reasoning.
- Every question must require aggregation across at least 5-10 nodes from its subgraph context.

MANDATORY STYLE:
1) Persona: Senior Graph Data Analyst.
2) Use the assigned category per context (`chosen_category`) as the primary question style.
3) The question must be analytical and difficult.
4) Strictly avoid trivial lookup questions like "Who is connected to X?".
5) The question must be answerable only from the provided context (no outside knowledge).
6) Keep output in English.
7) Write the question as a natural analyst inquiry, not as a meta instruction.
8) Forbidden starts/patterns in question text: "Based on this subgraph", "Based on the subgraph", "In this graph",
   "From this graph", "Given this graph", "Analyze the graph", "Using the graph below", "According to the topology".
9) Do not mention graph jargon in the final question: subgraph, graph, node, edge, relationship, topology, hop, path, cypher.

ALLOWED ANALYTICAL CATEGORIES:
- Structural Hubs & Bottlenecks: identify central nodes, chokepoints, bridge-like entities.
- Pattern Recognition & Commonalities: infer shared non-obvious traits among related entities.
- Impact/Cascading Analysis: "what-if" failure/removal propagation in the dense subgraph.
- Holistic Summarization: synthesize fragmented signals into one strategic conclusion.

GOOD STYLE EXAMPLE (tone reference only):
- "If NVIDIA Corp. were to abruptly cease operations, what cascading impacts would most likely occur across investor exposure,
  technology continuity, media narrative shifts, and partner dependencies, and which entities would be most immediately affected?"

ALREADY GENERATED QUESTIONS (DO NOT REPEAT OR PARAPHRASE):
{chr(10).join(f"- {q}" for q in (existing_questions or [])[-200:]) if existing_questions else "- (none yet)"}

CONTEXTS:
{contexts_text or "(empty)"}

Return only a valid JSON array with exactly {count} object(s):
[
  {{
    "complexity": "subgraph-deep-analytics",
    "graph_analysis": "Brief structural analysis of the provided subgraph (hubs, clusters, bridge patterns).",
    "question_concept": "Rationale for the chosen analytical question design.",
    "question": "A difficult analytical question in natural language.",
    "target_answer": "Detailed ground-truth answer with explicit nodes/patterns used in reasoning."
  }}
]
"""
    return system_prompt, user_prompt
