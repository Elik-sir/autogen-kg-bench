from __future__ import annotations

import re
from typing import Any

from benchmark_generator.prompt_settings import ANCHORS_PER_LABEL_LIMIT, MAX_PATHS_PER_ANCHOR
from benchmark_generator.utils.anchor_subgraph_context import (
    build_anchor_subgraph_context,
    build_balanced_anchor_order,
    get_stratified_anchor_pool,
)
from benchmark_generator.utils.company_subgraph_context import build_company_subgraph_contexts
from benchmark_generator.utils.llm_response_parser import parse_qa_pairs_response
from benchmark_generator.utils.prompt_builder import (
    build_aggregation_prompts,
    build_multi_hop_2_prompts,
    build_multi_hop_3_prompts,
    build_same_type_common_prompts,
    build_simple_prompts,
    build_subgraph_deep_analytics_prompts,
)
from benchmark_generator.utils.same_type_common_context import find_same_type_common_contexts


class QuestionGenerationEngine:
    def __init__(self, db, llm):
        self.db = db
        self.llm = llm
        # Сквозной курсор по subgraph-контекстам между вызовами генератора.
        self._subgraph_ctx_cursor = 0
        self._anchor_order: list[dict[str, Any]] = []
        self._anchor_cursor = 0
        self._multi_hop_path_cursor = 0
        self._cross_branch_pair_cursor = 0

    def _generate_by_prompt_builder(
        self,
        prompt_builder,
        schema,
        data_samples,
        num_questions,
        existing_questions=None,
    ):
        system_prompt, user_prompt = prompt_builder(
            schema,
            data_samples,
            num_questions,
            existing_questions=existing_questions,
        )
        response = self.llm.generate_response(system_prompt, user_prompt)
        return parse_qa_pairs_response(response)

    def _build_question_for_cypher(
        self,
        *,
        cypher: str,
        complexity: str,
        existing_questions=None,
    ) -> str:
        existing_list = [str(q).strip() for q in (existing_questions or []) if str(q).strip()]
        existing_block = "\n".join(f"- {q}" for q in existing_list[-80:]) if existing_list else "- (none)"
        prompt = f"""
You are generating benchmark question text for an existing Cypher query.
Your task: write exactly one natural English question that is answered by this query result.

Complexity: {complexity}
Cypher:
{cypher}

Rules:
1) The question must match what the query returns (entity vs metric/count/value).
2) Do not mention Cypher, graph, nodes, edges, relationships, or hops.
3) One sentence only, ending with a question mark.
4) Business-analyst style wording.
5) Do not copy or closely paraphrase already used questions.
6) Output only the question text.

Already generated questions:
{existing_block}
"""
        response = self.llm.generate_response(
            "You convert Cypher queries into natural benchmark questions.",
            prompt,
        )
        question = re.sub(r"\s+", " ", str(response or "").strip())
        question = re.sub(r"^['\"`]+|['\"`]+$", "", question).strip()
        if question and not question.endswith("?"):
            question = question.rstrip(".") + "?"
        return question

    def _align_items_to_cypher(
        self,
        *,
        items,
        complexity: str,
        existing_questions=None,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            cypher = str(item.get("cypher", "")).strip()
            if not cypher:
                continue
            question = self._build_question_for_cypher(
                cypher=cypher,
                complexity=complexity,
                existing_questions=existing_questions,
            )
            if not question:
                continue
            item["complexity"] = complexity
            item["question"] = question
            out.append(item)
        return out

    def generate_simple_pairs(self, schema, data_samples, num_questions=2, existing_questions=None):
        print(f"Генерация {num_questions} simple-вопросов...")
        generated = self._generate_by_prompt_builder(
            build_simple_prompts,
            schema,
            data_samples,
            num_questions,
            existing_questions=existing_questions,
        )
        if isinstance(generated, dict):
            generated = [generated]
        return self._align_items_to_cypher(
            items=generated,
            complexity="simple",
            existing_questions=existing_questions,
        )

    def generate_multi_hop_pairs(self, schema, data_samples, num_questions=2, existing_questions=None):
        # Backward-compatible alias: old "multi-hop" maps to 2-hop variant.
        return self.generate_multi_hop_x_pairs(
            schema=schema,
            hop_count=2,
            num_questions=num_questions,
            existing_questions=existing_questions,
        )

    def _ensure_anchor_order(self) -> None:
        if self._anchor_order:
            return
        anchor_pool = get_stratified_anchor_pool(
            self.db,
            limit_per_label=ANCHORS_PER_LABEL_LIMIT,
        )
        self._anchor_order = build_balanced_anchor_order(anchor_pool)
        self._anchor_cursor = 0

    def _next_anchor(self) -> dict[str, Any] | None:
        self._ensure_anchor_order()
        if not self._anchor_order:
            return None
        anchor = self._anchor_order[self._anchor_cursor % len(self._anchor_order)]
        self._anchor_cursor = (self._anchor_cursor + 1) % len(self._anchor_order)
        return anchor

    def generate_multi_hop_x_pairs(
        self,
        schema,
        *,
        hop_count: int,
        num_questions: int = 2,
        existing_questions=None,
    ):
        if hop_count not in (2, 3):
            print(f"[ПРОПУСК] Поддерживаются только multi-hop-2/3, передано: {hop_count}")
            return []

        complexity = f"multi-hop-{hop_count}"
        print(f"Генерация {num_questions} {complexity}-вопросов...")
        prompt_builder = build_multi_hop_2_prompts if hop_count == 2 else build_multi_hop_3_prompts
        out: list[dict[str, Any]] = []
        max_attempts = max(num_questions * 6, 15)
        attempts = 0
        while len(out) < num_questions and attempts < max_attempts:
            attempts += 1
            anchor = self._next_anchor()
            if not anchor:
                break
            local_context = build_anchor_subgraph_context(
                self.db,
                anchor=anchor,
                hop_count=hop_count,
                max_paths_per_anchor=MAX_PATHS_PER_ANCHOR,
            )
            if not local_context:
                continue
            generated = self._generate_by_prompt_builder(
                prompt_builder,
                schema,
                local_context,
                min(1, num_questions - len(out)),
                existing_questions=existing_questions,
            )
            if isinstance(generated, dict):
                generated = [generated]
            if not generated:
                continue
            aligned = self._align_items_to_cypher(
                items=generated,
                complexity=complexity,
                existing_questions=existing_questions,
            )
            if not aligned:
                continue
            item = aligned[0]
            out.append(item)

        if len(out) < num_questions:
            print(f"[ПРЕДУПРЕЖДЕНИЕ] {complexity}: получено {len(out)}/{num_questions}.")
        return out

    def _try_generate_path_multi_hop_item(
        self,
        *,
        hop_count: int,
        complexity: str,
        existing_questions,
    ) -> dict[str, Any] | None:
        anchor = self._next_anchor()
        if not anchor:
            return None
        local_context = build_anchor_subgraph_context(
            self.db,
            anchor=anchor,
            hop_count=hop_count,
            max_paths_per_anchor=MAX_PATHS_PER_ANCHOR,
        )
        if not local_context:
            return None
        path = self._select_path_for_cypher(local_context)
        if not path:
            return None
        deterministic = self._build_deterministic_multi_hop_cypher(path)
        if not deterministic:
            return None
        cypher, params = deterministic
        question = self._build_question_from_path(
            path=path,
            hop_count=hop_count,
            complexity=complexity,
            existing_questions=existing_questions,
            cypher=cypher,
            params=params,
            local_ontology=str(local_context.get("local_ontology", "")),
        )
        if not question:
            return None
        item: dict[str, Any] = {
            "question": question,
            "cypher": cypher,
            "params": params,
            "complexity": complexity,
        }
        return item

    def _try_generate_simple_property_item(self, *, existing_questions) -> dict[str, Any] | None:
        anchor = self._next_anchor()
        if not anchor:
            return None
        props = dict(anchor.get("props") or {})
        skip_keys = {"embedding", "embeddings", "vector", "password", "secret"}
        preferred = (
            "industry",
            "sector",
            "country",
            "region",
            "city",
            "category",
            "year",
            "status",
            "ticker",
            "symbol",
        )
        picked_key: str | None = None
        for k in preferred:
            if k in props and props[k] not in (None, ""):
                picked_key = k
                break
        if not picked_key:
            for k, v in props.items():
                kl = str(k).lower()
                if kl in skip_keys or v in (None, ""):
                    continue
                if isinstance(v, (dict, list)):
                    continue
                picked_key = str(k)
                break
        if not picked_key:
            return None
        anchor_fragment, anchor_params = self._build_node_match_fragment(
            var_name="n",
            node={
                "labels": labels if isinstance(labels, list) else [labels],
                "props": props,
            },
            param_prefix="anchor",
        )

        cypher = """
MATCH __ANCHOR_FRAGMENT__
RETURN n[$prop_key] AS value
LIMIT 5
""".strip().replace("__ANCHOR_FRAGMENT__", anchor_fragment)
        safe_prop_key = str(picked_key).replace("`", "``")
        cypher = cypher.replace("n[$prop_key] AS value", f"n.`{safe_prop_key}` AS value")
        params = {}

        labels = anchor.get("labels") or [anchor.get("label") or "Node"]
        anchor_node = {
            "labels": labels if isinstance(labels, list) else [labels],
            "props": props,
            "element_id": anchor.get("element_id"),
        }
        entity_name = self._node_name(anchor_node, fallback="this entity")
        label_s = self._node_label(anchor_node, fallback="entity").lower()

        q = f"What is the {picked_key.replace('_', ' ')} of {entity_name}?"
        q = re.sub(r"\s+", " ", q).strip()
        if not q.endswith("?"):
            q = q.rstrip(".") + "?"
        return {
            "question": q,
            "cypher": cypher,
            "params": params,
            "complexity": "simple",
        }

    def _select_path_for_cypher(self, local_context: dict[str, Any]) -> dict[str, Any] | None:
        paths = local_context.get("paths") if isinstance(local_context, dict) else None
        if not isinstance(paths, list) or not paths:
            return None
        idx = self._multi_hop_path_cursor % len(paths)
        self._multi_hop_path_cursor += 1
        path = paths[idx]
        return path if isinstance(path, dict) else None

    def _node_name(self, node: dict[str, Any], *, fallback: str = "entity") -> str:
        props = node.get("props") if isinstance(node, dict) else None
        if isinstance(props, dict):
            for key in ("name", "title", "ticker", "id", "uuid", "symbol"):
                value = props.get(key)
                if value not in (None, ""):
                    return str(value)
        labels = node.get("labels") if isinstance(node, dict) else None
        if isinstance(labels, list) and labels:
            return str(labels[0])
        return fallback

    def _node_label(self, node: dict[str, Any], *, fallback: str = "entity") -> str:
        labels = node.get("labels") if isinstance(node, dict) else None
        if isinstance(labels, list) and labels:
            return str(labels[0])
        return fallback

    def _build_node_lookup_predicate(
        self,
        *,
        var_name: str,
        node: dict[str, Any],
        param_prefix: str,
    ) -> tuple[str, dict[str, Any]] | None:
        props = node.get("props") if isinstance(node, dict) else None
        if not isinstance(props, dict):
            return None
        preferred_keys = ("uuid", "id", "ticker", "symbol", "name", "title")
        for key in preferred_keys:
            value = props.get(key)
            if value in (None, ""):
                continue
            param_name = f"{param_prefix}_{key}"
            return f"{var_name}.{key} = ${param_name}", {param_name: value}
        return None

    def _build_node_match_fragment(
        self,
        *,
        var_name: str,
        node: dict[str, Any],
        param_prefix: str,
    ) -> tuple[str, dict[str, Any]]:
        label = self._node_label(node, fallback="")
        label_hint = f":`{str(label).replace('`', '``')}`" if label else ""
        props = node.get("props") if isinstance(node, dict) else None
        if isinstance(props, dict):
            for key in ("uuid", "id", "ticker", "symbol", "name", "title"):
                value = props.get(key)
                if value in (None, ""):
                    continue
                literal = self._to_cypher_literal(value)
                return (
                    f"({var_name}{label_hint} {{{key}: {literal}}})",
                    {},
                )
        return f"({var_name}{label_hint})", {}

    def _to_cypher_literal(self, value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        text = str(value).replace("\\", "\\\\").replace("'", "\\'")
        return f"'{text}'"

    def _select_cross_branch_case(self, local_context: dict[str, Any]) -> dict[str, Any] | None:
        paths = local_context.get("paths") if isinstance(local_context, dict) else None
        if not isinstance(paths, list) or len(paths) < 2:
            return None
        valid_paths: list[dict[str, Any]] = []
        for path in paths:
            nodes = path.get("nodes") if isinstance(path, dict) else None
            rels = path.get("relationships") if isinstance(path, dict) else None
            if not isinstance(nodes, list) or not isinstance(rels, list):
                continue
            if len(nodes) != 3 or len(rels) != 2:
                continue
            if not nodes[1].get("element_id") or not nodes[2].get("element_id"):
                continue
            valid_paths.append(path)
        if len(valid_paths) < 2:
            return None

        cases: list[dict[str, Any]] = []
        for i in range(len(valid_paths)):
            for j in range(i + 1, len(valid_paths)):
                left = valid_paths[i]
                right = valid_paths[j]
                left_nodes = left.get("nodes") or []
                right_nodes = right.get("nodes") or []
                left_mid = str((left_nodes[1] or {}).get("element_id") or "").strip()
                right_mid = str((right_nodes[1] or {}).get("element_id") or "").strip()
                left_target = str((left_nodes[2] or {}).get("element_id") or "").strip()
                right_target = str((right_nodes[2] or {}).get("element_id") or "").strip()
                if not left_mid or not right_mid or not left_target or not right_target:
                    continue
                if left_mid == right_mid:
                    continue
                if left_target == right_target:
                    continue
                cases.append({"left_path": left, "right_path": right})
        if not cases:
            return None
        idx = self._cross_branch_pair_cursor % len(cases)
        self._cross_branch_pair_cursor += 1
        return cases[idx]

    def _build_cross_branch_cypher(
        self, *, left_path: dict[str, Any], right_path: dict[str, Any]
    ) -> tuple[str, dict[str, Any]] | None:
        left_nodes = left_path.get("nodes") if isinstance(left_path, dict) else None
        right_nodes = right_path.get("nodes") if isinstance(right_path, dict) else None
        left_rels = left_path.get("relationships") if isinstance(left_path, dict) else None
        right_rels = right_path.get("relationships") if isinstance(right_path, dict) else None
        if (
            not isinstance(left_nodes, list)
            or not isinstance(right_nodes, list)
            or not isinstance(left_rels, list)
            or not isinstance(right_rels, list)
        ):
            return None
        if len(left_nodes) != 3 or len(right_nodes) != 3 or len(left_rels) != 2 or len(right_rels) != 2:
            return None

        def _safe_label(label: str) -> str:
            raw = str(label).strip()
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", raw):
                return raw
            return f"`{raw.replace('`', '``')}`"

        def _safe_rel(rel_type: str) -> str:
            raw = str(rel_type).strip()
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", raw):
                return raw
            return f"`{raw.replace('`', '``')}`"

        left_rel_1 = str((left_rels[0] or {}).get("type") or "").strip()
        left_rel_2 = str((left_rels[1] or {}).get("type") or "").strip()
        right_rel_1 = str((right_rels[0] or {}).get("type") or "").strip()
        right_rel_2 = str((right_rels[1] or {}).get("type") or "").strip()
        if not left_rel_1 or not left_rel_2 or not right_rel_1 or not right_rel_2:
            return None

        left_mid_label = self._node_label(left_nodes[1], fallback="entity")
        left_target_label = self._node_label(left_nodes[2], fallback="entity")
        right_mid_label = self._node_label(right_nodes[1], fallback="entity")
        right_target_label = self._node_label(right_nodes[2], fallback="entity")

        anchor_fragment, anchor_params = self._build_node_match_fragment(
            var_name="a", node=left_nodes[0], param_prefix="anchor"
        )
        left_target_fragment, left_target_params = self._build_node_match_fragment(
            var_name="left_target", node=left_nodes[2], param_prefix="left_target"
        )
        right_target_fragment, right_target_params = self._build_node_match_fragment(
            var_name="right_target", node=right_nodes[2], param_prefix="right_target"
        )

        cypher = f"""
MATCH {anchor_fragment}
MATCH (a)-[:{_safe_rel(left_rel_1)}]-(left_mid:{_safe_label(left_mid_label)})
      -[:{_safe_rel(left_rel_2)}]-{left_target_fragment}
MATCH (a)-[:{_safe_rel(right_rel_1)}]-(right_mid:{_safe_label(right_mid_label)})
      -[:{_safe_rel(right_rel_2)}]-{right_target_fragment}
WHERE left_mid <> right_mid
RETURN DISTINCT coalesce(a.name, a.title, a.ticker, a.symbol, a.id, a.uuid) AS anchor_value
LIMIT 5
""".strip()
        params = {}
        params.update(anchor_params)
        params.update(left_target_params)
        params.update(right_target_params)
        return cypher, params

    def _build_cross_branch_question(
        self, *, anchor: dict[str, Any], left_path: dict[str, Any], right_path: dict[str, Any]
    ) -> str:
        left_nodes = left_path.get("nodes") if isinstance(left_path, dict) else []
        right_nodes = right_path.get("nodes") if isinstance(right_path, dict) else []
        if len(left_nodes) != 3 or len(right_nodes) != 3:
            return ""
        left_name = self._node_name(left_nodes[2], fallback="left endpoint")
        right_name = self._node_name(right_nodes[2], fallback="right endpoint")
        anchor_label = self._node_label(anchor, fallback="entity").lower()
        return (
            f"Which {anchor_label} sits at the center of two separate business chains, one leading to "
            f"{left_name} and the other leading to {right_name}?"
        )

    def _build_deterministic_multi_hop_cypher(
        self, path: dict[str, Any]
    ) -> tuple[str, dict[str, Any]] | None:
        nodes = path.get("nodes") if isinstance(path, dict) else None
        rels = path.get("relationships") if isinstance(path, dict) else None
        if not isinstance(nodes, list) or not isinstance(rels, list) or not nodes or not rels:
            return None
        if len(nodes) != len(rels) + 1:
            return None

        def _safe_label(label: str) -> str:
            raw = str(label).strip()
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", raw):
                return raw
            return f"`{raw.replace('`', '``')}`"

        def _safe_rel(rel_type: str) -> str:
            raw = str(rel_type).strip()
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", raw):
                return raw
            return f"`{raw.replace('`', '``')}`"

        anchor_fragment, anchor_params = self._build_node_match_fragment(
            var_name="n0", node=nodes[0], param_prefix="anchor"
        )
        pattern_parts = [anchor_fragment]
        for idx, rel in enumerate(rels):
            rel_type = str((rel or {}).get("type") or "").strip()
            if not rel_type:
                return None
            next_node = nodes[idx + 1] if idx + 1 < len(nodes) else {}
            if idx + 1 == len(rels):
                next_fragment, target_params = self._build_node_match_fragment(
                    var_name=f"n{idx + 1}",
                    node=next_node,
                    param_prefix="target",
                )
            else:
                labels = next_node.get("labels") if isinstance(next_node, dict) else None
                label_hint = ""
                if isinstance(labels, list) and labels:
                    label_hint = ":" + _safe_label(str(labels[0]))
                next_fragment = f"(n{idx + 1}{label_hint})"
            pattern_parts.append(f"-[:{_safe_rel(rel_type)}]-")
            pattern_parts.append(next_fragment)
        hop_count = len(rels)
        path_pattern = "".join(pattern_parts)
        cypher = f"""
MATCH p={path_pattern}
RETURN
  DISTINCT coalesce(
    n{hop_count}.name,
    n{hop_count}.title,
    n{hop_count}.ticker,
    n{hop_count}.symbol,
    n{hop_count}.id,
    n{hop_count}.uuid
  ) AS target_value
LIMIT 10
""".strip()
        params = {}
        params.update(anchor_params)
        return cypher, params

    def _build_question_from_path(
        self,
        *,
        path: dict[str, Any],
        hop_count: int,
        complexity: str,
        existing_questions=None,
        cypher: str = "",
        params: dict[str, Any] | None = None,
        local_ontology: str = "",
    ) -> str:
        nodes = path.get("nodes") if isinstance(path, dict) else None
        rels = path.get("relationships") if isinstance(path, dict) else None
        if not isinstance(nodes, list) or not isinstance(rels, list):
            return ""
        if len(nodes) != len(rels) + 1 or len(nodes) < 2:
            return ""

        def _node_label(node: dict[str, Any]) -> str:
            labels = node.get("labels") if isinstance(node, dict) else None
            if isinstance(labels, list) and labels:
                return str(labels[0])
            return "entity"

        def _node_name(node: dict[str, Any]) -> str:
            props = node.get("props") if isinstance(node, dict) else None
            if not isinstance(props, dict):
                return "the anchor entity"
            for key in ("name", "title", "ticker", "id", "uuid"):
                value = props.get(key)
                if value not in (None, ""):
                    return str(value)
            return "the anchor entity"

        def _humanize_rel_type(rel_type: str) -> str:
            mapping = {
                "INVESTED_IN": "investment ties",
                "OWNS": "ownership ties",
                "CEO_OF": "executive leadership ties",
                "MENTIONED_IN": "news mention ties",
                "NEWS_ABOUT_PRODUCT": "product news ties",
                "OPERATES_IN_INDUSTRY": "industry ties",
                "PRODUCES": "product portfolio ties",
                "WORKS_AT": "employment ties",
                "SUPPLIES": "supply-chain ties",
                "PARTNERS_WITH": "partnership ties",
                "LOCATED_IN": "location ties",
                "IN_STATE": "state-level location ties",
            }
            if rel_type in mapping:
                return mapping[rel_type]
            return str(rel_type).strip().replace("_", " ").lower()

        def _normalize_question(text: str) -> str:
            q = str(text or "").strip()
            q = re.sub(r"\s+", " ", q).strip()
            q = re.sub(r"^['\"`]+|['\"`]+$", "", q).strip()
            if q and not q.endswith("?"):
                q = q.rstrip(".") + "?"
            return q

        anchor = nodes[0]
        target = nodes[-1]
        anchor_name = _node_name(anchor).strip() or "the anchor entity"
        target_label = _node_label(target).strip() or "entity"
        target_label_text = target_label.lower()

        rel_types = [str((r or {}).get("type") or "RELATED_TO").strip() for r in rels]
        rel_hints = [_humanize_rel_type(rel_type) for rel_type in rel_types if rel_type]
        rel_hints = rel_hints[: max(1, min(3, hop_count))]
        rels_text = ", then ".join(rel_hints)

        if hop_count == 1:
            fallback_question = (
                f"Which {target_label_text} is directly associated with {anchor_name} "
                f"through {rels_text}?"
            )
        else:
            # For multi-hop we include intermediate entities from the actual path to keep
            # wording grounded in the deterministic Cypher path and avoid LLM hallucinations.
            bridge_entities: list[str] = []
            for bridge in nodes[1:-1]:
                bridge_name = _node_name(bridge).strip()
                if bridge_name and bridge_name != "the anchor entity":
                    bridge_entities.append(bridge_name)
                if len(bridge_entities) >= 2:
                    break
            bridge_hint = ""
            if bridge_entities:
                bridge_hint = " via " + " and ".join(bridge_entities)

            fallback_question = (
                f"Which {target_label_text} is connected to {anchor_name}{bridge_hint} "
                f"through {rels_text}?"
            )

        fallback_question = _normalize_question(fallback_question)

        # Reliability-first: for simple/multi-hop we keep question generation deterministic,
        # because LLM paraphrases can drift to a different target node than Cypher returns.
        return fallback_question

    def generate_aggregation_pairs(self, schema, data_samples, num_questions=2, existing_questions=None):
        print(f"Генерация {num_questions} aggregation-вопросов...")
        generated = self._generate_by_prompt_builder(
            build_aggregation_prompts, schema, data_samples, num_questions, existing_questions=existing_questions
        )
        if isinstance(generated, dict):
            generated = [generated]
        return self._align_items_to_cypher(
            items=generated,
            complexity="aggregation",
            existing_questions=existing_questions,
        )

    def generate_cross_branch_pairs(self, schema, data_samples, num_questions=2, existing_questions=None):
        print(f"Генерация {num_questions} cross-branch-вопросов...")
        out: list[dict[str, Any]] = []
        max_attempts = max(num_questions * 8, 18)
        attempts = 0
        while len(out) < num_questions and attempts < max_attempts:
            attempts += 1
            anchor = self._next_anchor()
            if not anchor:
                break
            local_context = build_anchor_subgraph_context(
                self.db,
                anchor=anchor,
                hop_count=2,
                max_paths_per_anchor=MAX_PATHS_PER_ANCHOR,
            )
            if not local_context:
                continue
            case = self._select_cross_branch_case(local_context)
            if not case:
                continue
            deterministic = self._build_cross_branch_cypher(
                left_path=case["left_path"],
                right_path=case["right_path"],
            )
            if not deterministic:
                continue
            cypher, params = deterministic
            question = self._build_cross_branch_question(
                anchor=anchor,
                left_path=case["left_path"],
                right_path=case["right_path"],
            )
            if not question:
                continue
            out.append(
                {
                    "complexity": "cross-branch",
                    "question": question,
                    "cypher": cypher,
                    "params": params,
                }
            )
        if len(out) < num_questions:
            print(
                f"[ПРЕДУПРЕЖДЕНИЕ] cross-branch: получено {len(out)}/{num_questions} "
                f"после {attempts} попыток."
            )
        return out

    def generate_same_type_common_pairs(self, schema, data_samples, num_questions=2, existing_questions=None):
        print(f"Генерация {num_questions} same-type-common-вопросов...")
        contexts = find_same_type_common_contexts(
            self.db,
            max_contexts=max(num_questions * 6, 16),
        )
        if not contexts:
            print(
                "[ПРОПУСК] Нет пар узлов одной метки без прямой связи, "
                "с общей сущностью в пределах 1-3 рёбер от каждого."
            )
            return []

        out: list = []
        max_attempts = max(num_questions * 5, len(contexts) * 3, 12)
        attempts = 0
        ctx_i = 0
        while len(out) < num_questions and attempts < max_attempts:
            attempts += 1
            ctx = contexts[ctx_i % len(contexts)]
            ctx_i += 1
            system_prompt, user_prompt = build_same_type_common_prompts(
                schema, data_samples, ctx, existing_questions=existing_questions
            )
            response = self.llm.generate_response(system_prompt, user_prompt)
            parsed = parse_qa_pairs_response(response)
            if isinstance(parsed, dict):
                parsed = [parsed]
            if not parsed:
                continue
            item = parsed[0]
            if not isinstance(item, dict):
                continue
            item["complexity"] = "same-type-common"
            out.append(item)

        if len(out) < num_questions:
            print(
                f"[ПРЕДУПРЕЖДЕНИЕ] same-type-common: получено {len(out)}/{num_questions} "
                f"после {attempts} попыток."
            )
        return out

    def _normalize_subgraph_analytics_question(self, question: str, ctx: dict[str, Any]) -> str:
        raw = str(question or "").strip()
        if not raw:
            return ""

        cleaned = re.sub(r"\s+", " ", raw).strip()
        cleaned = re.sub(r"^['\"`]+|['\"`]+$", "", cleaned).strip()
        if cleaned and not cleaned.endswith("?"):
            cleaned = cleaned.rstrip(".") + "?"

        banned_patterns = (
            r"^based on (this|the) subgraph\b",
            r"^based on (this|the) graph\b",
            r"^in this graph\b",
            r"^from this graph\b",
            r"^given this graph\b",
            r"^analyze (this|the) graph\b",
            r"^using the graph\b",
            r"^according to (this|the) topology\b",
        )
        contains_graph_jargon = bool(
            re.search(r"\b(subgraph|graph|node|edge|relationship|topology|hop|path|cypher)\b", cleaned, re.I)
        )
        has_banned_start = any(re.search(pattern, cleaned, re.I) for pattern in banned_patterns)
        if not has_banned_start and not contains_graph_jargon:
            return cleaned

        anchor_props = ctx.get("anchor_props") if isinstance(ctx, dict) else {}
        anchor_name = ""
        if isinstance(anchor_props, dict):
            for key in ("name", "title", "ticker", "symbol", "id"):
                value = anchor_props.get(key)
                if value not in (None, ""):
                    anchor_name = str(value).strip()
                    break
        anchor_name = anchor_name or "the anchor company"

        rewrite_prompt = f"""
Rewrite the benchmark question into natural business English.
Keep it difficult and analytical, but remove all graph-meta phrasing.

Constraints:
1) One sentence question only.
2) Start naturally; never start with "Based on this subgraph/graph".
3) Do not use these words: graph, subgraph, node, edge, relationship, topology, hop, path, cypher.
4) Keep the original analytical intent and entity references.
5) Mention at least one concrete entity (for example: {anchor_name}).

Original question:
{cleaned}
"""
        rewritten = self.llm.generate_response(
            "You rewrite benchmark questions into natural analyst language.",
            rewrite_prompt,
        )
        rewritten_clean = re.sub(r"\s+", " ", str(rewritten or "").strip())
        rewritten_clean = re.sub(r"^['\"`]+|['\"`]+$", "", rewritten_clean).strip()
        if rewritten_clean and not rewritten_clean.endswith("?"):
            rewritten_clean = rewritten_clean.rstrip(".") + "?"
        if rewritten_clean and not re.search(
            r"\b(subgraph|graph|node|edge|relationship|topology|hop|path|cypher)\b",
            rewritten_clean,
            re.I,
        ):
            return rewritten_clean

        return (
            f"If {anchor_name} were suddenly removed from the market context, which cascading impacts would most "
            f"likely emerge across investor exposure, operational continuity, media narrative, and partner dependencies?"
        )

    def generate_subgraph_deep_analytics_pairs(
        self, schema, num_questions=3, existing_questions=None
    ) -> list[dict[str, Any]]:
        print(f"Генерация {num_questions} subgraph-deep-analytics-вопросов...")
        subgraph_contexts = build_company_subgraph_contexts(
            db_manager=self.db,
            schema=schema,
            anchors_limit=3,
        )
        if not subgraph_contexts:
            print("[ПРОПУСК] Не удалось собрать контексты подграфа компаний.")
            return []
        contexts_for_prompt = [ctx for ctx in subgraph_contexts if ctx.get("useful_context")]
        if not contexts_for_prompt:
            print("[ПРОПУСК] Нет полезного контекста для subgraph-deep-analytics.")
            return []

        # Один вызов LLM на один контекст: иначе модель смешивает якоря, и ground_truth
        # по индексу не совпадает с answer.
        out: list[dict[str, Any]] = []
        max_attempts = max(num_questions * 4, len(contexts_for_prompt) * 3, 12)
        attempts = 0
        cursor = self._subgraph_ctx_cursor % len(contexts_for_prompt)
        while len(out) < num_questions and attempts < max_attempts:
            attempts += 1
            ctx = contexts_for_prompt[cursor]
            cursor = (cursor + 1) % len(contexts_for_prompt)
            generated = self._generate_by_prompt_builder(
                build_subgraph_deep_analytics_prompts, schema, [ctx], 1, existing_questions=existing_questions
            )
            if isinstance(generated, dict):
                generated = [generated]
            if not generated:
                continue
            item = generated[0]
            if not isinstance(item, dict):
                continue
            question = str(item.get("question", "")).strip()
            target_answer = str(item.get("target_answer", "")).strip()
            question = self._normalize_subgraph_analytics_question(question, ctx)
            if not question:
                continue

            # Для этого типа `cypher` нужен только для debug-выгрузки подграфа.
            # `ground_truth` берется из target_answer, чтобы валидация и метрики
            # опирались на аналитический эталон, а не на сырой контекст подграфа.
            item["complexity"] = "subgraph-deep-analytics"
            item["question"] = question
            item["cypher"] = ctx.get("debug_cypher", "")
            item["params"] = ctx.get("debug_params", {})
            item["debug_only_cypher"] = True
            item["graph_analysis"] = str(item.get("graph_analysis", "")).strip()
            item["question_concept"] = str(item.get("question_concept", "")).strip()
            item["answer"] = target_answer or str(item.get("answer", "")).strip()
            item["ground_truth"] = item["answer"] or str(ctx.get("useful_context", "")).strip()
            item["subgraph_context"] = ctx.get("subgraph_context", "")
            item["useful_context"] = str(ctx.get("useful_context", "")).strip()
            item["topology_metrics"] = ctx.get("topology_metrics", {})
            out.append(item)

        self._subgraph_ctx_cursor = cursor

        if len(out) < num_questions:
            print(
                f"[ПРЕДУПРЕЖДЕНИЕ] subgraph-deep-analytics: получено {len(out)}/{num_questions} "
                f"после {attempts} попыток (пустые или невалидные ответы LLM)."
            )
        return out
