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
    build_multi_hop_4_prompts,
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

    def generate_simple_pairs(self, schema, data_samples, num_questions=2, existing_questions=None):
        print(f"Генерация {num_questions} simple-вопросов...")
        return self._generate_by_prompt_builder(
            build_simple_prompts, schema, data_samples, num_questions, existing_questions=existing_questions
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
        complexity = f"multi-hop-{hop_count}"
        print(f"Генерация {num_questions} {complexity}-вопросов...")
        prompt_builders = {
            2: build_multi_hop_2_prompts,
            3: build_multi_hop_3_prompts,
            4: build_multi_hop_4_prompts,
        }
        if prompt_builders.get(hop_count) is None:
            print(f"[ПРОПУСК] Неизвестный hop_count={hop_count}")
            return []

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
                hop_count=hop_count,
                max_paths_per_anchor=MAX_PATHS_PER_ANCHOR,
            )
            if not local_context:
                continue
            path = self._select_path_for_cypher(local_context)
            if not path:
                continue
            deterministic = self._build_deterministic_multi_hop_cypher(path)
            if not deterministic:
                continue
            cypher, params = deterministic
            question = self._build_question_from_path(
                path=path,
                hop_count=hop_count,
                complexity=complexity,
                existing_questions=existing_questions,
            )
            if not question:
                continue
            item: dict[str, Any] = {
                "question": question,
                "cypher": cypher,
                "params": params,
            }
            item["complexity"] = complexity
            out.append(item)

        if len(out) < num_questions:
            print(
                f"[ПРЕДУПРЕЖДЕНИЕ] {complexity}: получено {len(out)}/{num_questions} "
                f"после {attempts} попыток."
            )
        return out

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
        self, *, anchor_element_id: str, left_path: dict[str, Any], right_path: dict[str, Any]
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
            return str(label).replace("`", "``")

        def _safe_rel(rel_type: str) -> str:
            return str(rel_type).replace("`", "``")

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

        left_target_id = str((left_nodes[2] or {}).get("element_id") or "").strip()
        right_target_id = str((right_nodes[2] or {}).get("element_id") or "").strip()
        if not anchor_element_id or not left_target_id or not right_target_id:
            return None

        cypher = f"""
MATCH (a)
WHERE elementId(a) = $anchor_element_id
MATCH (a)-[:`{_safe_rel(left_rel_1)}`]-(left_mid:`{_safe_label(left_mid_label)}`)
      -[:`{_safe_rel(left_rel_2)}`]-(left_target:`{_safe_label(left_target_label)}`)
MATCH (a)-[:`{_safe_rel(right_rel_1)}`]-(right_mid:`{_safe_label(right_mid_label)}`)
      -[:`{_safe_rel(right_rel_2)}`]-(right_target:`{_safe_label(right_target_label)}`)
WHERE elementId(left_target) = $left_target_element_id
  AND elementId(right_target) = $right_target_element_id
  AND elementId(left_mid) <> elementId(right_mid)
RETURN DISTINCT coalesce(a.name, a.title, a.ticker, elementId(a)) AS anchor_value
LIMIT 5
""".strip()
        params = {
            "anchor_element_id": anchor_element_id,
            "left_target_element_id": left_target_id,
            "right_target_element_id": right_target_id,
        }
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
            return str(label).replace("`", "``")

        def _safe_rel(rel_type: str) -> str:
            return str(rel_type).replace("`", "``")

        pattern_parts = ["(n0)"]
        for idx, rel in enumerate(rels):
            rel_type = str((rel or {}).get("type") or "").strip()
            if not rel_type:
                return None
            next_node = nodes[idx + 1] if idx + 1 < len(nodes) else {}
            labels = next_node.get("labels") if isinstance(next_node, dict) else None
            label_hint = ""
            if isinstance(labels, list) and labels:
                label_hint = ":" + "`" + _safe_label(str(labels[0])) + "`"
            pattern_parts.append(f"-[:`{_safe_rel(rel_type)}`]-")
            pattern_parts.append(f"(n{idx + 1}{label_hint})")

        anchor_eid = str((nodes[0] or {}).get("element_id") or "").strip()
        target_eid = str((nodes[-1] or {}).get("element_id") or "").strip()
        if not anchor_eid or not target_eid:
            return None
        hop_count = len(rels)
        path_pattern = "".join(pattern_parts)
        cypher = f"""
MATCH p={path_pattern}
WHERE elementId(n0) = $anchor_element_id
  AND elementId(n{hop_count}) = $target_element_id
RETURN
  DISTINCT coalesce(n{hop_count}.name, n{hop_count}.title, n{hop_count}.ticker, elementId(n{hop_count})) AS target_value
LIMIT 10
""".strip()
        params = {
            "anchor_element_id": anchor_eid,
            "target_element_id": target_eid,
        }
        return cypher, params

    def _build_question_from_path(
        self,
        *,
        path: dict[str, Any],
        hop_count: int,
        complexity: str,
        existing_questions=None,
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

        def _rel_hint(rel_type: str) -> str:
            mapping = {
                "INVESTED_IN": "investment links",
                "OWNS": "ownership ties",
                "CEO_OF": "executive leadership ties",
                "MENTIONED_IN": "news co-mention signals",
                "OPERATES_IN_INDUSTRY": "industry affiliation",
                "PRODUCES": "product portfolio relations",
                "WORKS_AT": "employment links",
                "SUPPLIES": "supply-chain links",
                "PARTNERS_WITH": "partnership signals",
            }
            return mapping.get(rel_type, "indirect relationship signals")

        def _node_key_fact(node: dict[str, Any]) -> str:
            props = node.get("props") if isinstance(node, dict) else None
            if not isinstance(props, dict):
                return ""
            for key in ("industry", "sector", "country", "region", "city", "date", "year", "category"):
                value = props.get(key)
                if value not in (None, ""):
                    return f"{key}={value}"
            return ""

        def _path_business_clues(path_nodes: list[dict[str, Any]]) -> list[str]:
            clues: list[str] = []
            for node in path_nodes[1:-1]:
                name = _node_name(node)
                label = _node_label(node)
                fact = _node_key_fact(node)
                if name and name != "the anchor entity":
                    clues.append(f"{label} {name}")
                if fact:
                    clues.append(f"{label} with {fact}")
                if len(clues) >= 4:
                    break
            return clues

        def _clean_question(text: str) -> str:
            q = str(text or "").strip()
            q = re.sub(r"^['\"`]+|['\"`]+$", "", q).strip()
            if "\n" in q:
                q = q.splitlines()[0].strip()
            if not q.endswith("?"):
                q = q.rstrip(".") + "?"
            return q

        def _looks_too_abstract(question_text: str) -> bool:
            lowered = question_text.lower()
            banned = (
                "indirectly connected",
                "chain of",
                "intermediate",
                "through exactly",
                "relationship chain",
                "hops",
            )
            return any(token in lowered for token in banned)

        anchor = nodes[0]
        target = nodes[-1]
        anchor_name = _node_name(anchor)
        target_label = _node_label(target)
        rel_types = [str((r or {}).get("type") or "RELATED_TO") for r in rels]
        rel_hints = []
        seen_hints: set[str] = set()
        for rel_type in rel_types:
            hint = _rel_hint(rel_type)
            if hint in seen_hints:
                continue
            seen_hints.add(hint)
            rel_hints.append(hint)
        hints_text = ", ".join(rel_hints[:3]) if rel_hints else "indirect graph signals"
        clues = _path_business_clues(nodes)
        clues_text = "; ".join(clues) if clues else "no extra clues"

        existing_block = ""
        existing_list = [str(q).strip() for q in (existing_questions or []) if str(q).strip()]
        if existing_list:
            existing_block = "\n".join(f"- {q}" for q in existing_list[-50:])

        prompt = f"""
Write exactly one natural-sounding English benchmark question.
The question must be answerable by a graph query and must target exactly one {target_label}.

Facts you may use:
- Anchor entity: {anchor_name}
- Required reasoning depth: {hop_count} hops
- Relevant evidence themes: {hints_text}
- Concrete path clues: {clues_text}

Constraints:
1) One sentence, English, business-analyst tone.
2) Do NOT mention graph jargon: graph, node, edge, relationship, hop, cypher, chain.
3) Do NOT reveal the final target value directly.
4) Avoid abstract wording like "indirectly connected", "intermediate firms", or "chain of relationships".
5) Mention at least one concrete named entity from the facts.
6) Keep the intent aligned with complexity "{complexity}".
7) Avoid very similar wording to existing questions.

Existing questions to avoid:
{existing_block if existing_block else "- (none)"}
"""
        response = self.llm.generate_response(
            "You create natural benchmark questions for enterprise graph QA.",
            prompt,
        )
        question = _clean_question(response)
        if question and not _looks_too_abstract(question):
            return question
        return (
            f"Which {target_label} is most likely implicated in the same business context as "
            f"{anchor_name}, considering {hints_text}?"
        )

    def generate_aggregation_pairs(self, schema, data_samples, num_questions=2, existing_questions=None):
        print(f"Генерация {num_questions} aggregation-вопросов...")
        return self._generate_by_prompt_builder(
            build_aggregation_prompts, schema, data_samples, num_questions, existing_questions=existing_questions
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
            anchor_element_id = str(local_context.get("anchor_element_id") or "").strip()
            deterministic = self._build_cross_branch_cypher(
                anchor_element_id=anchor_element_id,
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
