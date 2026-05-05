from __future__ import annotations

import json
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
    build_cross_branch_prompts,
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
        prompt_builder = prompt_builders.get(hop_count)
        if prompt_builder is None:
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
            question = self._generate_question_for_fixed_path(
                schema=schema,
                local_context=local_context,
                complexity=complexity,
                hop_count=hop_count,
                fixed_cypher=cypher,
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
  coalesce(n{hop_count}.name, n{hop_count}.title, n{hop_count}.ticker, elementId(n{hop_count})) AS target_value,
  labels(n{hop_count}) AS target_labels
LIMIT 10
""".strip()
        params = {
            "anchor_element_id": anchor_eid,
            "target_element_id": target_eid,
        }
        return cypher, params

    def _generate_question_for_fixed_path(
        self,
        *,
        schema,
        local_context: dict[str, Any],
        complexity: str,
        hop_count: int,
        fixed_cypher: str,
        existing_questions=None,
    ) -> str:
        system_prompt, user_prompt = build_multi_hop_2_prompts(
            schema,
            local_context,
            1,
            existing_questions=existing_questions,
        )
        if hop_count == 3:
            system_prompt, user_prompt = build_multi_hop_3_prompts(
                schema, local_context, 1, existing_questions=existing_questions
            )
        elif hop_count == 4:
            system_prompt, user_prompt = build_multi_hop_4_prompts(
                schema, local_context, 1, existing_questions=existing_questions
            )

        guidance = f"""

=== FIXED CYPHER (DO NOT CHANGE) ===
{fixed_cypher}

Return JSON array with exactly 1 object and use this exact cypher text in "cypher":
[
  {{
    "complexity": "{complexity}",
    "question": "English question requiring exactly {hop_count} hops",
    "cypher": {json.dumps(fixed_cypher)}
  }}
]
"""
        response = self.llm.generate_response(system_prompt, user_prompt + guidance)
        parsed = parse_qa_pairs_response(response)
        if isinstance(parsed, dict):
            parsed = [parsed]
        if not parsed or not isinstance(parsed, list):
            return ""
        item = parsed[0] if parsed else {}
        if not isinstance(item, dict):
            return ""
        question = str(item.get("question", "")).strip()
        return question

    def generate_aggregation_pairs(self, schema, data_samples, num_questions=2, existing_questions=None):
        print(f"Генерация {num_questions} aggregation-вопросов...")
        return self._generate_by_prompt_builder(
            build_aggregation_prompts, schema, data_samples, num_questions, existing_questions=existing_questions
        )

    def generate_cross_branch_pairs(self, schema, data_samples, num_questions=2, existing_questions=None):
        print(f"Генерация {num_questions} cross-branch-вопросов...")
        return self._generate_by_prompt_builder(
            build_cross_branch_prompts, schema, data_samples, num_questions, existing_questions=existing_questions
        )

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

            # Для этого типа `cypher` нужен только для debug-выгрузки подграфа.
            # `answer` - эталонный ответ от LLM; `ground_truth` - тот же useful_context,
            # что был в промпте (должен достаточен для проверки answer).
            item["complexity"] = "subgraph-deep-analytics"
            item["cypher"] = ctx.get("debug_cypher", "")
            item["params"] = ctx.get("debug_params", {})
            item["debug_only_cypher"] = True
            item["answer"] = str(item.get("answer", "")).strip()
            item["ground_truth"] = str(ctx.get("useful_context", "")).strip()
            item["subgraph_context"] = ctx.get("subgraph_context", "")
            out.append(item)

        self._subgraph_ctx_cursor = cursor

        if len(out) < num_questions:
            print(
                f"[ПРЕДУПРЕЖДЕНИЕ] subgraph-deep-analytics: получено {len(out)}/{num_questions} "
                f"после {attempts} попыток (пустые или невалидные ответы LLM)."
            )
        return out
