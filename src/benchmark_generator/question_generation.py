from __future__ import annotations

import re
from typing import Any

from benchmark_generator.cypher_factory import (
    build_aggregation_candidates_from_path,
    build_multi_hop_candidate,
    build_simple_candidate_from_path,
)
from benchmark_generator.prompt_settings import ANCHORS_PER_LABEL_LIMIT, MAX_PATHS_PER_ANCHOR
from benchmark_generator.utils.anchor_subgraph_context import (
    build_anchor_subgraph_context,
    build_balanced_anchor_order,
    get_stratified_anchor_pool,
)
from benchmark_generator.utils.company_subgraph_context import build_company_subgraph_contexts
from benchmark_generator.utils.llm_response_parser import parse_qa_pairs_response
from benchmark_generator.utils.prompt_builder import (
    build_same_type_common_prompts,
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
                hop_count=1,
                max_paths_per_anchor=MAX_PATHS_PER_ANCHOR,
            )
            if not local_context:
                continue
            path = self._select_path_for_cypher(local_context)
            if not path:
                continue
            candidate = build_simple_candidate_from_path(path=path, path_index=self._multi_hop_path_cursor - 1)
            if not candidate:
                continue
            out.append(candidate.to_item())
        if len(out) < num_questions:
            print(
                f"[ПРЕДУПРЕЖДЕНИЕ] simple: получено {len(out)}/{num_questions} "
                f"после {attempts} попыток."
            )
        return out

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
        if hop_count not in {2, 3, 4}:
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
            candidate = build_multi_hop_candidate(
                path=path,
                hop_count=hop_count,
                complexity=complexity,
                path_index=self._multi_hop_path_cursor - 1,
            )
            if not candidate:
                continue
            out.append(candidate.to_item())

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

        def _normalize_text(text: str) -> str:
            normalized = str(text or "").strip().lower()
            normalized = re.sub(r"[^\w\s]", " ", normalized, flags=re.UNICODE)
            normalized = re.sub(r"\s+", " ", normalized).strip()
            return normalized

        def _target_aliases(node: dict[str, Any]) -> list[str]:
            props = node.get("props") if isinstance(node, dict) else None
            if not isinstance(props, dict):
                return []
            aliases: list[str] = []
            for key in ("name", "title", "ticker", "id", "uuid", "symbol"):
                value = props.get(key)
                if value in (None, ""):
                    continue
                aliases.append(str(value).strip())
            return [x for x in aliases if x]

        def _question_leaks_target(question_text: str, aliases: list[str]) -> bool:
            nq = _normalize_text(question_text)
            if not nq:
                return False
            for alias in aliases:
                na = _normalize_text(alias)
                if len(na) < 3:
                    continue
                if na and na in nq:
                    return True
            return False

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
        target_aliases = _target_aliases(target)
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
8) Do NOT mention any direct target aliases in the question.

Existing questions to avoid:
{existing_block if existing_block else "- (none)"}

Target aliases forbidden in question:
{", ".join(target_aliases) if target_aliases else "(none)"}
"""
        response = self.llm.generate_response(
            "You create natural benchmark questions for enterprise graph QA.",
            prompt,
        )
        question = _clean_question(response)
        if (
            question
            and not _looks_too_abstract(question)
            and not _question_leaks_target(question, target_aliases)
        ):
            return question
        return (
            f"Which {target_label} is most likely implicated in the same business context as "
            f"{anchor_name}, considering {hints_text}?"
        )

    def generate_aggregation_pairs(self, schema, data_samples, num_questions=2, existing_questions=None):
        print(f"Генерация {num_questions} aggregation-вопросов...")
        out: list[dict[str, Any]] = []
        max_attempts = max(num_questions * 10, 22)
        attempts = 0
        while len(out) < num_questions and attempts < max_attempts:
            attempts += 1
            anchor = self._next_anchor()
            if not anchor:
                break
            local_context = build_anchor_subgraph_context(
                self.db,
                anchor=anchor,
                hop_count=1,
                max_paths_per_anchor=MAX_PATHS_PER_ANCHOR,
            )
            if not local_context:
                continue
            path = self._select_path_for_cypher(local_context)
            if not path:
                continue
            candidates = build_aggregation_candidates_from_path(
                path=path, path_index=self._multi_hop_path_cursor - 1
            )
            if not candidates:
                continue
            for candidate in candidates:
                out.append(candidate.to_item())
                if len(out) >= num_questions:
                    break
        if len(out) < num_questions:
            print(
                f"[ПРЕДУПРЕЖДЕНИЕ] aggregation: получено {len(out)}/{num_questions} "
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
