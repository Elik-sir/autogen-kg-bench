from dotenv import load_dotenv
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from neo4j_manager import Neo4jManager
from llm_client import LLMClient
from utils.prompt_builder import (
    build_aggregation_prompts,
    build_multi_hop_prompts,
    build_same_type_common_prompts,
    build_simple_prompts,
    build_subgraph_deep_analytics_prompts,
)
from utils.multi_hop_context import find_multi_hop_path_contexts
from utils.same_type_common_context import find_same_type_common_contexts
from utils.llm_response_parser import parse_qa_pairs_response
from utils.benchmark_validation import (
    is_insufficient_answer,
    is_trivial_self_return,
    result_to_ground_truth,
)
from utils.schema_context import get_schema, get_samples
from utils.company_subgraph_context import build_company_subgraph_contexts

load_dotenv()

GROUND_TRUTH_ROW_LIMIT = max(1, int(os.getenv("BENCHMARK_GROUND_TRUTH_LIMIT", "10")))


def normalize_question_text(text: str) -> str:
    text = str(text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    # Убираем пунктуацию, чтобы "?" и "," не мешали дедупликации
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    return text


def is_near_duplicate_question(question: str, seen_normalized_questions: list[str], threshold: float = 0.92) -> bool:
    normalized = normalize_question_text(question)
    if not normalized:
        return True
    return any(
        SequenceMatcher(None, normalized, seen).ratio() >= threshold
        for seen in seen_normalized_questions
    )


def _is_zero_like_value(value) -> bool:
    return isinstance(value, (int, float)) and value == 0


def _row_is_empty_like(row: dict) -> bool:
    if not isinstance(row, dict) or not row:
        return True
    values = list(row.values())
    return all(value is None or _is_zero_like_value(value) for value in values)


def _should_skip_low_signal_result(cypher_query: str, result: list[dict]) -> bool:
    """Отбрасывает low-signal агрегатные результаты вида None/0."""
    if not result:
        return True
    if not all(isinstance(row, dict) for row in result):
        return False

    has_aggregate = bool(
        re.search(r"\b(count|sum|avg|min|max)\s*\(", cypher_query or "", flags=re.IGNORECASE)
    )
    if has_aggregate and all(_row_is_empty_like(row) for row in result):
        return True
    return False


def _append_limit_if_missing(cypher_query: str, row_limit: int) -> str:
    """Добавляет LIMIT к Cypher, если его нет, чтобы не раздувать ground_truth."""
    q = str(cypher_query or "").strip()
    if not q:
        return q
    if re.search(r"\blimit\s+\d+\b", q, flags=re.IGNORECASE):
        return q
    # UNION обычно требует отдельного LIMIT по подзапросам; не трогаем автоматически.
    if re.search(r"\bunion\b", q, flags=re.IGNORECASE):
        return q
    q = q.rstrip(";")
    return f"{q} LIMIT {row_limit}"


def _safe_ident(name: str) -> str:
    return str(name or "").replace("`", "``")


class BenchmarkGenerator:
    def __init__(self):
        self.db = Neo4jManager()
        self.llm = LLMClient()
        self.generation_workers = max(1, int(os.getenv("BENCHMARK_GENERATION_WORKERS", "4")))
        self.multi_hop_scan_limit = max(100, int(os.getenv("BENCHMARK_MULTI_HOP_SCAN_LIMIT", "1200")))
        # Сквозные курсоры по контекстам между вызовами генератора.
        self._subgraph_ctx_cursor = 0
        self._multi_hop_ctx_cursor = {2: 0, 3: 0}

    def _build_answer_from_context(self, question: str, ground_truth: str, fallback: str = "") -> str:
        """
        Формирует эталонный answer по тому же принципу, что и subgraph-deep-analytics:
        LLM получает вопрос + контекст и возвращает краткий проверяемый ответ.
        """
        if not str(ground_truth or "").strip():
            return str(fallback or "").strip()
        system_prompt = (
            "You are an analyst who writes reference answers for a GraphRAG benchmark. "
            "Answer only from the given context—no invention or outside knowledge. "
            "If the context contains a direct answer, you must return it and must not claim insufficient data."
        )
        user_prompt = f"""
Question:
{question}

Context (ground_truth):
{ground_truth}

Produce a short, precise reference answer based only on this context.

MANDATORY RULES:
1) If the context contains a direct answer, you must return it.
2) Do not write "no information", "cannot determine", "insufficient data", or similar hedges
   when the context contains relevant entities.
3) Do not invent facts: use only what is explicitly present in the context.
4) For questions like "who works at which company", return pairs as "<employee>, <company>" (one pair per line).
5) If there is truly no relevant data, reply only with: "No data for this query."
6) No explanations or markdown—only the final answer text.
7) Write the entire answer in English (keep proper names and literals as in the context).
"""
        response = self.llm.generate_response(system_prompt, user_prompt)
        if response is None:
            return str(fallback or ground_truth).strip()
        answer = str(response).strip()
        return answer or str(fallback or ground_truth).strip()

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

    @staticmethod
    def _normalize_parsed_items(parsed) -> list[dict]:
        if isinstance(parsed, dict):
            parsed = [parsed]
        if not isinstance(parsed, list):
            return []
        return [item for item in parsed if isinstance(item, dict)]

    @staticmethod
    def _generate_with_fresh_llm(system_prompt: str, user_prompt: str):
        worker_llm = LLMClient()
        response = worker_llm.generate_response(system_prompt, user_prompt)
        return parse_qa_pairs_response(response)

    def _run_prompt_batch_parallel(self, prompts: list[tuple[str, str]]) -> list:
        if not prompts:
            return []
        if self.generation_workers <= 1 or len(prompts) == 1:
            out = []
            for system_prompt, user_prompt in prompts:
                out.append(self._generate_with_fresh_llm(system_prompt, user_prompt))
            return out

        max_workers = min(self.generation_workers, len(prompts))
        out: list = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._generate_with_fresh_llm, system_prompt, user_prompt)
                for system_prompt, user_prompt in prompts
            ]
            for future in futures:
                try:
                    out.append(future.result())
                except Exception as e:
                    print(f"[ПРЕДУПРЕЖДЕНИЕ] Ошибка генерации в потоке: {e}")
        return out

    def _build_multi_hop_cypher_from_context(self, ctx: dict) -> tuple[str, dict]:
        nodes = ctx.get("nodes") or []
        relationships = [str(r) for r in (ctx.get("relationships") or [])]
        hop_count = int(ctx.get("hop_count") or 0)
        if len(nodes) != hop_count + 1 or len(relationships) != hop_count:
            return str(ctx.get("seed_cypher") or ""), {}

        start_useful = nodes[0].get("useful_props") or {}
        anchor_field = next(
            (k for k in ("ticker", "name", "title") if start_useful.get(k) not in ("", None)),
            None,
        )
        if not anchor_field:
            return str(ctx.get("seed_cypher") or ""), {}
        anchor_value = start_useful[anchor_field]

        answer_fields = [str(f) for f in (ctx.get("answer_fields") or []) if str(f).strip()]
        answer_field = answer_fields[0] if answer_fields else "name"

        aliases = ["a", "b", "c", "d"][: hop_count + 1]
        pattern_parts = []
        for i, alias in enumerate(aliases):
            labels = [str(lbl) for lbl in (nodes[i].get("labels") or []) if str(lbl).strip()]
            label_suffix = "".join(f":`{_safe_ident(lbl)}`" for lbl in labels)
            pattern_parts.append(f"({alias}{label_suffix})")
            if i < hop_count:
                rel = _safe_ident(relationships[i])
                pattern_parts.append(f"-[:`{rel}`]-")

        end_alias = aliases[-1]
        query = (
            "MATCH "
            + "".join(pattern_parts)
            + f" WHERE a.`{_safe_ident(anchor_field)}` = $anchor_value"
            + f" AND {end_alias}.`{_safe_ident(answer_field)}` IS NOT NULL"
            + f" RETURN {end_alias}.`{_safe_ident(answer_field)}` AS answer"
            + " LIMIT 10"
        )
        return query, {"anchor_value": anchor_value}

    def generate_simple_pairs(self, schema, data_samples, num_questions=2, existing_questions=None):
        print(f"Генерация {num_questions} simple-вопросов...")
        if num_questions <= 1 or self.generation_workers <= 1:
            return self._generate_by_prompt_builder(
                build_simple_prompts, schema, data_samples, num_questions, existing_questions=existing_questions
            )
        prompts = [
            build_simple_prompts(schema, data_samples, 1, existing_questions=existing_questions)
            for _ in range(num_questions)
        ]
        parsed_batches = self._run_prompt_batch_parallel(prompts)
        out: list[dict] = []
        for parsed in parsed_batches:
            items = self._normalize_parsed_items(parsed)
            if items:
                out.append(items[0])
        return out

    def generate_multi_hop_pairs(
        self,
        schema,
        data_samples,
        num_questions=2,
        hop_count: int = 2,
        existing_questions=None,
    ):
        if hop_count not in (2, 3):
            raise ValueError("hop_count must be 2 or 3")
        complexity = f"multi-hop-{hop_count}"
        print(f"Генерация {num_questions} {complexity}-вопросов...")

        path_contexts = find_multi_hop_path_contexts(
            self.db,
            hop_count,
            max_contexts=max(num_questions * 10, 24),
            scan_limit=self.multi_hop_scan_limit,
        )
        if not path_contexts:
            print(
                f"[ПРОПУСК] Нет реальных {hop_count}-hop путей в Neo4j для {complexity}."
            )
            return []

        out: list = []
        max_attempts = max(num_questions * 5, len(path_contexts) * 3, 12)
        attempts = 0
        cursor = self._multi_hop_ctx_cursor.get(hop_count, 0) % len(path_contexts)
        while len(out) < num_questions and attempts < max_attempts:
            remaining = num_questions - len(out)
            attempts_left = max_attempts - attempts
            batch_size = min(self.generation_workers, remaining, attempts_left)
            prompts: list[tuple[str, str]] = []
            batch_contexts: list[dict] = []
            for _ in range(batch_size):
                ctx = path_contexts[cursor]
                cursor = (cursor + 1) % len(path_contexts)
                batch_contexts.append(ctx)
                prompts.append(
                    build_multi_hop_prompts(
                        schema,
                        data_samples,
                        1,
                        existing_questions=existing_questions,
                        hop_count=hop_count,
                        path_context=ctx,
                    )
                )
            attempts += len(prompts)
            parsed_batches = self._run_prompt_batch_parallel(prompts)
            for i, parsed in enumerate(parsed_batches):
                items = self._normalize_parsed_items(parsed)
                if not items:
                    continue
                item = items[0]
                if not str(item.get("question", "")).strip():
                    continue
                ctx = batch_contexts[i % len(batch_contexts)]
                fixed_cypher, fixed_params = self._build_multi_hop_cypher_from_context(ctx)
                if not fixed_cypher:
                    continue
                item["cypher"] = fixed_cypher
                if fixed_params:
                    item["params"] = fixed_params
                item["complexity"] = complexity
                out.append(item)
                if len(out) >= num_questions:
                    break

        self._multi_hop_ctx_cursor[hop_count] = cursor

        if len(out) < num_questions:
            print(
                f"[ПРЕДУПРЕЖДЕНИЕ] {complexity}: получено {len(out)}/{num_questions} "
                f"после {attempts} попыток."
            )
        return out

    def generate_aggregation_pairs(self, schema, data_samples, num_questions=2, existing_questions=None):
        print(f"Генерация {num_questions} aggregation-вопросов...")
        if num_questions <= 1 or self.generation_workers <= 1:
            return self._generate_by_prompt_builder(
                build_aggregation_prompts, schema, data_samples, num_questions, existing_questions=existing_questions
            )
        prompts = [
            build_aggregation_prompts(schema, data_samples, 1, existing_questions=existing_questions)
            for _ in range(num_questions)
        ]
        parsed_batches = self._run_prompt_batch_parallel(prompts)
        out: list[dict] = []
        for parsed in parsed_batches:
            items = self._normalize_parsed_items(parsed)
            if items:
                out.append(items[0])
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
                "с общей сущностью в пределах 1–3 рёбер от каждого."
            )
            return []

        out: list = []
        max_attempts = max(num_questions * 5, len(contexts) * 3, 12)
        attempts = 0
        ctx_i = 0
        while len(out) < num_questions and attempts < max_attempts:
            remaining = num_questions - len(out)
            attempts_left = max_attempts - attempts
            batch_size = min(self.generation_workers, remaining, attempts_left)
            prompts: list[tuple[str, str]] = []
            for _ in range(batch_size):
                ctx = contexts[ctx_i % len(contexts)]
                ctx_i += 1
                prompts.append(
                    build_same_type_common_prompts(
                        schema, data_samples, ctx, existing_questions=existing_questions
                    )
                )
            attempts += len(prompts)
            parsed_batches = self._run_prompt_batch_parallel(prompts)
            for parsed in parsed_batches:
                items = self._normalize_parsed_items(parsed)
                if not items:
                    continue
                item = items[0]
                item["complexity"] = "same-type-common"
                out.append(item)
                if len(out) >= num_questions:
                    break

        if len(out) < num_questions:
            print(
                f"[ПРЕДУПРЕЖДЕНИЕ] same-type-common: получено {len(out)}/{num_questions} "
                f"после {attempts} попыток."
            )
        return out

    def generate_subgraph_deep_analytics_pairs(self, schema, num_questions=3, existing_questions=None):
        print(f"Генерация {num_questions} subgraph-deep-analytics-вопросов...")
        subgraph_contexts = build_company_subgraph_contexts(
            db_manager=self.db,
            schema=schema,
            anchors_limit=3,
        )
        if not subgraph_contexts:
            print("[ПРОПУСК] Не удалось собрать контексты подграфа компаний.")
            return []
        contexts_for_prompt = [
            ctx for ctx in subgraph_contexts if ctx.get("useful_context")
        ]
        if not contexts_for_prompt:
            print("[ПРОПУСК] Нет полезного контекста для subgraph-deep-analytics.")
            return []

        # Один вызов LLM на один контекст: иначе модель смешивает якоря, и ground_truth
        # по индексу не совпадает с answer.
        out = []
        max_attempts = max(num_questions * 4, len(contexts_for_prompt) * 3, 12)
        attempts = 0
        cursor = self._subgraph_ctx_cursor % len(contexts_for_prompt)
        while len(out) < num_questions and attempts < max_attempts:
            remaining = num_questions - len(out)
            attempts_left = max_attempts - attempts
            batch_size = min(self.generation_workers, remaining, attempts_left)
            prompts: list[tuple[str, str]] = []
            batch_contexts: list[dict] = []
            for _ in range(batch_size):
                ctx = contexts_for_prompt[cursor]
                cursor = (cursor + 1) % len(contexts_for_prompt)
                system_prompt, user_prompt = build_subgraph_deep_analytics_prompts(
                    schema, [ctx], 1, existing_questions=existing_questions
                )
                prompts.append((system_prompt, user_prompt))
                batch_contexts.append(ctx)
            attempts += len(prompts)
            parsed_batches = self._run_prompt_batch_parallel(prompts)

            # Привязываем результаты к контекстам по порядку подготовки батча.
            for i, parsed in enumerate(parsed_batches):
                items = self._normalize_parsed_items(parsed)
                if not items:
                    continue
                item = items[0]
                if not isinstance(item, dict):
                    continue
                ctx = batch_contexts[i % len(batch_contexts)]

                # Для этого типа `cypher` нужен только для debug-выгрузки подграфа.
                # `answer` — эталонный ответ от LLM; `ground_truth` — тот же useful_context,
                # что был в промпте (должен достаточен для проверки answer).
                item["complexity"] = "subgraph-deep-analytics"
                item["cypher"] = ctx.get("debug_cypher", "")
                item["params"] = ctx.get("debug_params", {})
                item["debug_only_cypher"] = True
                item["answer"] = str(item.get("answer", "")).strip()
                item["ground_truth"] = str(ctx.get("useful_context", "")).strip()
                item["subgraph_context"] = ctx.get("subgraph_context", "")
                out.append(item)
                if len(out) >= num_questions:
                    break

        self._subgraph_ctx_cursor = cursor

        if len(out) < num_questions:
            print(
                f"[ПРЕДУПРЕЖДЕНИЕ] subgraph-deep-analytics: получено {len(out)}/{num_questions} "
                f"после {attempts} попыток (пустые или невалидные ответы LLM)."
            )
        return out

    def validate_and_build_benchmark(
        self,
        generated_items,
        seen_exact_questions=None,
        seen_normalized_questions=None,
        output_file=None,
        existing_benchmark=None,
    ):
        """Выполняет Cypher в базе. Если есть результат -> сохраняем в бенчмарк."""
        print("Валидация запросов в Neo4j...")
        benchmark_dataset = []
        seen_exact_questions = seen_exact_questions if seen_exact_questions is not None else set()
        seen_normalized_questions = seen_normalized_questions if seen_normalized_questions is not None else []
        prefix = existing_benchmark if existing_benchmark is not None else []

        for item in generated_items:
            cypher_query = item.get("cypher", "")
            question = item.get("question", "")
            params = item.get("params")
            has_precomputed_context = bool(item.get("ground_truth"))
            debug_only_cypher = bool(item.get("debug_only_cypher"))
            
            try:
                normalized_question = normalize_question_text(question)
                if not normalized_question:
                    print("[ПРОПУСК] Пустой вопрос после нормализации.")
                    continue
                if normalized_question in seen_exact_questions:
                    print(f"[ПРОПУСК] Дубликат вопроса (exact): {question}")
                    continue
                if is_near_duplicate_question(question, seen_normalized_questions):
                    print(f"[ПРОПУСК] Дубликат вопроса (near): {question}")
                    continue

                if cypher_query and is_trivial_self_return(cypher_query):
                    print(f"[ПРОПУСК] Тривиальный запрос (WHERE/RETURN одного поля): {question}")
                    continue

                result = []
                if cypher_query:
                    if not debug_only_cypher:
                        limited_cypher = _append_limit_if_missing(
                            cypher_query,
                            row_limit=GROUND_TRUTH_ROW_LIMIT,
                        )
                        if limited_cypher != cypher_query:
                            item["cypher"] = limited_cypher
                            cypher_query = limited_cypher
                    # Для subgraph-deep-analytics это debug-запрос, не источник ground_truth.
                    result = self.db.run_query(cypher_query, params)
                    if not result and not (debug_only_cypher and has_precomputed_context):
                        print(f"[ПРОПУСК] Запрос вернул 0 строк: {question}")
                        continue
                    if has_precomputed_context:
                        pass
                    elif _should_skip_low_signal_result(cypher_query, result):
                        print(f"[ПРОПУСК] Low-signal результат (None/0): {question}")
                        continue

                # Если ground_truth уже подготовлен заранее, используем его.
                if not has_precomputed_context:
                    item["ground_truth"] = result_to_ground_truth(question, result)
                if is_insufficient_answer(str(item.get("ground_truth", ""))):
                    print(
                        f"[ПРОПУСК] Неинформативный ground_truth: {question} | "
                        f"ground_truth={item.get('ground_truth')!r}"
                    )
                    continue
                ground_truth_text = str(item.get("ground_truth", "")).strip()
                if debug_only_cypher and has_precomputed_context and str(item.get("answer", "")).strip():
                    # Для subgraph-deep-analytics сохраняем уже сгенерированный answer.
                    candidate_answer = str(item.get("answer", "")).strip()
                else:
                    candidate_answer = self._build_answer_from_context(
                        question=question,
                        ground_truth=ground_truth_text,
                        fallback=str(item.get("answer", "")),
                    )

                if is_insufficient_answer(candidate_answer):
                    print(f"[ПРОПУСК] Недостаточный answer: {question} | answer={candidate_answer!r}")
                    continue
                item["answer"] = candidate_answer
                benchmark_dataset.append(item)
                seen_exact_questions.add(normalized_question)
                seen_normalized_questions.append(normalized_question)
                print(f"[УСПЕХ] Добавлен вопрос ({item['complexity']}): {question}")
                if output_file is not None:
                    snapshot = [*prefix, *benchmark_dataset]
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(snapshot, f, ensure_ascii=False, indent=2)

            except Exception as e:
                # Если синтаксическая ошибка в Cypher - бракуем
                print(f"[ОШИБКА SYNTAX] {e} | Query: {cypher_query}")

        return benchmark_dataset

    def run(
        self,
        target_size=5,
        output_file="graphrag_benchmark.json",
        sample_entities_per_type=10,
        per_type_targets=None,
    ):
        """Генерирует бенчмарк по типам по очереди: simple -> multi-hop-2/3 -> aggregation -> subgraph."""
        schema = get_schema(self.db)
        data_samples = get_samples(self.db, per_label_limit=sample_entities_per_type)
        final_benchmark =[]
        seen_exact_questions = set()
        seen_normalized_questions = []

        generation_plan = [
            (
                "simple",
                lambda n, existing_questions=None: self.generate_simple_pairs(
                    schema, data_samples, num_questions=n, existing_questions=existing_questions
                ),
                1,
            ),
            (
                "multi-hop-2",
                lambda n, existing_questions=None: self.generate_multi_hop_pairs(
                    schema,
                    data_samples,
                    num_questions=n,
                    hop_count=2,
                    existing_questions=existing_questions,
                ),
                4,
            ),
            (
                "multi-hop-3",
                lambda n, existing_questions=None: self.generate_multi_hop_pairs(
                    schema,
                    data_samples,
                    num_questions=n,
                    hop_count=3,
                    existing_questions=existing_questions,
                ),
                4,
            ),
            (
                "aggregation",
                lambda n, existing_questions=None: self.generate_aggregation_pairs(
                    schema, data_samples, num_questions=n, existing_questions=existing_questions
                ),
                7,
            ),
            (
                "subgraph-deep-analytics",
                lambda n, existing_questions=None: self.generate_subgraph_deep_analytics_pairs(
                    schema, num_questions=n, existing_questions=existing_questions
                ),
                5,
            ),
            # ("same-type-common", lambda n, existing_questions=None: self.generate_same_type_common_pairs(schema, data_samples, num_questions=n, existing_questions=existing_questions), 2),
        ]

        if per_type_targets is None:
            base = target_size // len(generation_plan)
            remainder = target_size % len(generation_plan)
            per_type_targets = {
                type_name: base + (1 if i < remainder else 0)
                for i, (type_name, _, _) in enumerate(generation_plan)
            }

        # Поддерживаем только известные типы; если цель не задана — 0.
        per_type_targets = {
            type_name: int(max(0, per_type_targets.get(type_name, 0)))
            for type_name, _, _ in generation_plan
        }

        print("\nПлан генерации по типам:")
        for type_name, _, _ in generation_plan:
            print(f"- {type_name}: {per_type_targets[type_name]}")

        for type_name, generator_fn, batch_size in generation_plan:
            target_for_type = per_type_targets[type_name]
            if target_for_type <= 0:
                continue

            print(f"\n=== Этап: {type_name} (цель {target_for_type}) ===")
            collected_for_type = 0
            attempts = 0
            max_attempts = max(target_for_type * 8, 20)

            while collected_for_type < target_for_type and attempts < max_attempts:
                attempts += 1
                remaining = target_for_type - collected_for_type
                request_n = min(batch_size, remaining)
                existing_questions = [
                    str(item.get("question", "")).strip()
                    for item in final_benchmark
                    if item.get("complexity") == type_name and str(item.get("question", "")).strip()
                ]
                generated_items = generator_fn(request_n, existing_questions=existing_questions)
                if isinstance(generated_items, dict):
                    generated_items = [generated_items]

                valid_items = self.validate_and_build_benchmark(
                    generated_items or [],
                    seen_exact_questions=seen_exact_questions,
                    seen_normalized_questions=seen_normalized_questions,
                    output_file=output_file,
                    existing_benchmark=final_benchmark,
                )
                final_benchmark.extend(valid_items)
                added_for_type = sum(
                    1 for item in valid_items if item.get("complexity") == type_name
                )
                collected_for_type += added_for_type
                print(
                    f"[ПРОГРЕСС] {type_name}: +{added_for_type}, "
                    f"итого {collected_for_type}/{target_for_type} (попытка {attempts}/{max_attempts})"
                )

            if collected_for_type < target_for_type:
                print(
                    f"[ПРЕДУПРЕЖДЕНИЕ] Тип {type_name}: собрано {collected_for_type}/{target_for_type}. "
                    "Лимит попыток исчерпан."
                )

        
        # Сохраняем в файл
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_benchmark, f, ensure_ascii=False, indent=2)
            
        print(f"\nГотово! Бенчмарк на {len(final_benchmark)} вопросов сохранен в {output_file}")
        self.db.close()


if __name__ == "__main__":
    # Запуск генератора
    generator = BenchmarkGenerator()
    generator.run(
        target_size=30,
        sample_entities_per_type=10,
        per_type_targets={
            "simple": 0,
            "multi-hop-2": 0,
            "multi-hop-3": 5,
            "aggregation": 1,
            "subgraph-deep-analytics": 0,
        },
    )