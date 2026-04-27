from dotenv import load_dotenv
import json
import re
from difflib import SequenceMatcher
from neo4j_manager import Neo4jManager
from llm_client import LLMClient
from utils.prompt_builder import (
    build_aggregation_prompts,
    build_cross_branch_prompts,
    build_multi_hop_prompts,
    build_same_type_common_prompts,
    build_simple_prompts,
    build_subgraph_deep_analytics_prompts,
)
from utils.same_type_common_context import find_same_type_common_contexts
from utils.llm_response_parser import parse_qa_pairs_response
from utils.benchmark_validation import is_trivial_self_return, result_to_ground_truth
from utils.schema_context import get_schema, get_samples
from utils.company_subgraph_context import build_company_subgraph_contexts

load_dotenv()


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


class BenchmarkGenerator:
    def __init__(self):
        self.db = Neo4jManager()
        self.llm = LLMClient()

    def _generate_by_prompt_builder(self, prompt_builder, schema, data_samples, num_questions):
        system_prompt, user_prompt = prompt_builder(schema, data_samples, num_questions)
        response = self.llm.generate_response(system_prompt, user_prompt)
        return parse_qa_pairs_response(response)

    def generate_simple_pairs(self, schema, data_samples, num_questions=2):
        print(f"Генерация {num_questions} simple-вопросов...")
        return self._generate_by_prompt_builder(
            build_simple_prompts, schema, data_samples, num_questions
        )

    def generate_multi_hop_pairs(self, schema, data_samples, num_questions=2):
        print(f"Генерация {num_questions} multi-hop-вопросов...")
        return self._generate_by_prompt_builder(
            build_multi_hop_prompts, schema, data_samples, num_questions
        )

    def generate_aggregation_pairs(self, schema, data_samples, num_questions=2):
        print(f"Генерация {num_questions} aggregation-вопросов...")
        return self._generate_by_prompt_builder(
            build_aggregation_prompts, schema, data_samples, num_questions
        )

    def generate_cross_branch_pairs(self, schema, data_samples, num_questions=2):
        print(f"Генерация {num_questions} cross-branch-вопросов...")
        return self._generate_by_prompt_builder(
            build_cross_branch_prompts, schema, data_samples, num_questions
        )

    def generate_same_type_common_pairs(self, schema, data_samples, num_questions=2):
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
            attempts += 1
            ctx = contexts[ctx_i % len(contexts)]
            ctx_i += 1
            system_prompt, user_prompt = build_same_type_common_prompts(
                schema, data_samples, ctx
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

    def generate_subgraph_deep_analytics_pairs(self, schema, num_questions=3):
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
        ctx_i = 0
        while len(out) < num_questions and attempts < max_attempts:
            attempts += 1
            ctx = contexts_for_prompt[ctx_i % len(contexts_for_prompt)]
            ctx_i += 1
            generated = self._generate_by_prompt_builder(
                build_subgraph_deep_analytics_prompts, schema, [ctx], 1
            )
            if isinstance(generated, dict):
                generated = [generated]
            if not generated:
                continue
            item = generated[0]
            if not isinstance(item, dict):
                continue

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
            item["useful_context"] = ctx.get("useful_context", "")
            out.append(item)

        if len(out) < num_questions:
            print(
                f"[ПРЕДУПРЕЖДЕНИЕ] subgraph-deep-analytics: получено {len(out)}/{num_questions} "
                f"после {attempts} попыток (пустые или невалидные ответы LLM)."
            )
        return out

    def validate_and_build_benchmark(self, generated_items, seen_exact_questions=None, seen_normalized_questions=None):
        """Выполняет Cypher в базе. Если есть результат -> сохраняем в бенчмарк."""
        print("Валидация запросов в Neo4j...")
        benchmark_dataset =[]
        seen_exact_questions = seen_exact_questions if seen_exact_questions is not None else set()
        seen_normalized_questions = seen_normalized_questions if seen_normalized_questions is not None else []

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
                    # Для subgraph-deep-analytics это debug-запрос, не источник ground_truth.
                    result = self.db.run_query(cypher_query, params)
                    if not result and not (debug_only_cypher and has_precomputed_context):
                        print(f"[ПРОПУСК] Запрос вернул 0 строк: {question}")
                        continue

                # Если ground_truth уже подготовлен заранее, используем его.
                if not has_precomputed_context:
                    item["ground_truth"] = result_to_ground_truth(question, result)
                benchmark_dataset.append(item)
                seen_exact_questions.add(normalized_question)
                seen_normalized_questions.append(normalized_question)
                print(f"[УСПЕХ] Добавлен вопрос ({item['complexity']}): {question}")
                
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
        """Генерирует бенчмарк по типам по очереди: simple -> multi-hop -> aggregation -> cross-branch -> subgraph."""
        schema = get_schema(self.db)
        data_samples = get_samples(self.db, per_label_limit=sample_entities_per_type)
        final_benchmark =[]
        seen_exact_questions = set()
        seen_normalized_questions = []

        generation_plan = [
            ("simple", lambda n: self.generate_simple_pairs(schema, data_samples, num_questions=n), 1),
            ("multi-hop", lambda n: self.generate_multi_hop_pairs(schema, data_samples, num_questions=n), 1),
            ("aggregation", lambda n: self.generate_aggregation_pairs(schema, data_samples, num_questions=n), 1),
            ("cross-branch", lambda n: self.generate_cross_branch_pairs(schema, data_samples, num_questions=n), 1),
            ("subgraph-deep-analytics", lambda n: self.generate_subgraph_deep_analytics_pairs(schema, num_questions=n), 1),
            # ("same-type-common", lambda n: self.generate_same_type_common_pairs(schema, data_samples, num_questions=n), 2),
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
                generated_items = generator_fn(request_n)
                if isinstance(generated_items, dict):
                    generated_items = [generated_items]

                valid_items = self.validate_and_build_benchmark(
                    generated_items or [],
                    seen_exact_questions=seen_exact_questions,
                    seen_normalized_questions=seen_normalized_questions,
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
            "simple": 5,
            "multi-hop": 10,
            "aggregation": 10,
            "cross-branch": 2,
            "subgraph-deep-analytics": 10,
        },
    )