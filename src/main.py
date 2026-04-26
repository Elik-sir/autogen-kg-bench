from dotenv import load_dotenv
import json
from neo4j_manager import Neo4jManager
from llm_client import LLMClient
from utils.prompt_builder import (
    build_aggregation_prompts,
    build_cross_branch_prompts,
    build_multi_hop_prompts,
    build_simple_prompts,
    build_subgraph_deep_analytics_prompts,
)
from utils.llm_response_parser import parse_qa_pairs_response
from utils.benchmark_validation import is_trivial_self_return, result_to_ground_truth
from utils.schema_context import get_schema, get_samples
from utils.company_subgraph_context import build_company_subgraph_contexts

load_dotenv()

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

    def validate_and_build_benchmark(self, generated_items):
        """Выполняет Cypher в базе. Если есть результат -> сохраняем в бенчмарк."""
        print("Валидация запросов в Neo4j...")
        benchmark_dataset =[]

        for item in generated_items:
            cypher_query = item.get("cypher", "")
            question = item.get("question", "")
            params = item.get("params")
            has_precomputed_context = bool(item.get("ground_truth"))
            debug_only_cypher = bool(item.get("debug_only_cypher"))
            
            try:
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
                print(f"[УСПЕХ] Добавлен вопрос ({item['complexity']}): {question}")
                
            except Exception as e:
                # Если синтаксическая ошибка в Cypher - бракуем
                print(f"[ОШИБКА SYNTAX] {e} | Query: {cypher_query}")

        return benchmark_dataset

    def run(self, target_size=5, output_file="graphrag_benchmark.json"):
        """Основной цикл. Крутимся, пока не наберем нужное количество валидных пар."""
        schema = get_schema(self.db)
        data_samples = get_samples(self.db)
        final_benchmark =[]
        
        attempt = 1
        while len(final_benchmark) < target_size and attempt <= 3:
            print(f"\n--- Итерация {attempt} (Собрано {len(final_benchmark)}/{target_size}) ---")

            # Каждый тип генерируется отдельной функцией и своим промптом.
            generated_items = []
            generated_items.extend(self.generate_simple_pairs(schema, data_samples, num_questions=2))
            generated_items.extend(self.generate_multi_hop_pairs(schema, data_samples, num_questions=3))
            generated_items.extend(self.generate_aggregation_pairs(schema, data_samples, num_questions=4))
            generated_items.extend(self.generate_cross_branch_pairs(schema, data_samples, num_questions=3))
            generated_items.extend(self.generate_subgraph_deep_analytics_pairs(schema, num_questions=5))
            
            valid_items = self.validate_and_build_benchmark(generated_items)
            final_benchmark.extend(valid_items)
            
            attempt += 1

        
        # Сохраняем в файл
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_benchmark, f, ensure_ascii=False, indent=2)
            
        print(f"\nГотово! Бенчмарк на {len(final_benchmark)} вопросов сохранен в {output_file}")
        self.db.close()


if __name__ == "__main__":
    # Запуск генератора
    generator = BenchmarkGenerator()
    # Сгенерируем 5 качественных вопросов для начала
    generator.run(target_size=3)