from dotenv import load_dotenv
import json
from neo4j_manager import Neo4jManager
from llm_client import LLMClient
from utils.prompt_builder import build_generation_prompts
from utils.llm_response_parser import parse_qa_pairs_response
from utils.benchmark_validation import is_trivial_self_return, result_to_ideal_context
from utils.schema_context import get_schema, get_samples

load_dotenv()

class BenchmarkGenerator:
    def __init__(self):
        self.db = Neo4jManager()
        self.llm = LLMClient()

    def generate_qa_pairs(self, schema, data_samples, num_questions=10):
        """Просит LLM сгенерировать вопросы и Cypher запросы."""
        system_prompt, user_prompt = build_generation_prompts(schema, data_samples, num_questions)
        
        print(f"Генерация {num_questions} кандидатов через LLM...")
        response = self.llm.generate_response(system_prompt, user_prompt)
        return parse_qa_pairs_response(response)

    def validate_and_build_benchmark(self, generated_items):
        """Выполняет Cypher в базе. Если есть результат -> сохраняем в бенчмарк."""
        print("Валидация запросов в Neo4j...")
        benchmark_dataset =[]

        for item in generated_items:
            cypher_query = item.get("cypher", "")
            question = item.get("question", "")
            
            try:
                if is_trivial_self_return(cypher_query):
                    print(f"[ПРОПУСК] Тривиальный запрос (WHERE/RETURN одного поля): {question}")
                    continue

                # Пытаемся выполнить запрос
                result = self.db.run_query(cypher_query)
                
                # Если результат пустой - бракуем вопрос
                if not result:
                    print(f"[ПРОПУСК] Запрос вернул 0 строк: {question}")
                    continue
                
                # Сохраняем успешный кейс
                item["ideal_context"] = result_to_ideal_context(question, result)
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
            
            # Просим чуть больше, так как часть отбракуется на этапе валидации
            items_to_generate = 5
            generated_items = self.generate_qa_pairs(schema, data_samples, items_to_generate)
            
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
    generator.run(target_size=15)