from dotenv import load_dotenv
import json
import re
from neo4j_manager import Neo4jManager
from llm_client import LLMClient

load_dotenv()

class BenchmarkGenerator:
    def __init__(self):
        self.db = Neo4jManager()
        self.llm = LLMClient()

    def get_schema_and_samples(self):
        """Извлекает схему БД и примеры реальных данных, чтобы LLM не галлюцинировала значения."""
        print("Извлечение схемы и примеров данных...")
        
        # 1. Получаем схему
        schema_result = self.db.run_query("CALL apoc.meta.schema() YIELD value RETURN value")
        if not schema_result:
            raise ValueError("APOC не установлен или база пуста.")
        schema = schema_result[0]["value"]

        # 2. Получаем примеры реальных данных (по 1 узлу на каждый Label)
        # Это нужно, чтобы LLM использовала реальные 'name', 'id' и т.д.
        samples_query = """
        CALL db.labels() YIELD label
        CALL apoc.cypher.run('MATCH (n:`'+label+'`) RETURN properties(n) AS props LIMIT 1', {}) 
        YIELD value
        RETURN label, value.props AS sample_properties
        """
        try:
            samples_result = self.db.run_query(samples_query)
            samples = {res["label"]: res["sample_properties"] for res in samples_result}
        except Exception as e:
            print(f"Не удалось получить примеры (возможно нет APOC): {e}")
            samples = "Примеры недоступны"

        return json.dumps({"schema": schema, "data_samples": samples}, ensure_ascii=False)

    def generate_qa_pairs(self, context_str, num_questions=10):
        """Просит LLM сгенерировать вопросы и Cypher запросы."""
        system_prompt = (
            "Ты — Data Scientist. Твоя задача — создать бенчмарк для тестирования систем GraphRAG. "
            "Тебе даны схема графовой базы Neo4j и ПРИМЕРЫ реальных данных из нее. "
            "Ты должен вернуть строго валидный JSON-массив объектов. Без markdown, без текста до/после. "
            "Генерируй только вопросы, на которые можно дать однозначный ответ по данным графа."
        )

        user_prompt = f"""
        Вот структура базы и примеры реальных узлов:
        {context_str}

        Сгенерируй {num_questions} разнообразных вопросов к этой базе.
        Включай: 
        - Простые вопросы (поиск атрибута).
        - Многошаговые (multi-hop) вопросы (связь 2-3 узлов).
        - Агрегационные вопросы (COUNT, сортировка).

        КРИТИЧЕСКИ ВАЖНО: 
        В условиях WHERE используй ТОЛЬКО те значения свойств, которые ты видишь в секции 'data_samples'! 
        Иначе запрос вернет пустоту.
        
        Правила качества (обязательные):
        1) Вопрос должен быть конкретным, с привязкой к сущности/значению из data_samples.
           Плохо: "Как называется отрасль, представленная в базе данных?"
           Хорошо: "Чем занимается отрасль Электромобили?"
        2) Не делай тривиальные пары, где в WHERE фиксируется свойство и это же свойство возвращается.
           Запрещено: WHERE i.name='Электромобили' RETURN i.name
           Разрешено: WHERE i.name='Электромобили' RETURN i.description
        3) Если вопрос на существование ("Есть ли ...?"), используй Cypher c COUNT и верни факт существования:
           пример Cypher: MATCH (i:Industry {{name: 'Электромобили'}}) RETURN count(i) > 0 AS exists
        4) Для агрегаций избегай синтетических меток/паттернов вроде :ANY, используй только реальные Labels/Relations из схемы.
        5) Каждый вопрос должен иметь один понятный, проверяемый ответ.

        Формат JSON:[
          {{
            "complexity": "multi-hop",
            "question": "Естественный вопрос на русском языке...",
            "cypher": "MATCH ... RETURN ..."
          }}
        ]
        """
        
        print(f"Генерация {num_questions} кандидатов через LLM...")
        response = self.llm.generate_response(system_prompt, user_prompt)
        
        if not response:
            return[]

        # Очистка от markdown
        cleaned_text = response.strip()
        if cleaned_text.startswith("```"):
            match = re.search(r'```(?:json)?\s*(.*?)\s*```', cleaned_text, re.DOTALL | re.IGNORECASE)
            if match:
                cleaned_text = match.group(1).strip()

        try:
            return json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            print(f"Ошибка парсинга LLM ответа: {e}\nОтвет:\n{response}")
            return[]

    def _value_to_text(self, value):
        """Преобразует значение Neo4j в компактный человекочитаемый текст."""
        if isinstance(value, dict):
            return ", ".join(f"{k}: {self._value_to_text(v)}" for k, v in value.items())
        if isinstance(value, list):
            return ", ".join(self._value_to_text(v) for v in value)
        return str(value)

    def _is_trivial_self_return(self, cypher_query):
        """
        Отсекает тривиальные запросы вида:
        WHERE x.prop = ... RETURN x.prop
        """
        normalized = " ".join(cypher_query.strip().split())
        where_match = re.search(r"WHERE\s+([A-Za-z_]\w*)\.([A-Za-z_]\w*)\s*=", normalized, re.IGNORECASE)
        return_match = re.search(r"RETURN\s+([A-Za-z_]\w*)\.([A-Za-z_]\w*)\b", normalized, re.IGNORECASE)
        if not where_match or not return_match:
            return False
        return (
            where_match.group(1).lower() == return_match.group(1).lower()
            and where_match.group(2).lower() == return_match.group(2).lower()
        )

    def _result_to_ideal_context(self, question, result_rows):
        """Преобразует список словарей (результат Cypher) в текстовый контекст."""
        lowered_question = question.strip().lower()
        if lowered_question.startswith("есть ли"):
            first_row = result_rows[0] if result_rows else {}
            first_value = next(iter(first_row.values()), None)
            if isinstance(first_value, bool):
                return "Да, есть." if first_value else "Нет, не найдено."
            if isinstance(first_value, (int, float)):
                return "Да, есть." if first_value > 0 else "Нет, не найдено."
            return "Да, есть."

        row_texts = []
        for row in result_rows:
            row_text = ", ".join(
                f"{key}: {self._value_to_text(value)}"
                for key, value in row.items()
            )
            if row_text:
                row_texts.append(row_text)
        return "; ".join(row_texts)

    def validate_and_build_benchmark(self, generated_items):
        """Выполняет Cypher в базе. Если есть результат -> сохраняем в бенчмарк."""
        print("Валидация запросов в Neo4j...")
        benchmark_dataset =[]

        for item in generated_items:
            cypher_query = item.get("cypher", "")
            question = item.get("question", "")
            
            try:
                if self._is_trivial_self_return(cypher_query):
                    print(f"[ПРОПУСК] Тривиальный запрос (WHERE/RETURN одного поля): {question}")
                    continue

                # Пытаемся выполнить запрос
                result = self.db.run_query(cypher_query)
                
                # Если результат пустой - бракуем вопрос
                if not result:
                    print(f"[ПРОПУСК] Запрос вернул 0 строк: {question}")
                    continue
                
                # Сохраняем успешный кейс
                item["ideal_context"] = self._result_to_ideal_context(question, result)
                benchmark_dataset.append(item)
                print(f"[УСПЕХ] Добавлен вопрос ({item['complexity']}): {question}")
                
            except Exception as e:
                # Если синтаксическая ошибка в Cypher - бракуем
                print(f"[ОШИБКА SYNTAX] {e} | Query: {cypher_query}")

        return benchmark_dataset

    def run(self, target_size=5, output_file="graphrag_benchmark.json"):
        """Основной цикл. Крутимся, пока не наберем нужное количество валидных пар."""
        context_str = self.get_schema_and_samples()
        final_benchmark =[]
        
        attempt = 1
        while len(final_benchmark) < target_size and attempt <= 3:
            print(f"\n--- Итерация {attempt} (Собрано {len(final_benchmark)}/{target_size}) ---")
            
            # Просим чуть больше, так как часть отбракуется на этапе валидации
            items_to_generate = target_size - len(final_benchmark) + 3 
            generated_items = self.generate_qa_pairs(context_str, items_to_generate)
            
            valid_items = self.validate_and_build_benchmark(generated_items)
            final_benchmark.extend(valid_items)
            
            attempt += 1

        # Оставляем только нужное количество
        final_benchmark = final_benchmark[:target_size]
        
        # Сохраняем в файл
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_benchmark, f, ensure_ascii=False, indent=2)
            
        print(f"\nГотово! Бенчмарк на {len(final_benchmark)} вопросов сохранен в {output_file}")
        self.db.close()


if __name__ == "__main__":
    # Запуск генератора
    generator = BenchmarkGenerator()
    # Сгенерируем 5 качественных вопросов для начала
    generator.run(target_size=5)