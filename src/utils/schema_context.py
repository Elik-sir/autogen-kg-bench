
def get_schema(db_manager):
    """Извлекает схему БД, чтобы LLM не галлюцинировала значения."""
    print("Извлечение схемы данных...")

    schema_result = db_manager.run_query("CALL apoc.meta.schema() YIELD value RETURN value")
    if not schema_result:
        raise ValueError("APOC не установлен или база пуста.")
    schema = schema_result[0]["value"]

    return schema

def get_samples(db_manager):
    """Извлекает примеры реальных данных, чтобы LLM не галлюцинировала значения."""
    print("Извлечение примеров данных...")

    samples_query = """
    CALL db.labels() YIELD label
    CALL apoc.cypher.run('MATCH (n:`'+label+'`) RETURN properties(n) AS props LIMIT 2', {})
    YIELD value
    RETURN label, value.props AS sample_properties
    """
    try:
        samples_result = db_manager.run_query(samples_query)
        samples = {res["label"]: res["sample_properties"] for res in samples_result}
    except Exception as error:
        print(f"Не удалось получить примеры (возможно нет APOC): {error}")
        samples = "Примеры недоступны"

    return samples