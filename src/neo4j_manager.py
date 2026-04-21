import os
from neo4j import GraphDatabase

class Neo4jManager:
    def __init__(self):
        # Достаем переменные окружения напрямую
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "")
        
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def run_query(self, query, parameters=None):
        """Выполняет запрос и возвращает список записей (словарей)."""
        with self.driver.session(database=os.getenv("NEO4J_DB_NAME", "neo4j")) as session:
            result = session.run(query, parameters)
            return [record.data() for record in result]