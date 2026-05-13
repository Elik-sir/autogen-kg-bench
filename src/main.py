from dotenv import load_dotenv

from benchmark_generator.generator import BenchmarkGenerator

load_dotenv()


if __name__ == "__main__":
    # Запуск генератора
    generator = BenchmarkGenerator()
    generator.run(
        target_size=5,
        sample_entities_per_type=10,
        per_type_targets={
            "simple": 5,
            "multi-hop-2": 5,
            "multi-hop-3": 2,
            "multi-hop-4": 2,
            "aggregation": 5,
            "subgraph-deep-analytics":5,
        },
    )
