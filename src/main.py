from dotenv import load_dotenv

from benchmark_generator.generator import BenchmarkGenerator

load_dotenv()


if __name__ == "__main__":
    # Запуск генератора
    generator = BenchmarkGenerator()
    generator.run(
        target_size=20,
        sample_entities_per_type=10,
        intra_type_workers=4,
        per_type_targets={
            "simple": 5,
            "multi-hop-2": 5,
            "multi-hop-3": 5,
            "aggregation": 3,
            "subgraph-deep-analytics": 2,
        },
    )