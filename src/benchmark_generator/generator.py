from __future__ import annotations

from llm_client import LLMClient
from neo4j_manager import Neo4jManager

from benchmark_generator.pipeline import run_generation_pipeline
from benchmark_generator.question_generation import QuestionGenerationEngine
from benchmark_generator.validation import validate_generated_items


class BenchmarkGenerator:
    """High-level facade for benchmark generation pipeline."""

    def __init__(self):
        self.db = Neo4jManager()
        self.llm = LLMClient()
        self.question_engine = QuestionGenerationEngine(db=self.db, llm=self.llm)

    def generate_simple_pairs(self, schema, data_samples, num_questions=2, existing_questions=None):
        return self.question_engine.generate_simple_pairs(
            schema=schema,
            data_samples=data_samples,
            num_questions=num_questions,
            existing_questions=existing_questions,
        )

    def generate_multi_hop_pairs(self, schema, data_samples, num_questions=2, existing_questions=None):
        # Backward-compatible alias for multi-hop-2.
        return self.question_engine.generate_multi_hop_pairs(
            schema=schema,
            data_samples=data_samples,
            num_questions=num_questions,
            existing_questions=existing_questions,
        )

    def generate_multi_hop_x_pairs(self, schema, hop_count=2, num_questions=2, existing_questions=None):
        return self.question_engine.generate_multi_hop_x_pairs(
            schema=schema,
            hop_count=hop_count,
            num_questions=num_questions,
            existing_questions=existing_questions,
        )

    def generate_aggregation_pairs(self, schema, data_samples, num_questions=2, existing_questions=None):
        return self.question_engine.generate_aggregation_pairs(
            schema=schema,
            data_samples=data_samples,
            num_questions=num_questions,
            existing_questions=existing_questions,
        )

    def generate_cross_branch_pairs(self, schema, data_samples, num_questions=2, existing_questions=None):
        return self.question_engine.generate_cross_branch_pairs(
            schema=schema,
            data_samples=data_samples,
            num_questions=num_questions,
            existing_questions=existing_questions,
        )

    def generate_same_type_common_pairs(self, schema, data_samples, num_questions=2, existing_questions=None):
        return self.question_engine.generate_same_type_common_pairs(
            schema=schema,
            data_samples=data_samples,
            num_questions=num_questions,
            existing_questions=existing_questions,
        )

    def generate_subgraph_deep_analytics_pairs(self, schema, num_questions=3, existing_questions=None):
        return self.question_engine.generate_subgraph_deep_analytics_pairs(
            schema=schema,
            num_questions=num_questions,
            existing_questions=existing_questions,
        )

    def validate_and_build_benchmark(
        self,
        generated_items,
        seen_exact_questions=None,
        seen_normalized_questions=None,
        output_file=None,
        existing_benchmark=None,
    ):
        return validate_generated_items(
            db=self.db,
            llm=self.llm,
            generated_items=generated_items,
            seen_exact_questions=seen_exact_questions,
            seen_normalized_questions=seen_normalized_questions,
            output_file=output_file,
            existing_benchmark=existing_benchmark,
        )

    def run(
        self,
        target_size=5,
        output_file="graphrag_benchmark.json",
        sample_entities_per_type=10,
        per_type_targets=None,
        strict_deterministic_mode: bool = False,
        intra_type_workers: int = 8,
    ):
        try:
            return run_generation_pipeline(
                question_engine=self.question_engine,
                validate_fn=self.validate_and_build_benchmark,
                db=self.db,
                target_size=target_size,
                output_file=output_file,
                sample_entities_per_type=sample_entities_per_type,
                per_type_targets=per_type_targets,
                strict_deterministic_mode=strict_deterministic_mode,
                intra_type_workers=intra_type_workers,
            )
        finally:
            self.db.close()
