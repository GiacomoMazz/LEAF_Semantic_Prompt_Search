# test_pipeline.py
"""
Test script for the end-to-end search pipeline.

This tests combinations of:
- semantic / keyword / hybrid retrieval
- optional reranking
- optional metadata-aware scoring

Before running semantic or hybrid retrieval, make sure the Chroma index
has already been built.
"""

from config import get_config
from pipelines import SearchPipeline


def print_results(query: str, config, results: list[dict]) -> None:
    print(f"\nQuery: {query}")
    print(f"Retrieval mode: {config.retrieval_mode}")
    print(f"Use reranker: {config.use_reranker}")
    print(f"Use metadata scoring: {config.use_metadata_scoring}")
    print(f"Results shown: {len(results)}")

    for rank, result in enumerate(results, start=1):
        metadata = result.get("metadata", {})

        print("\n" + "-" * 80)
        print(f"Rank: {rank}")
        print(f"ID: {result.get('id')}")
        print(f"Title: {metadata.get('title')}")
        print(f"Category: {metadata.get('category')}")
        print(f"Subcategory: {metadata.get('subcategory')}")
        print(f"Tags: {metadata.get('tags')}")
        print(f"Difficulty: {metadata.get('difficulty')}")

        print(f"Semantic score: {result.get('semantic_score', 0.0):.4f}")
        print(f"Keyword score: {result.get('keyword_score', 0.0):.4f}")
        print(f"Hybrid score: {result.get('hybrid_score', 0.0):.4f}")

        print(f"Reranker score: {result.get('reranker_score', 0.0):.4f}")

        print(f"Search score: {result.get('search_score', 0.0):.4f}")
        print(f"Metadata score: {result.get('metadata_score', 0.0):.4f}")

        print(f"Final score: {result.get('final_score', 0.0):.4f}")


def main():
    config = get_config(
        embedding_model_key="minilm",

        # options: "semantic", "keyword", "hybrid"
        retrieval_mode="hybrid",

        semantic_top_k=20,
        keyword_top_k=20,
        merged_top_k=30,
        final_top_k=5,

        semantic_weight=0.70,
        keyword_weight=0.30,

        use_reranker=True,
        use_metadata_scoring=True,

        search_score_weight=0.85,
        metadata_score_weight=0.15,
    )

    query = "prompt for debugging Python code"

    pipeline = SearchPipeline(config)
    results = pipeline.search(query)

    print_results(query, config, results)


if __name__ == "__main__":
    main()