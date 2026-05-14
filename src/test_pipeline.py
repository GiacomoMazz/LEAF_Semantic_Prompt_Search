# test_pipeline.py

from config import get_config
from pipelines import SearchPipeline


def main():
    config = get_config(
        embedding_model_key="minilm",
        retrieval_mode="hybrid",
        semantic_top_k=20,
        keyword_top_k=20,
        merged_top_k=30,
        final_top_k=5,
        semantic_weight=0.70,
        keyword_weight=0.30,
        use_reranker=True,
    )

    query = "prompt for debugging Python code"

    pipeline = SearchPipeline(config)
    results = pipeline.search(query)

    print(f"\nQuery: {query}")
    print(f"Retrieval mode: {config.retrieval_mode}")
    print(f"Use reranker: {config.use_reranker}")

    for rank, result in enumerate(results, start=1):
        metadata = result["metadata"]

        print("\n" + "-" * 80)
        print(f"Rank: {rank}")
        print(f"ID: {result['id']}")
        print(f"Title: {metadata.get('title')}")
        print(f"Category: {metadata.get('category')}")
        print(f"Tags: {metadata.get('tags')}")
        print(f"Semantic score: {result.get('semantic_score', 0.0):.4f}")
        print(f"Keyword score: {result.get('keyword_score', 0.0):.4f}")
        print(f"Hybrid score: {result.get('hybrid_score', 0.0):.4f}")
        print(f"Reranker score: {result.get('reranker_score', 0.0):.4f}")
        print(f"Final score: {result.get('final_score', 0.0):.4f}")


if __name__ == "__main__":
    main()