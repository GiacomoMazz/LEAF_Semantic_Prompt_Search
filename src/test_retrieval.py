# test_retrieval.py
"""
Test script for semantic, keyword, and hybrid retrieval.

Before running semantic or hybrid retrieval, make sure the Chroma index
has already been built.
"""

from config import get_config
from data_loader import load_raw_data
from preprocessing import preprocess_records
from embedders import SentenceTransformerEmbedder
from vectorstores import ChromaVectorStore
from keyword_retriever import TfidfKeywordRetriever
from retrievers import SearchRetriever


def print_results(query: str, config, results: list[dict]) -> None:
    print(f"\nQuery: {query}")
    print(f"Retrieval mode: {config.retrieval_mode}")
    print(f"Results found: {len(results)}")

    for rank, result in enumerate(results[: config.final_top_k], start=1):
        metadata = result["metadata"]

        print("\n" + "-" * 80)
        print(f"Rank: {rank}")
        print(f"ID: {result['id']}")
        print(f"Title: {metadata.get('title')}")
        print(f"Category: {metadata.get('category')}")
        print(f"Subcategory: {metadata.get('subcategory')}")
        print(f"Tags: {metadata.get('tags')}")
        print(f"Difficulty: {metadata.get('difficulty')}")

        print(f"Semantic score: {result.get('semantic_score', 0.0):.4f}")
        print(f"Keyword score: {result.get('keyword_score', 0.0):.4f}")
        print(f"Hybrid score: {result.get('hybrid_score', 0.0):.4f}")
        print(f"Final score: {result.get('final_score', 0.0):.4f}")


def main():
    config = get_config(
        embedding_model_key="minilm",
        retrieval_mode="hybrid",  # options: "semantic", "keyword", "hybrid"
        semantic_top_k=10,
        keyword_top_k=10,
        merged_top_k=20,
        final_top_k=5,
        semantic_weight=0.70,
        keyword_weight=0.30,
    )

    query = "prompt for debugging Python code"

    raw_records = load_raw_data()
    processed_records = preprocess_records(raw_records)

    embedder = None
    vector_store = None
    keyword_retriever = None

    if config.retrieval_mode in {"semantic", "hybrid"}:
        embedder = SentenceTransformerEmbedder(config)
        vector_store = ChromaVectorStore(config)

        if vector_store.count() == 0:
            raise RuntimeError(
                f"Chroma collection '{config.collection_name}' is empty. "
                "Run build_index.py first."
            )

    if config.retrieval_mode in {"keyword", "hybrid"}:
        keyword_retriever = TfidfKeywordRetriever(config)
        keyword_retriever.fit(processed_records)

    retriever = SearchRetriever(
        config=config,
        embedder=embedder,
        vector_store=vector_store,
        keyword_retriever=keyword_retriever,
    )

    results = retriever.search(query)

    print_results(query, config, results)


if __name__ == "__main__":
    main()