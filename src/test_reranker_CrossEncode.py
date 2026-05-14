# test_reranking.py

from config import get_config
from data_loader import load_raw_data
from preprocessing import preprocess_records
from embedders import SentenceTransformerEmbedder
from vectorstores import ChromaVectorStore
from keyword_retriever import TfidfKeywordRetriever
from retrievers import SearchRetriever
from rerankers import CrossEncoderReranker


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

    raw_records = load_raw_data()
    processed_records = preprocess_records(raw_records)

    embedder = SentenceTransformerEmbedder(config)
    vector_store = ChromaVectorStore(config)

    if vector_store.count() == 0:
        raise RuntimeError("Chroma collection is empty. Run build_index.py first.")

    keyword_retriever = TfidfKeywordRetriever(config)
    keyword_retriever.fit(processed_records)

    retriever = SearchRetriever(
        config=config,
        embedder=embedder,
        vector_store=vector_store,
        keyword_retriever=keyword_retriever,
    )

    candidates = retriever.search(query)

    reranker = CrossEncoderReranker(config)
    reranked_results = reranker.rerank(query, candidates)

    for rank, result in enumerate(reranked_results[: config.final_top_k], start=1):
        metadata = result["metadata"]

        print("\n" + "-" * 80)
        print(f"Rank: {rank}")
        print(f"ID: {result['id']}")
        print(f"Title: {metadata.get('title')}")
        print(f"Hybrid score: {result.get('hybrid_score', 0.0):.4f}")
        print(f"Reranker score: {result.get('reranker_score', 0.0):.4f}")
        print(f"Final score: {result.get('final_score', 0.0):.4f}")


if __name__ == "__main__":
    main()