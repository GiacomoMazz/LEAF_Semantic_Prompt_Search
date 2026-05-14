from config import get_config
from embedders import SentenceTransformerEmbedder
from vectorstores import ChromaVectorStore


def main():
    config = get_config(
        embedding_model_key="minilm",
        retrieval_mode="semantic",
        semantic_top_k=5,
        final_top_k=5,
    )

    query = "I want to learn more about the virus"

    embedder = SentenceTransformerEmbedder(config)
    store = ChromaVectorStore(config)

    query_embedding = embedder.embed_query(query)

    results = store.search(
        query_embedding=query_embedding,
        top_k=config.semantic_top_k,
    )

    print(f"Query: {query}")
    print(f"Results found: {len(results)}")

    for rank, result in enumerate(results[:config.final_top_k], start=1):
        metadata = result["metadata"]

        print("\n" + "-" * 80)
        print(f"Rank: {rank}")
        print(f"ID: {result['id']}")
        print(f"Distance: {result['distance']}")
        print(f"Title: {metadata.get('title')}")
        print(f"Category: {metadata.get('category')}")
        print(f"Subcategory: {metadata.get('subcategory')}")
        print(f"Tags: {metadata.get('tags')}")
        print(f"Difficulty: {metadata.get('difficulty')}")


if __name__ == "__main__":
    main()