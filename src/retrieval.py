'''
retrieval.py

This file performs semantic retrieval over the ChromaDB vector store.
It converts the user query into an embedding, compares it with the stored
prompt embeddings, and returns the most relevant prompts together with
their metadata and similarity scores.

'''

from time import perf_counter

from embeddings import generate_embeddings
from vector_store import DEFAULT_COLLECTION_NAME, _get_or_create_collection


# Purpose:
# Retrieve the most relevant prompts for a user query by comparing the
# query embedding with the prompt embeddings stored in the vector database.
#
# Behavior:
# The function encodes the query with the same embedding model used for the
# prompts, searches the ChromaDB collection, converts the returned cosine
# distance into a similarity_score, and builds a ranked list of prompt results.
#
# Output:
# A list of dictionaries containing the retrieved prompts, their metadata,
# and the corresponding similarity_score values.

def retrieve(query: str, top_k: int = 50) -> list[dict]:
    if top_k <= 0:
        return []

    collection = _get_or_create_collection(DEFAULT_COLLECTION_NAME)
    collection_count = collection.count()
    if collection_count == 0:
        raise RuntimeError(
            "The ChromaDB collection is empty. Build the vector store before retrieval."
        )

    effective_top_k = min(top_k, collection_count)
    query_embedding = generate_embeddings([query])[0].tolist()
    response = collection.query(
        query_embeddings=[query_embedding],
        n_results=effective_top_k,
        include=["metadatas", "distances", "documents"],
    )

    metadatas = response.get("metadatas", [[]])[0]
    distances = response.get("distances", [[]])[0]
    documents = response.get("documents", [[]])[0]
    ids = response.get("ids", [[]])[0]

    results = []
    for prompt_id, metadata, distance, document in zip(ids, metadatas, distances, documents):
        metadata = metadata or {}
        similarity_score = float(1.0 - distance)

        result = {
            "id": metadata.get("id", prompt_id),
            "title": metadata.get("title", ""),
            "content": metadata.get("content", document or ""),
            "category": metadata.get("category", ""),
            "subcategory": metadata.get("subcategory", ""),
            "tags": metadata.get("tags", ""),
            "likes": metadata.get("likes", 0),
            "upvotes": metadata.get("upvotes", 0),
            "downvotes": metadata.get("downvotes", 0),
            "author_reputation": metadata.get("author_reputation", 0),
            "created_at": metadata.get("created_at", ""),
            "similarity_score": similarity_score,
        }
        results.append(result)

    return results


# Purpose:
# Measure how long semantic retrieval takes over a set of input queries.
#
# Behavior:
# The function runs retrieve(...) for each query in the input list, records
# the execution time of each retrieval, and summarizes the latency values.
#
# Output:
# A dictionary containing the number of tested queries together with the
# average, minimum, and maximum retrieval latency.

def measure_retrieval_latency(queries: list[str], top_k: int = 50) -> dict:
    if not queries:
        raise ValueError("The query list must contain at least one query.")

    durations = []
    for query in queries:
        start = perf_counter()
        retrieve(query, top_k=top_k)
        durations.append(perf_counter() - start)

    return {
        "query_count": len(queries),
        "average_latency_seconds": sum(durations) / len(durations),
        "min_latency_seconds": min(durations),
        "max_latency_seconds": max(durations),
    }


if __name__ == "__main__":
    sample_results = retrieve(
        "Write a speech to have a salary raise",
        top_k=5,
    )

    for result in sample_results:
        print(result)


# Final considerations:
# Similarity here compares query meaning against prompt meaning and is used
# only to rank retrieved prompts.
# This module stops at semantic retrieval and does not include reranking or
# metadata-aware score fusion.
