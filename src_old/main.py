
# refactor - note for all files:
# i removed from __future__ import annotations because it is not necessary
# already built into our python versions for the kind of type hints we do

'''
main.py

Purpose:
Provide a simple command-line comparison between the TF-IDF baseline
and the semantic retrieval pipeline for the same query.

Behavior:
The script ensures the processed dataset and vector store exist, then
runs the same query through baseline_search(...) and retrieve(...).
The two result lists are printed in separate sections for manual review.

Output:
Console comparison of TF-IDF baseline results and semantic retrieval
results for the selected query.
'''

from __future__ import annotations
import sys

from baseline_tfidf import baseline_search
from preprocessing import PROCESSED_DATA_PATH, build_processed_dataset
from retrieval import retrieve
from vector_store import ensure_vector_store

DEFAULT_QUERY = "Write a speech to have a salary raise"


def _print_results(title: str, results: list[dict], top_k: int) -> None:
    print(title)
    print("-" * len(title))

    for index, result in enumerate(results[:top_k], start=1):
        print(
            f"{index}. [{result['similarity_score']:.4f}] "
            f"{result['title']} ({result['category']})"
        )
        print(f"   id: {result['id']}")
        print(
            f"   likes: {result['likes']}, upvotes: {result['upvotes']}, "
            f"downvotes: {result['downvotes']}"
        )
        print(
            f"   author_reputation: {result['author_reputation']}, "
            f"created_at: {result['created_at']}"
        )
        print(f"   content preview: {result['content'][:160]}")

# refactor - added debug printlines throughout the project can probably be removed at the end

def main() -> None:
    query = " ".join(sys.argv[1:]).strip() or DEFAULT_QUERY
    top_k = 5
    
    print("starting main")

    if not PROCESSED_DATA_PATH.exists():
        print("building processed dataset...")
        build_processed_dataset()
        print("processed dataset built.")
    print("ensuring vector store...")
    ensure_vector_store()
    print("vector store ready.")

    print("running tf-idf baseline...")
    baseline_results = baseline_search(query, top_k=top_k)
    print("tf-id done.")

    print("rinnung semantic retrieval...")
    semantic_results = retrieve(query, top_k=top_k)
    print("semantic retrieval done.")

    print(f"Query: {query}\n")
    _print_results("TF-IDF Baseline", baseline_results, top_k=top_k)
    print()
    _print_results("Semantic Retrieval", semantic_results, top_k=top_k)


if __name__ == "__main__":
    main()


'''
Final considerations:
This entrypoint is intentionally simple and is meant for manual validation.
It compares lexical retrieval and semantic retrieval without taking on
later-stage responsibilities such as reranking or evaluation metrics.
'''