# check if adding target model and difficulty help with accuracy or hurt

# test_keyword_search.py

from config import get_config
from data_loader import load_raw_data
from preprocessing import preprocess_records
from keyword_retriever import TfidfKeywordRetriever


def main():
    config = get_config(
        retrieval_mode="keyword",
        keyword_top_k=5,
        final_top_k=5,
    )

    raw_records = load_raw_data()
    processed_records = preprocess_records(raw_records)

    retriever = TfidfKeywordRetriever(config)
    retriever.fit(processed_records)

    query = "I want to learn more about the virus"

    results = retriever.search(
        query=query,
        top_k=config.keyword_top_k,
    )

    print(f"Query: {query}")
    print(f"Results found: {len(results)}")

    for rank, result in enumerate(results[:config.final_top_k], start=1):
        metadata = result["metadata"]

        print("\n" + "-" * 80)
        print(f"Rank: {rank}")
        print(f"ID: {result['id']}")
        print(f"Keyword score: {result['keyword_score']}")
        print(f"Title: {metadata.get('title')}")
        print(f"Category: {metadata.get('category')}")
        print(f"Tags: {metadata.get('tags')}")


if __name__ == "__main__":
    main()