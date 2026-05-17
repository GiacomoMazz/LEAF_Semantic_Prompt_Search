'''
Making CSV for manual relevance labeling

labels:
0 = not relevant
1 = somewhat relevant
2 = highly relevant
'''

import csv
import json
import time
from pathlib import Path

from config import DATA_DIR, get_config, EMBEDDING_MODELS
from pipelines import SearchPipeline

JUDGMENTS_OUTPUT_PATH = DATA_DIR / "relevance_judgments.csv"
RUNS_OUTPUT_PATH = DATA_DIR / "retrieval_runs.csv"
QUERY_PATH = DATA_DIR / "evaluation_queries.json"

def load_evaluation_queries(path: Path = QUERY_PATH) -> list[dict]:
    with open(path, "r", encoding = "utf-8") as file:
        return json.load(file)
    
def make_experiments() -> dict:
    experiments = {}

    experiments["keyword"] = get_config(
        retrieval_mode = "keyword",
        keyword_top_k = 50,
        final_top_k = 10,
        use_reranker = False,
        use_metadata_scoring = False
    )

    for embedding_model_key in EMBEDDING_MODELS:
        prefix = embedding_model_key

        experiments[f"{prefix}_semantic"] = get_config(
            embedding_model_key = embedding_model_key,
            retrieval_mode = "semantic",
            semantic_top_k = 50,
            final_top_k = 10,
            use_reranker = False,
            use_metadata_scoring = False
        )

        experiments[f"{prefix}_semantic_reranker_msmarco"] = get_config(
            embedding_model_key = embedding_model_key,
            retrieval_mode = "semantic",
            semantic_top_k = 50,
            final_top_k = 10,
            use_reranker = True,
            reranker_model_key = "msmarco",
            use_metadata_scoring = False
        )

        experiments[f"{prefix}_semantic_reranker_tinybert"] = get_config(
            embedding_model_key = embedding_model_key,
            retrieval_mode = "semantic",
            semantic_top_k = 50,
            final_top_k = 10,
            use_reranker = True,
            reranker_model_key = "tinybert",
            use_metadata_scoring = False
        )

        experiments[f"{prefix}_semantic_reranker_metadata_msmarco"] = get_config(
            embedding_model_key = embedding_model_key,
            retrieval_mode = "semantic",
            semantic_top_k = 50,
            final_top_k = 10,
            use_reranker = True,
            reranker_model_key = "msmarco",
            use_metadata_scoring = True,
            search_score_weight = 0.85,
            metadata_score_weight = 0.15
        )

        experiments[f"{prefix}_semantic_reranker_metadata_tinybert"] = get_config(
            embedding_model_key = embedding_model_key,
            retrieval_mode = "semantic",
            semantic_top_k = 50,
            final_top_k = 10,
            use_reranker = True,
            reranker_model_key = "tinybert",
            use_metadata_scoring = True,
            search_score_weight = 0.85,
            metadata_score_weight = 0.15
        )


        experiments[f"{prefix}_hybrid"] = get_config(
            embedding_model_key = embedding_model_key,
            retrieval_mode = "hybrid",
            semantic_top_k = 50,
            keyword_top_k = 50,
            merged_top_k = 75,
            final_top_k = 10,
            semantic_weight = 0.7,
            keyword_weight = 0.3,
            use_reranker = False,
            use_metadata_scoring = False
        )

        experiments[f"{prefix}_hybrid_reranker_msmarco"] = get_config(
            embedding_model_key = embedding_model_key,
            retrieval_mode = "hybrid",
            semantic_top_k = 50,
            keyword_top_k = 50,
            merged_top_k = 75,
            final_top_k = 10,
            semantic_weight = 0.7,
            keyword_weight = 0.3,
            use_reranker = True,
            reranker_model_key = "msmarco",
            use_metadata_scoring = False
        )

        experiments[f"{prefix}_hybrid_reranker_tinybert"] = get_config(
            embedding_model_key = embedding_model_key,
            retrieval_mode = "hybrid",
            semantic_top_k = 50,
            keyword_top_k = 50,
            merged_top_k = 75,
            final_top_k = 10,
            semantic_weight = 0.7,
            keyword_weight = 0.3,
            use_reranker = True,
            reranker_model_key = "tinybert",
            use_metadata_scoring = False
        )

        experiments[f"{prefix}_hybrid_reranker_metadata_msmarco"] = get_config(
            embedding_model_key = embedding_model_key,
            retrieval_mode = "hybrid",
            semantic_top_k = 50,
            keyword_top_k = 50,
            merged_top_k = 75,
            final_top_k = 10,
            semantic_weight = 0.7,
            keyword_weight = 0.3,
            use_reranker = True,
            reranker_model_key = "msmarco",
            use_metadata_scoring = True,
            search_score_weight = 0.85,
            metadata_score_weight = 0.15
        )

        experiments[f"{prefix}_hybrid_reranker_metadata_tinybert"] = get_config(
            embedding_model_key = embedding_model_key,
            retrieval_mode = "hybrid",
            semantic_top_k = 50,
            keyword_top_k = 50,
            merged_top_k = 75,
            final_top_k = 10,
            semantic_weight = 0.7,
            keyword_weight = 0.3,
            use_reranker = True,
            reranker_model_key = "tinybert",
            use_metadata_scoring = True,
            search_score_weight = 0.85,
            metadata_score_weight = 0.15
        )
    

    
    return experiments

def text_preview(text: str, max_chars: int = 300) -> str:
    text = " ".join(str(text).split())

    if len(text) <= max_chars:
        return text
    
    return text[:max_chars] + "..."

def main():
    queries = load_evaluation_queries()
    experiments = make_experiments()

    pipelines = {}
    for name, config in experiments.items():
        pipelines[name] = SearchPipeline(config)

    judgment_rows_by_key = {}
    run_rows = []

    for query_dict in queries:
        query = query_dict["query"]

        print(f"\nProcessing query: {query}")

        for experiment_name, pipeline in pipelines.items():
            print(f"Running: {experiment_name}")

            start_time = time.perf_counter()
            results = pipeline.search(query)
            latency_seconds = time.perf_counter() - start_time

            for rank, result in enumerate(results, start = 1):
                result_id = result["id"]
                judgment_key = (query, result_id)

                metadata = result.get("metadata", {})

                if judgment_key not in judgment_rows_by_key:

                    judgment_rows_by_key[judgment_key]={
                        "query": query,
                        "result_id": result_id,
                        "title": metadata.get("title", ""),
                        "category": metadata.get("category", ""),
                        "subcategory": metadata.get("subcategory", ""),
                        "tags": metadata.get("tags", ""),
                        "difficulty": metadata.get("difficulty", ""),
                        "content_preview": text_preview(metadata.get("content", "")),
                        "relevance": ""
                    }

                run_rows.append(
                    {
                        "query": query,
                        "experiment": experiment_name,
                        "result_id": result_id,
                        "rank": rank,
                        "final_score": result.get("final_score", ""),
                        "semantic_score": result.get("semantic_score", ""),
                        "keyword_score": result.get("keyword_score", ""),
                        "hybrid_score": result.get("hybrid_score", ""),
                        "reranker_score": result.get("reranker_score", ""),
                        "metadata_score": result.get("metadata_score", ""),
                        "latency_seconds" : latency_seconds
                    }
                )
    judgment_rows = list(judgment_rows_by_key.values())

    with open(JUDGMENTS_OUTPUT_PATH, "w", encoding = "utf-8", newline = "") as file:
        fieldnames = [
            "query",
            "result_id",
            "title",
            "category",
            "subcategory",
            "tags",
            "difficulty",
            "content_preview",
            "relevance"
        ]

        writer = csv.DictWriter(file, fieldnames = fieldnames)
        writer.writeheader()
        writer.writerows(judgment_rows)

    with open(RUNS_OUTPUT_PATH, "w", encoding = "utf-8", newline = "") as file:
        fieldnames = [
            "query",
            "experiment",
            "result_id",
            "rank",
            "final_score",
            "semantic_score",
            "keyword_score",
            "hybrid_score",
            "reranker_score",
            "metadata_score",
            "latency_seconds"
        ]

        writer = csv.DictWriter(file, fieldnames = fieldnames)
        writer.writeheader()
        writer.writerows(run_rows)

    print(f"\nSaved relevance judgment file to : {JUDGMENTS_OUTPUT_PATH}")
    print(f"Rows to label: {len(judgment_rows)}")

    print(f"\nSaved retrieval file to : {RUNS_OUTPUT_PATH}")
    print(f"Rows to label: {len(run_rows)}")
    
if __name__ == "__main__":
    main()
