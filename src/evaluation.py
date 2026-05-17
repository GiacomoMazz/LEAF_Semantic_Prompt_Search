import csv
import math
from collections import defaultdict

from config import DATA_DIR

"""
Evaluations run:

Precision@K: how many top-k results are relevant
MRR: hw high the first relevant result appears
nDCG@K: ranking quality with 0,1,2 graded relevance
"""

JUDGMENTS_PATH = DATA_DIR / "completed_relevance_judgments.csv"
RUNS_PATH = DATA_DIR / "retrieval_runs.csv"
OUTPUT_PATH = DATA_DIR / "evaluation_results.csv"

def load_judgments(path = JUDGMENTS_PATH) -> dict:

    judgments = {}

    with open(path, "r", encoding = "utf-8") as file:

        reader = csv.DictReader(file)

        for row in reader:
            query = row["query"]
            result_id = row["result_id"]

            relevance_raw = row.get("relevance", "").strip()

            if relevance_raw == "":
                continue

            relevance = int(relevance_raw)
            judgments[(query, result_id)] = relevance

    return judgments

def load_runs(path = RUNS_PATH) -> list[dict]:

    # list of dictionaries containing retrieval_runs data
    rows = []

    with open(path, "r", encoding = "utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            row["rank"] = int(row["rank"])

            if "latency_seconds" in row and row["latency_seconds"] != "":
                row["latency_seconds"] = float(row["latency_seconds"])
            else:
                row["latency_seconds"] = 0.0

            rows.append(row)
    
    return rows

def group_runs_by_experiment_and_query(run_rows: list[dict]) -> dict:

    # Grouped is a dictionary where each key is a query experiment combo.
    # Each value is a list of result dictionaries sorted by rank.
    grouped = defaultdict(list)

    for row in run_rows:
        key = (row["experiment"], row["query"])
        grouped[key].append(row)

    for key in grouped:
        grouped[key].sort(key = lambda row: row["rank"])

    return grouped

def precision_at_k(rows: list[dict], judgments: dict, k: int) -> float:
    top_k = rows[: k]

    if not top_k:
        return 0.0
    
    relevant_count = 0

    for row in top_k:
        query = row["query"]
        result_id = row["result_id"]
        relevance = judgments.get((query, result_id), 0)
        
        # 2 is considered relevant here
        if relevance > 1:
            relevant_count += 1

    return relevant_count / len(top_k)

def mrr(rows: list[dict], judgments: dict) -> float:
    
    for row in rows:
        query = row["query"]
        result_id = row["result_id"]
        relevance = judgments.get((query, result_id), 0)

        # 2 is considered relevant to get better results
        if relevance > 1:
            return 1.0 / row["rank"]
        
    return 0.0

def dcg_at_k(relevances: list[int], k: int) -> float:
    score = 0.0
    # this metric weighs higher relevance scores more
    for index, relevance in enumerate(relevances[: k], start = 1):
        gain = (2 ** relevance) - 1
        discount = math.log2(index + 1)
        score += gain / discount

    return score

def ideal_relevances_for_query(query : str, judgments : dict, k : int) -> list[int]:
    relevances = []

    for (judged_query, result_id), relevance in judgments.items():
        if judged_query == query:
            relevances.append(relevance)

    return sorted(relevances, reverse = True)[: k]

def ndcg_at_k(rows: list[dict], judgments: dict, k: int) -> float:
    relevances = []

    for row in rows[: k]:
        query = row["query"]
        result_id = row["result_id"]
        relevance = judgments.get((query, result_id), 0)
        relevances.append(relevance)

    actual_dcg = dcg_at_k(relevances, k)

    ideal_relevances = ideal_relevances_for_query(query, judgments, k)
    ideal_dcg = dcg_at_k(ideal_relevances, k)

    if ideal_dcg == 0:
        return 0.0
    
    return actual_dcg / ideal_dcg

# evaluating at both k = 5 and 10
def evaluate(k_values = (5, 10)) -> list[dict]:
    judgments = load_judgments()
    run_rows = load_runs()
    grouped = group_runs_by_experiment_and_query(run_rows)

    experiment_metrics = defaultdict(list)

    for (experiment, query), rows in grouped.items():
        query_metrics = {
            "experiment" : experiment,
            "query" : query,
            "mrr" : mrr(rows, judgments),
            "avg_latency_seconds" : rows[0].get("latency_seconds", 0.0)
        }
        for k in k_values:
            query_metrics[f"precision_at_{k}"] = precision_at_k(rows, judgments, k)
            query_metrics[f"ndcg_at_{k}"] = ndcg_at_k(rows, judgments, k)
        
        experiment_metrics[experiment].append(query_metrics)

    summary_rows = []

    for experiment, metrics_list in experiment_metrics.items():
        summary = {
            "experiment" : experiment,
            "num_queries" : len(metrics_list)
        }

        metric_names = []
        for key in metrics_list[0].keys():
            if key not in {"experiment", "query"}:
                metric_names.append(key)
        
        for metric_name in metric_names:
            
            values = []
            for row in metrics_list:
                values.append(row[metric_name])
            
            summary[metric_name] = round(sum(values) / len(values), 4)
        
        summary_rows.append(summary)

    summary_rows.sort(key = lambda row : row.get("ndcg_at_10", 0.0), reverse = True)

    return summary_rows

def save_results(rows : list[dict], path = OUTPUT_PATH) -> None:
    if not rows:
        return
    
    fieldnames = list(rows[0].keys())

    with open(path, "w", encoding = "utf-8", newline = "") as file:
        writer = csv.DictWriter(file, fieldnames = fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def main():
    results = evaluate(k_values = (5, 10))
    save_results(results)

    print(f"Saved evaluation results to : {OUTPUT_PATH}")

if __name__ == "__main__":
    main()