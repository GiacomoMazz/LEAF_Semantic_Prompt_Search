"""
Plot evaluation results.

This script reads the aggregated evaluation results and generates figures that
compare ranking quality, pipeline families, embedding models, rerankers, latency,
and quality-latency tradeoffs.
"""

import pandas as pd
import matplotlib.pyplot as plt

from config import PROJECT_ROOT, DATA_DIR


RESULTS_PATH = DATA_DIR / "evaluation_results.csv"
FIGURES_DIR = PROJECT_ROOT / "figures"


def simplify_name(name: str) -> str:
    replacements = {
        "bge_small": "BGE",
        "minilm": "MiniLM",
        "semantic": "Sem",
        "hybrid": "Hybrid",
        "reranker": "Rerank",
        "metadata": "Meta",
        "msmarco": "MSMARCO",
        "tinybert": "TinyBERT",
        "keyword": "Keyword",
    }

    output = name

    for old, new in replacements.items():
        output = output.replace(old, new)

    return output


def add_pipeline_family(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def family(name: str) -> str:
        if name == "keyword":
            return "Keyword"

        if "hybrid" in name and "reranker" in name and "metadata" in name:
            return "Hybrid + Reranker + Metadata"

        if "hybrid" in name and "reranker" in name:
            return "Hybrid + Reranker"

        if "hybrid" in name and "metadata" in name:
            return "Hybrid + Metadata"

        if "hybrid" in name:
            return "Hybrid"

        if "semantic" in name and "reranker" in name and "metadata" in name:
            return "Semantic + Reranker + Metadata"

        if "semantic" in name and "reranker" in name:
            return "Semantic + Reranker"

        if "semantic" in name:
            return "Semantic"

        return "Other"

    df["pipeline_family"] = df["experiment"].apply(family)
    return df

def plot_latency_by_experiment(df: pd.DataFrame, top_n: int = 12) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_df = df.sort_values("avg_latency_seconds", ascending=True).head(top_n).copy()
    plot_df["experiment_short"] = plot_df["experiment"].apply(simplify_name)

    plt.figure(figsize=(16, 9))
    plt.bar(plot_df["experiment_short"], plot_df["avg_latency_seconds"])

    plt.xlabel("Experiment")
    plt.ylabel("Average latency per query (seconds)")
    plt.title("Fastest Search Configurations by Average Query Latency")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    output_path = FIGURES_DIR / "latency_by_experiment.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def plot_ndcg_vs_latency(df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_df = df.copy()
    plot_df["experiment_short"] = plot_df["experiment"].apply(simplify_name)

    plt.figure(figsize=(12, 8))
    plt.scatter(plot_df["avg_latency_seconds"], plot_df["ndcg_at_10"])

    for _, row in plot_df.iterrows():
        plt.annotate(
            row["experiment_short"],
            (row["avg_latency_seconds"], row["ndcg_at_10"]),
            fontsize=8,
            xytext=(5, 5),
            textcoords="offset points",
        )

    plt.xlabel("Average latency per query (seconds)")
    plt.ylabel("nDCG@10")
    plt.title("Quality-Latency Tradeoff")

    plt.tight_layout()

    output_path = FIGURES_DIR / "ndcg_vs_latency.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def plot_top_ndcg(df: pd.DataFrame, top_n: int = 12) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_df = df.sort_values("ndcg_at_10", ascending=False).head(top_n).copy()
    plot_df["experiment_short"] = plot_df["experiment"].apply(simplify_name)

    plt.figure(figsize=(16, 9))
    plt.bar(plot_df["experiment_short"], plot_df["ndcg_at_10"])

    plt.xlabel("Experiment")
    plt.ylabel("nDCG@10")
    plt.title("Top Search Configurations by nDCG@10")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)

    plt.tight_layout()

    output_path = FIGURES_DIR / "top_experiments_ndcg_at_10.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def plot_pipeline_family(df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = add_pipeline_family(df)

    family_scores = (
        df.groupby("pipeline_family")["ndcg_at_10"]
        .max()
        .sort_values(ascending=False)
        .reset_index()
    )

    plt.figure(figsize=(14, 8))
    plt.bar(family_scores["pipeline_family"], family_scores["ndcg_at_10"])

    plt.xlabel("Pipeline Type")
    plt.ylabel("Best nDCG@10")
    plt.title("Best nDCG@10 by Pipeline Type")
    plt.xticks(rotation=35, ha="right")
    plt.ylim(0, 1)

    plt.tight_layout()

    output_path = FIGURES_DIR / "pipeline_family_ndcg_at_10.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def plot_embedding_comparison(df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    selected = df[
        df["experiment"].isin(
            [
                "minilm_hybrid_reranker_metadata_msmarco",
                "bge_small_hybrid_reranker_metadata_msmarco",
            ]
        )
    ].copy()

    selected["model"] = selected["experiment"].map(
        {
            "minilm_hybrid_reranker_metadata_msmarco": "MiniLM",
            "bge_small_hybrid_reranker_metadata_msmarco": "BGE-small",
        }
    )

    plt.figure(figsize=(10, 6))
    plt.bar(selected["model"], selected["ndcg_at_10"])

    plt.xlabel("Embedding Model")
    plt.ylabel("nDCG@10")
    plt.title("Embedding Model Comparison: Full Pipeline")
    plt.ylim(0, 1)

    plt.tight_layout()

    output_path = FIGURES_DIR / "embedding_model_comparison_ndcg_at_10.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def plot_reranker_comparison(df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    selected = df[
        df["experiment"].isin(
            [
                "minilm_hybrid_reranker_metadata_msmarco",
                "minilm_hybrid_reranker_metadata_tinybert",
                "bge_small_hybrid_reranker_metadata_msmarco",
                "bge_small_hybrid_reranker_metadata_tinybert",
            ]
        )
    ].copy()

    selected["label"] = selected["experiment"].map(
        {
            "minilm_hybrid_reranker_metadata_msmarco": "MiniLM + MSMARCO",
            "minilm_hybrid_reranker_metadata_tinybert": "MiniLM + TinyBERT",
            "bge_small_hybrid_reranker_metadata_msmarco": "BGE + MSMARCO",
            "bge_small_hybrid_reranker_metadata_tinybert": "BGE + TinyBERT",
        }
    )

    selected = selected.sort_values("ndcg_at_10", ascending=False)

    plt.figure(figsize=(12, 7))
    plt.bar(selected["label"], selected["ndcg_at_10"])

    plt.xlabel("Full Pipeline Variant")
    plt.ylabel("nDCG@10")
    plt.title("Reranker Comparison in Full Pipeline")
    plt.xticks(rotation=25, ha="right")
    plt.ylim(0, 1)

    plt.tight_layout()

    output_path = FIGURES_DIR / "reranker_comparison_ndcg_at_10.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def main():
    df = pd.read_csv(RESULTS_PATH)

    print("Available columns:")
    print(df.columns.tolist())

    plot_top_ndcg(df)
    plot_pipeline_family(df)
    plot_embedding_comparison(df)
    plot_reranker_comparison(df)

    if "avg_latency_seconds" in df.columns:
        plot_latency_by_experiment(df)
        plot_ndcg_vs_latency(df)


if __name__ == "__main__":
    main()