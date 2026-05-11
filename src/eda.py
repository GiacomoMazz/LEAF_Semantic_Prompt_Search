'''
eda.py

This file performs the exploratory data analysis of the raw dataset.
It checks dataset quality, duplicates, missing values, and the main
metadata distributions in order to confirm that the data is ready for
the following preprocessing and retrieval steps.

'''

from __future__ import annotations

import pandas as pd

from load_data import load_raw_dataset


# Purpose:
# Analyze the raw dataset and summarize the main statistics needed to
# understand its structure and quality.
#
# Behavior:
# The function computes dataset size, duplicate counts, missing values,
# category and language distributions, prompt length statistics, and
# descriptive summaries of the main metadata fields.
#
# Output:
# A dictionary containing the EDA summary results.

def run_eda(df: pd.DataFrame) -> dict:
    content_length = df["content"].astype(str).str.split().str.len()

    summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "missing_values_total": int(df.isna().sum().sum()),
        "missing_values_by_column": df.isna().sum().to_dict(),
        "duplicate_id_count": int(df["id"].duplicated().sum()),
        "duplicate_title_content_count": int(
            df.duplicated(subset=["title", "content"]).sum()
        ),
        "top_categories": df["category"].value_counts().head(10).to_dict(),
        "top_category_percentages": (
            (df["category"].value_counts(normalize=True) * 100).head(10).round(2)
        ).to_dict(),
        "top_languages": df["language"].value_counts().head(10).to_dict(),
        "difficulty_distribution": df["difficulty"].value_counts().to_dict(),
        "target_models": df["target_model"].value_counts().head(10).to_dict(),
        "content_length_summary": content_length.describe().round(2).to_dict(),
        "engagement_summary": (
            df[
                [
                    "likes",
                    "upvotes",
                    "downvotes",
                    "author_reputation",
                    "views",
                    "uses",
                ]
            ]
            .describe()
            .round(2)
            .to_dict()
        ),
    }

    return summary


if __name__ == "__main__":
    dataset = load_raw_dataset()
    eda_summary = run_eda(dataset)

    print("Dataset shape:")
    print(eda_summary["shape"])

    print("\nMissing values total:")
    print(eda_summary["missing_values_total"])

    print("\nDuplicate counts:")
    print(
        {
            "duplicate_id_count": eda_summary["duplicate_id_count"],
            "duplicate_title_content_count": eda_summary[
                "duplicate_title_content_count"
            ],
        }
    )

    print("\nTop categories:")
    print(eda_summary["top_categories"])

    print("\nTop languages:")
    print(eda_summary["top_languages"])

    print("\nDifficulty distribution:")
    print(eda_summary["difficulty_distribution"])

    print("\nContent length summary:")
    print(eda_summary["content_length_summary"])

    print("\nEngagement summary:")
    print(eda_summary["engagement_summary"])


# Final considerations:
# The EDA intentionally reports issues without modifying the dataset.
# The summary is compact enough for repeated debugging while still capturing
# the checks that matter before preprocessing and retrieval.
