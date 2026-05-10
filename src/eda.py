# eda.py
#
# Purpose:
# Perform lightweight exploratory data analysis on the LEAF PromptKaban
# dataset and confirm that it is ready for the semantic search pipeline.
#
# Behavior:
# The module exposes run_eda(df), which computes dataset shape, field checks,
# duplicate counts, and compact summaries of the main metadata distributions.
# When the file is executed directly, it prints the EDA summary.
#
# Output:
# Dictionary containing the computed EDA statistics. When run directly,
# the summary is also printed to the console.

from __future__ import annotations

import pandas as pd

from load_data import load_raw_dataset


def run_eda(df: pd.DataFrame) -> dict:
    content_length = df["content"].astype(str).str.split().str.len()

    summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "missing_values_total": int(df.isna().sum().sum()),
        "missing_values_by_column": df.isna().sum().to_dict(),
        "duplicate_id_count": int(df["id"].duplicated().sum()),
        "duplicate_title_content_count": int(df.duplicated(subset=["title", "content"]).sum()),
        "top_categories": df["category"].value_counts().head(10).to_dict(),
        "top_category_percentages": ((df["category"].value_counts(normalize=True) * 100).head(10).round(2)).to_dict(),
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
# The dataset is ready to use: no missing values were found and prompt ids are unique.
# A small number of duplicated title-content pairs was found, but they are reported and not removed.
# The dataset covers many categories, with no single category dominating the collection.
# Engagement fields such as likes, upvotes, views, and uses are kept as metadata.
# The text_for_embedding field should therefore focus on semantic fields:
# title, category, subcategory, tags, and content.
