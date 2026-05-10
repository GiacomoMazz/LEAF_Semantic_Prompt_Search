# preprocessing.py
#
# Purpose:
# Create the text_for_embedding field required by both the TF-IDF baseline
# and the semantic retrieval pipeline, then save the processed dataset.
#
# Behavior:
# The module exposes build_text_for_embedding(df) and
# save_processed_dataset(df, path). It preserves the original dataset fields
# and adds a single semantic text column built from title, category,
# subcategory, tags, and content.
#
# Output:
# Processed pandas DataFrame and processed_prompts.csv saved to disk.

from __future__ import annotations

from pathlib import Path

import pandas as pd

from load_data import RAW_DATA_PATH, load_raw_dataset


BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DATA_PATH = BASE_DIR.parent / "data" / "processed" / "processed_prompts.csv"


def _format_tags(value: object) -> str:
    if isinstance(value, list):
        return " ".join(str(tag) for tag in value)
    if pd.isna(value):
        return ""
    return str(value)


def build_text_for_embedding(df: pd.DataFrame) -> pd.DataFrame:
    df_processed = df.copy()

    # The combined field focuses on semantic prompt meaning rather than
    # engagement metadata, which will remain available separately.
    df_processed["text_for_embedding"] = (
        df_processed["title"].astype(str)
        + ". "
        + df_processed["category"].astype(str)
        + ". "
        + df_processed["subcategory"].astype(str)
        + ". "
        + df_processed["tags"].apply(_format_tags)
        + ". "
        + df_processed["content"].astype(str)
    )

    return df_processed


def save_processed_dataset(
    df: pd.DataFrame, path: str | Path = PROCESSED_DATA_PATH
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def build_processed_dataset(
    raw_path: str | Path = RAW_DATA_PATH,
    output_path: str | Path = PROCESSED_DATA_PATH,
) -> pd.DataFrame:
    df = load_raw_dataset(raw_path)
    df_processed = build_text_for_embedding(df)
    save_processed_dataset(df_processed, output_path)
    return df_processed


if __name__ == "__main__":
    raw_df = load_raw_dataset()
    processed_df = build_text_for_embedding(raw_df)

    print(processed_df[["id", "text_for_embedding"]].head())
    print(processed_df["text_for_embedding"].iloc[0])
    print(raw_df.shape)
    print(processed_df.shape)

    save_processed_dataset(processed_df)
    check_df = pd.read_csv(PROCESSED_DATA_PATH)
    print(check_df.shape)


# Final considerations:
# The preprocessing step preserves all original columns and adds only the
# text_for_embedding field required for retrieval.
# Metadata such as likes, upvotes, downvotes, author_reputation, and
# created_at are intentionally retained for later analysis and ranking work.
