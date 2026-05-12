'''
preprocessing.py

This file prepares the dataset for retrieval by creating the
text_for_embedding field and saving the processed dataset to disk.
It preserves the original prompt information and adds the semantic
text column that will be used by both the TF-IDF baseline and the
semantic retrieval pipeline.

'''

from pathlib import Path

import pandas as pd

from load_data import RAW_DATA_PATH, load_raw_dataset


BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "processed_prompts.csv"


def _format_tags(value: object) -> str:
    if isinstance(value, list):
        return " ".join(str(tag) for tag in value)
    if pd.isna(value):
        return ""
    return str(value)


# Purpose:
# Create the semantic text field that represents each prompt in the
# retrieval pipeline.
#
# Behavior:
# The function copies the input DataFrame and combines title, category,
# subcategory, tags, and content into a single text_for_embedding column.
#
# Output:
# A processed pandas DataFrame with the additional text_for_embedding field.

def build_text_for_embedding(df: pd.DataFrame) -> pd.DataFrame:
    df_processed = df.copy()

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


# Purpose:
# Save the processed dataset to disk so it can be reused by the baseline,
# embedding, and retrieval steps.
#
# Behavior:
# The function creates the output folder if needed and writes the processed
# DataFrame to a CSV file.
#
# Output:
# No direct return value. The processed dataset is saved as a CSV file.

def save_processed_dataset(
    df: pd.DataFrame, path: str | Path = PROCESSED_DATA_PATH
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


# Purpose:
# Build the full processed dataset starting from the raw data and save it
# to disk in a single step.
#
# Behavior:
# The function loads the raw dataset, creates the text_for_embedding field,
# saves the processed dataset, and returns the processed DataFrame.
#
# Output:
# A processed pandas DataFrame ready to be used in the retrieval pipeline.

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
