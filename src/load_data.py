# load_data.py
#
# Purpose:
# Load the raw LEAF PromptKaban dataset from disk and validate the fields
# expected by the semantic search pipeline.
#
# Behavior:
# The module exposes a single public function, load_raw_dataset(path),
# that reads the JSON dataset and returns it as a pandas DataFrame.
# The function also checks that the required columns are present.
#
# Output:
# pandas DataFrame containing the raw dataset rows and columns.


from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = BASE_DIR.parent / "data" / "dataset.json"

REQUIRED_COLUMNS = {
    "id",
    "author_reputation",
    "version",
    "fork_count",
    "likes",
    "upvotes",
    "downvotes",
    "views",
    "uses",
    "created_at",
    "title",
    "content",
    "category",
    "subcategory",
    "tags",
    "has_placeholders",
    "placeholders",
    "difficulty",
    "language",
    "target_model",
}


def load_raw_dataset(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    dataset_path = Path(path)
    df = pd.read_json(dataset_path)

    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        missing_columns_str = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"The raw dataset is missing required columns: {missing_columns_str}."
        )

    return df


if __name__ == "__main__":
    dataset = load_raw_dataset()
    print(dataset.head())
    print(dataset.shape)


# Final considerations:
# The loader has no side effects beyond reading the dataset.
# Validation is intentionally lightweight so the same function can be reused
# by EDA, preprocessing, and any later evaluation code.
