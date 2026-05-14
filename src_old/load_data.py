'''
load_data.py

This file loads the raw LEAF PromptKaban dataset from disk and checks
that the required columns are present. It provides the raw DataFrame
that will be used by the following steps of the project, such as EDA
and preprocessing.

'''

from pathlib import Path

import pandas as pd

# refactor - i changed name to project root which is the location of the entire project file
# i also pulled raw data from a subfolder called data instead as we will not store the raw data in src

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "dataset.json"

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


# Purpose:
# Load the raw dataset from disk and verify that all the required fields
# for the project are available.
#
# Behavior:
# The function reads the JSON dataset file, checks whether the expected
# columns are present, and raises an error if any required column is missing.
#
# Output:
# A pandas DataFrame containing the raw prompt dataset.

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
