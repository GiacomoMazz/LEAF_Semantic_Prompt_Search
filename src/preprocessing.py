# preprocessing.py
#
# Purpose:
# Create the text_for_embedding field required by the semantic search pipeline.
#
# Behavior:
# Preserves the original dataset and adds one derived text column combining
# title, category, subcategory, tags, and content.

import pandas as pd

from load_data import df

df_processed = df.copy()

df_processed["text_for_embedding"] = (
    df_processed["title"].astype(str) + ". " +
    df_processed["category"].astype(str) + ". " +
    df_processed["subcategory"].astype(str) + ". " +
    df_processed["tags"].apply(lambda x: " ".join(x)).astype(str) + ". " +
    df_processed["content"].astype(str)
)

print(df_processed[["id", "text_for_embedding"]].head())

print(df_processed["text_for_embedding"].iloc[0])

print(df.shape)

print(df_processed.shape)

df_processed.to_csv("../data/processed_prompts.csv", index=False)

check_df = pd.read_csv("../data/processed_prompts.csv")

print(check_df.shape)
