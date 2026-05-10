# preprocessing.py
#
# Purpose:
# Create the text_for_embedding field required by the semantic search pipeline.
# The script preserves the original dataset and exports the processed CSV.

import pandas as pd

from load_data import df

df_processed = df.copy()

# title, category, subcategory, tags, and content are combined because they describe
# the semantic meaning of each prompt
df_processed["text_for_embedding"] = (
    df_processed["title"].astype(str) + ". " +
    df_processed["category"].astype(str) + ". " +
    df_processed["subcategory"].astype(str) + ". " +
    df_processed["tags"].apply(lambda x: " ".join(x)).astype(str) + ". " +
    df_processed["content"].astype(str)
)

print(df_processed[["id", "text_for_embedding"]].head())

print(df_processed["text_for_embedding"].iloc[0])
# Output: complete example of the generated text_for_embedding text

print(df.shape)

print(df_processed.shape)

df_processed.to_csv("../data/processed_prompts.csv", index=False)

check_df = pd.read_csv("../data/processed_prompts.csv")

print(check_df.shape)
# Output: confirms that the saved CSV keeps the expected shape, (20000, 21)

# Final considerations:
# The original dataset structure is preserved.
# The only added column is text_for_embedding, which is required by the semantic search pipeline.
# This field combines only semantic information: title, category, subcategory, tags, and content.
# Metadata fields such as likes, upvotes, views, uses, difficulty, language, and target_model are kept unchanged.
# The final processed dataset is saved as ../data/processed_prompts.csv.
