# eda.py
#
# Purpose:
# Perform lightweight exploratory data analysis on the LEAF PromptKaban dataset.
#
# Behavior:
# Prints dataset structure, quality checks, and descriptive summaries.

from load_data import df

print(df.head())

print(df.shape)

print(df.columns)

print(df.info())

print(df.isna().sum())

print(df.duplicated(subset=["title", "content"]).sum())
# The dataset contains 93 duplicated title-content pairs.
# Since LEAF described the dataset as ready to use, duplicates are reported but not removed.

print(df["id"].duplicated().sum())

print(df["category"].value_counts().head(10))

print((df["category"].value_counts(normalize=True) * 100).head(10))
# The largest category represents less than 8% of the dataset,
# suggesting that prompts are distributed across several domains.

df["content_length"] = df["content"].astype(str).str.split().str.len()

print(df["content_length"].describe())
# Prompt content has moderate length on average, with a maximum of 272 words.

print(df[["likes", "upvotes", "views", "uses"]].describe())
# Engagement metadata is highly variable, so these fields are better kept as metadata
# rather than included directly in text_for_embedding.

print(df["language"].value_counts().head(10))

print(df["target_model"].value_counts().head(10))

print(df["difficulty"].value_counts())

print(df["difficulty"].value_counts(normalize=True) * 100)
