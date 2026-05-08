# eda.py

#

# Purpose:
# Perform a lightweight exploratory data analysis on the LEAF PromptKaban dataset.
# The goal is to inspect dataset structure, quality checks, and metadata distributions.

from load_data import df

print(df.head())

print(df.shape)
# Output: the dataset contains 20,000 rows and 20 original columns.

print(df.columns)

print(df.info())

print(df.isna().sum())
# Output: no missing values are present.

print(df.duplicated(subset=["title", "content"]).sum())
# Output: 93 duplicated title-content pairs were found.
# Since the dataset is provided as ready to use, duplicates are reported but not removed.

print(df["id"].duplicated().sum())

print(df["category"].value_counts().head(10))

print((df["category"].value_counts(normalize=True) * 100).head(10))
# Output: the largest category is below 8%, so the dataset is not dominated by one single category.

df["content_length"] = df["content"].astype(str).str.split().str.len()

print(df["content_length"].describe())
# Output: prompt content has an average length of about 55 words and a maximum of 272 words.

print(df[["likes", "upvotes", "views", "uses"]].describe())
# Output: engagement metadata has high variance.
# These fields are kept as metadata and are not included in text_for_embedding.

print(df["language"].value_counts().head(10))
#Output: the dataset is almost entirely in English

print(df["target_model"].value_counts().head(10))

print(df["difficulty"].value_counts())

print(df["difficulty"].value_counts(normalize=True) * 100)
# Output: intermediate and beginner prompts are the most frequent difficulty levels.

# Final considerations:
# The dataset is ready to use: no missing values were found and prompt ids are unique.
# A small number of duplicated title-content pairs was found, but they are reported and not removed.
# The dataset covers many categories, with no single category dominating the collection.
# Engagement fields such as likes, upvotes, views, and uses are kept as metadata.
# The text_for_embedding field should therefore focus on semantic fields:
# title, category, subcategory, tags, and content.
