
import pandas as pd
import re

# Load cleaned dataset
df = pd.read_csv("dataset/WELFake_Cleaned.csv")

print("Original dataset shape:", df.shape)

# Remove duplicate rows
df = df.drop_duplicates(subset=["title", "text"])

# Remove missing values
df = df.dropna(subset=["title", "text"])

# Remove empty title and text
df = df[
    (df["title"].str.strip() != "") &
    (df["text"].str.strip() != "")
].copy()

# Combine title and article
df["content"] = df["title"] + " " + df["text"]

# Convert to lowercase
df["content"] = df["content"].str.lower()

# Remove URLs
df["content"] = df["content"].apply(
    lambda x: re.sub(r"http\S+|www\S+", "", x)
)

# Remove special characters and numbers
df["content"] = df["content"].apply(
    lambda x: re.sub(r"[^a-z\s]", " ", x)
)

# Remove extra spaces
df["content"] = df["content"].apply(
    lambda x: re.sub(r"\s+", " ", x).strip()
)

# Remove empty processed articles
df = df[df["content"].str.strip() != ""].copy()

# Save processed dataset
df.to_csv("dataset/WELFake_Preprocessed.csv", index=False)

print("\nPreprocessing completed!")
print("Final dataset shape:", df.shape)

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nMissing values:")
print(df[["title", "text", "content", "label"]].isnull().sum())

print("\nEmpty processed articles:")
print((df["content"].str.strip() == "").sum())

print("\nFirst 3 headlines:")
print(df["title"].head(3).to_string(index=False))
