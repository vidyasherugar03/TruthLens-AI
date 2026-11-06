print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nMissing values:")
print(df[["title", "text", "content", "label"]].isnull().sum())

print("\nEmpty processed articles:")
print((df["content"].str.strip() == "").sum())

print("\nFirst 3 headlines:")
print(df["title"].head(3).to_string(index=False))