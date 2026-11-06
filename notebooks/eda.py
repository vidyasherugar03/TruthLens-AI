
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("dataset/WELFake_Preprocessed.csv")

print("Dataset Shape:", df.shape)

print("\nLabel Distribution:")
print(df["label"].value_counts())

# Calculate article length
df["article_length"] = df["content"].str.split().str.len()

print("\nArticle Length Statistics:")
print(df["article_length"].describe())

# Fake vs Real news graph
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="label")
plt.title("Fake vs Real News Distribution")
plt.xlabel("Label (0 = Fake, 1 = Real)")
plt.ylabel("Number of Articles")
plt.tight_layout()
plt.savefig("screenshots/label_distribution.png")
plt.show()

# Article length graph
plt.figure(figsize=(8, 4))
sns.histplot(data=df, x="article_length", bins=50)
plt.title("Article Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("screenshots/article_length.png")
plt.show()

print("\nEDA completed successfully!")
