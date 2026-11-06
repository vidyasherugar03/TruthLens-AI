
import pandas as pd
import joblib
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Load preprocessed dataset
df = pd.read_csv("dataset/WELFake_Preprocessed.csv")

print("Dataset loaded:", df.shape)

# Separate text and labels
X = df["content"]
y = df["label"]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Create TF-IDF vectorizer
tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

# Learn vocabulary from training data only
X_train_tfidf = tfidf.fit_transform(X_train)

# Convert testing data
X_test_tfidf = tfidf.transform(X_test)

print("\nTF-IDF conversion completed!")
print("Training matrix shape:", X_train_tfidf.shape)
print("Testing matrix shape:", X_test_tfidf.shape)

print("\nSample important words:")
print(list(tfidf.get_feature_names_out()[:20]))

# Save generated files
os.makedirs("models", exist_ok=True)

joblib.dump(tfidf, "models/tfidf_vectorizer.joblib")

joblib.dump(
    {
        "X_train": X_train_tfidf,
        "X_test": X_test_tfidf,
        "y_train": y_train,
        "y_test": y_test
    },
    "models/tfidf_data.joblib",
    compress=3
)

print("\nTF-IDF files saved successfully!")
