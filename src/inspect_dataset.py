
import pandas as pd

dataset_path = "dataset/WELFake_Dataset.csv"

df = pd.read_csv(
    dataset_path,
    encoding="latin-1",
    engine="c",
    on_bad_lines="warn"
)

print("\n===== DATASET SUMMARY =====")

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())
