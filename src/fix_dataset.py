
import csv
import pandas as pd

input_file = "dataset/WELFake_Dataset.csv"
output_file = "dataset/WELFake_Cleaned.csv"

rows = []
invalid_labels = 0

with open(input_file, "r", encoding="latin-1", newline="") as file:
    reader = csv.reader(file)
    next(reader)

    for row in reader:
        if len(row) < 4:
            continue

        last_index = len(row) - 1

        while last_index >= 3 and not row[last_index].strip():
            last_index -= 1

        label = row[last_index].strip()

        if label not in ["0", "1"]:
            invalid_labels += 1
            continue

        news_id = row[0]
        title = row[1]
        article = ",".join(row[2:last_index])

        rows.append([news_id, title, article, int(label)])

df = pd.DataFrame(
    rows,
    columns=["id", "title", "text", "label"]
)

df.to_csv(output_file, index=False, encoding="utf-8")

print("Dataset reconstruction completed!")
print("Shape:", df.shape)
print("Invalid rows skipped:", invalid_labels)
print("Average article length:", df["text"].str.len().mean())
