from datasets import load_dataset
import csv
import random
from collections import Counter


# ---------------------------------------
# 1. Load harmful Burmese dataset
# ---------------------------------------

print("Loading harmful Burmese dataset...")

harmful_dataset = load_dataset(
    "simbolo-ai/burmese-hatespeech"
)

harmful_comments = []

for row in harmful_dataset["train"]:
    text = row["text"].strip()

    if text:
        harmful_comments.append({
            "text": text,
            "label": "harmful"
        })


# ---------------------------------------
# 2. Load safe Burmese dataset
# ---------------------------------------

print("Loading safe Burmese dataset...")

safe_dataset = load_dataset(
    "chuuhtetnaing/myanmar-social-media-sentiment-analysis-dataset"
)

safe_comments = []

for row in safe_dataset["train"]:
    text = row["Text-MM"].strip()

    if text:
        safe_comments.append({
            "text": text,
            "label": "safe"
        })


# ---------------------------------------
# 3. Balance the datasets
# ---------------------------------------

random.seed(42)

# Use the same number of harmful and safe examples
number_of_examples = min(
    len(harmful_comments),
    len(safe_comments)
)

harmful_comments = random.sample(
    harmful_comments,
    number_of_examples
)

safe_comments = random.sample(
    safe_comments,
    number_of_examples
)


# ---------------------------------------
# 4. Combine datasets
# ---------------------------------------

combined_data = harmful_comments + safe_comments

# Shuffle the combined dataset
random.shuffle(combined_data)


# ---------------------------------------
# 5. Display information
# ---------------------------------------

print("\nDataset information:")
print("--------------------")

print("Harmful:", len(harmful_comments))
print("Safe:", len(safe_comments))
print("Total:", len(combined_data))

print("\nLabel distribution:")
print(
    Counter(
        row["label"]
        for row in combined_data
    )
)


# ---------------------------------------
# 6. Save dataset
# ---------------------------------------

output_file = "data/burmese/burmese_moderation_dataset.csv"

with open(
    output_file,
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=["text", "label"]
    )

    writer.writeheader()
    writer.writerows(combined_data)


print("\nDataset successfully saved!")
print(output_file)