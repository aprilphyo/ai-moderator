import pandas as pd
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ==========================================
# 1. Paths
# ==========================================

MODEL_PATH = "models/burmese-moderator-v2/final"
TEST_PATH = "data/burmese/test_v2.csv"


# ==========================================
# 2. Load test dataset
# ==========================================

print("Loading V2 test dataset...")

df = pd.read_csv(TEST_PATH)

print(f"Test examples: {len(df)}")


# ==========================================
# 3. Load model
# ==========================================

print("\nLoading V2 model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()

print("V2 model loaded!")


# ==========================================
# 4. Label mapping
# ==========================================

label_map = {
    0: "SAFE",
    1: "HARMFUL"
}

reverse_label_map = {
    "safe": 0,
    "harmful": 1
}


# ==========================================
# 5. Analyze every test example
# ==========================================

results = []

print("\nAnalyzing test examples...")

for index, row in df.iterrows():

    text = str(row["text"])
    actual_label = reverse_label_map[row["label"]]

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=-1)

    prediction = torch.argmax(probabilities, dim=-1).item()

    confidence = probabilities[0][prediction].item()

    predicted_label = label_map[prediction]

    correct = prediction == actual_label

    results.append({
        "text": text,
        "actual": label_map[actual_label],
        "predicted": predicted_label,
        "confidence": confidence,
        "correct": correct
    })


# ==========================================
# 6. Create results dataframe
# ==========================================

results_df = pd.DataFrame(results)


# ==========================================
# 7. Overall results
# ==========================================

total = len(results_df)
correct = results_df["correct"].sum()
incorrect = total - correct

accuracy = correct / total

print("\n==========================================")
print("V2 TEST ANALYSIS")
print("==========================================")

print(f"Total examples: {total}")
print(f"Correct: {correct}")
print(f"Incorrect: {incorrect}")
print(f"Accuracy: {accuracy * 100:.2f}%")


# ==========================================
# 8. False positives
# ==========================================

false_positives = results_df[
    (results_df["actual"] == "SAFE") &
    (results_df["predicted"] == "HARMFUL")
]

print("\n==========================================")
print("FALSE POSITIVES")
print("==========================================")

print(f"Total false positives: {len(false_positives)}")

if len(false_positives) > 0:

    print(
        false_positives[
            ["text", "confidence"]
        ].to_string(index=False)
    )


# ==========================================
# 9. False negatives
# ==========================================

false_negatives = results_df[
    (results_df["actual"] == "HARMFUL") &
    (results_df["predicted"] == "SAFE")
]

print("\n==========================================")
print("FALSE NEGATIVES")
print("==========================================")

print(f"Total false negatives: {len(false_negatives)}")

if len(false_negatives) > 0:

    print(
        false_negatives[
            ["text", "confidence"]
        ].to_string(index=False)
    )


# ==========================================
# 10. Save complete analysis
# ==========================================

output_path = "data/burmese/v2_test_analysis.csv"

results_df.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("\n==========================================")
print("Analysis saved!")
print("==========================================")
print(f"Saved to: {output_path}")