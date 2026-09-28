import pandas as pd
import numpy as np

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

print("Testing Burmese moderation model V2...")

# -----------------------------
# 1. Load test dataset
# -----------------------------

test_df = pd.read_csv("data/burmese/test_v2.csv")

print("\nTest examples:", len(test_df))

print("\nTest labels:")
print(test_df["label"].value_counts())


# -----------------------------
# 2. Convert labels
# -----------------------------

label_map = {
    "safe": 0,
    "harmful": 1
}

test_df["label"] = test_df["label"].map(label_map)


# -----------------------------
# 3. Convert to Hugging Face Dataset
# -----------------------------

test_dataset = Dataset.from_pandas(test_df)


# -----------------------------
# 4. Load tokenizer
# -----------------------------

model_path = "models/burmese-moderator-v2/final"

tokenizer = AutoTokenizer.from_pretrained(model_path)


# -----------------------------
# 5. Tokenize test data
# -----------------------------

def tokenize_data(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )


test_dataset = test_dataset.map(tokenize_data)


# -----------------------------
# 6. Prepare dataset
# -----------------------------

test_dataset = test_dataset.remove_columns(["text"])


# -----------------------------
# 7. Load trained model
# -----------------------------

model = AutoModelForSequenceClassification.from_pretrained(
    model_path
)

print("\nV2 model loaded successfully!")


# -----------------------------
# 8. Evaluation metrics
# -----------------------------

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0
    )

    accuracy = accuracy_score(labels, predictions)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# -----------------------------
# 9. Create Trainer
# -----------------------------

trainer = Trainer(
    model=model,
    compute_metrics=compute_metrics
)


# -----------------------------
# 10. Evaluate
# -----------------------------

print("\nRunning test evaluation...")

results = trainer.evaluate(test_dataset)

print("\n==============================")
print("V2 TEST RESULTS")
print("==============================")

print(f"Accuracy:  {results['eval_accuracy']:.4f}")
print(f"Precision: {results['eval_precision']:.4f}")
print(f"Recall:    {results['eval_recall']:.4f}")
print(f"F1 Score:  {results['eval_f1']:.4f}")