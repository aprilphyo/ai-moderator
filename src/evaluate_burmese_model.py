import pandas as pd
import numpy as np
import evaluate
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer
)

# Load the unseen test data
test_df = pd.read_csv("data/burmese/test.csv")
# Convert labels to numbers
label_map = {
    "safe": 0,
    "harmful": 1
}

test_df["label"] = test_df["label"].map(label_map)

# Convert the test data to a Hugging Face dataset
test_dataset = Dataset.from_pandas(test_df)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "FacebookAI/xlm-roberta-base"
)

# Load the best trained model
model = AutoModelForSequenceClassification.from_pretrained(
    "models/burmese-moderator/checkpoint-147"
)

# Tokenize the test data
def tokenize_data(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

test_dataset = test_dataset.map(tokenize_data)

# Load evaluation metrics
accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_metric.compute(
        predictions=predictions,
        references=labels
    )

    precision = precision_metric.compute(
        predictions=predictions,
        references=labels,
        average="binary"
    )

    recall = recall_metric.compute(
        predictions=predictions,
        references=labels,
        average="binary"
    )

    f1 = f1_metric.compute(
        predictions=predictions,
        references=labels,
        average="binary"
    )

    return {
        "accuracy": accuracy["accuracy"],
        "precision": precision["precision"],
        "recall": recall["recall"],
        "f1": f1["f1"]
    }


# Create the evaluator
trainer = Trainer(
    model=model,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# Evaluate the model
print("\nEvaluating Burmese moderation model...")

results = trainer.evaluate()

print("\nTest Results:")
print("-------------------------")
print(f"Accuracy:  {results['eval_accuracy']:.4f}")
print(f"Precision: {results['eval_precision']:.4f}")
print(f"Recall:    {results['eval_recall']:.4f}")
print(f"F1 Score:  {results['eval_f1']:.4f}")