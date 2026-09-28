import pandas as pd

print("Preparing Burmese moderation model V2...")
# Load the V2 training and validation datasets
train_df = pd.read_csv("data/burmese/train_v2.csv")
validation_df = pd.read_csv("data/burmese/validation_v2.csv")

print("\nTraining examples:", len(train_df))
print("Validation examples:", len(validation_df))

print("\nTraining labels:")
print(train_df["label"].value_counts())

print("\nValidation labels:")
print(validation_df["label"].value_counts())

# Convert text labels to numbers
label_map = {
    "safe": 0,
    "harmful": 1
}

train_df["label"] = train_df["label"].map(label_map)
validation_df["label"] = validation_df["label"].map(label_map)

print("\nNumeric labels:")
print(train_df["label"].value_counts())

from transformers import AutoTokenizer

# Load the multilingual tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "FacebookAI/xlm-roberta-base"
)

print("\nTokenizer loaded successfully!")

from datasets import Dataset

# Convert pandas DataFrames to Hugging Face datasets
train_dataset = Dataset.from_pandas(train_df)
validation_dataset = Dataset.from_pandas(validation_df)

print("\nDatasets converted successfully!")
print("Training dataset:", train_dataset)
print("Validation dataset:", validation_dataset)

# Tokenize the Burmese text
def tokenize_data(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )


train_dataset = train_dataset.map(tokenize_data)
validation_dataset = validation_dataset.map(tokenize_data)

print("\nTokenization completed!")

from transformers import AutoModelForSequenceClassification

# Load the multilingual model for classification
model = AutoModelForSequenceClassification.from_pretrained(
    "FacebookAI/xlm-roberta-base",
    num_labels=2,
    id2label={
        0: "SAFE",
        1: "HARMFUL"
    },
    label2id={
        "SAFE": 0,
        "HARMFUL": 1
    }
)

print("\nV2 model loaded successfully!")

# Remove the original text column and prepare datasets for training
train_dataset = train_dataset.remove_columns(["text"])
validation_dataset = validation_dataset.remove_columns(["text"])

print("\nDatasets prepared for training!")
print("Training columns:", train_dataset.column_names)
print("Validation columns:", validation_dataset.column_names)

from transformers import TrainingArguments

# Training configuration
training_args = TrainingArguments(
    output_dir="models/burmese-moderator-v2",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_steps=10,
    report_to="none"
)

print("\nV2 training configuration created successfully!")

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


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


print("\nEvaluation metrics configured successfully!")

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    compute_metrics=compute_metrics,
)

print("\nTrainer created successfully!")

print("\nStarting Burmese moderation model V2 training...")

trainer.train()

print("\nV2 training completed!")

# Save the final V2 model and tokenizer
final_model_path = "models/burmese-moderator-v2/final"

trainer.save_model(final_model_path)
tokenizer.save_pretrained(final_model_path)

print("\nFinal V2 model saved to:")
print(final_model_path)