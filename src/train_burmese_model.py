import pandas as pd

# Load the Burmese dataset splits
train_df = pd.read_csv("data/burmese/train.csv")
validation_df = pd.read_csv("data/burmese/validation.csv")
test_df = pd.read_csv("data/burmese/test.csv")

print("Training:", len(train_df))
print("Validation:", len(validation_df))
print("Testing:", len(test_df))


# Convert labels to numbers
label_map = {
    "safe": 0,
    "harmful": 1
}

train_df["label"] = train_df["label"].map(label_map)
validation_df["label"] = validation_df["label"].map(label_map)
test_df["label"] = test_df["label"].map(label_map)

print("\nConverted labels:")
print(train_df.head())

print("\nConverted labels:")
print(train_df.head())

import pandas as pd
from transformers import AutoTokenizer
# Load the XLM-RoBERTa tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "FacebookAI/xlm-roberta-base"
)

print("\nTokenizer loaded successfully!")
# Test the tokenizer with a Burmese sentence
sample_text = train_df.iloc[0]["text"]

tokens = tokenizer(
    sample_text,
    truncation=True,
    padding="max_length",
    max_length=128
)

print("\nTokenizer test:")
print(tokens)
from datasets import Dataset

# Convert pandas DataFrames to Hugging Face datasets
train_dataset = Dataset.from_pandas(train_df)
validation_dataset = Dataset.from_pandas(validation_df)
test_dataset = Dataset.from_pandas(test_df)

print("\nHugging Face datasets created!")
print(train_dataset)
# Tokenize the datasets
def tokenize_data(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )


train_dataset = train_dataset.map(tokenize_data)
validation_dataset = validation_dataset.map(tokenize_data)
test_dataset = test_dataset.map(tokenize_data)

print("\nDatasets tokenized successfully!")
print(train_dataset[0])

from transformers import AutoModelForSequenceClassification

# Load XLM-RoBERTa for binary classification
model = AutoModelForSequenceClassification.from_pretrained(
    "FacebookAI/xlm-roberta-base",
    num_labels=2
)

print("\nXLM-RoBERTa model loaded successfully!")

from transformers import TrainingArguments

# Configure the training process
training_args = TrainingArguments(
    output_dir="models/burmese-moderator",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none"
)

print("\nTraining configuration created successfully!")

import evaluate
import numpy as np

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


print("\nEvaluation metrics configured successfully!")
from transformers import Trainer

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    compute_metrics=compute_metrics
)

print("\nTrainer created successfully!")
# Start model training
print("\nStarting Burmese model training...")

trainer.train()

print("\nTraining completed!")