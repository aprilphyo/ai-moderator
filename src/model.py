from transformers import pipeline

# Load the AI model only once
toxicity_classifier = pipeline(
    "text-classification",
    model="unitary/toxic-bert"
)