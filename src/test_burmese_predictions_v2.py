import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "models/burmese-moderator-v2/final"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()

label_names = {
    0: "SAFE",
    1: "HARMFUL"
}

test_comments = [
    "မင်းကို ရိုက်မယ်",
    "ဒီနေ့ အရမ်းပျော်တယ်",
    "မင်းက အရမ်းမိုက်တယ်",
    "ဒီနေ့ ရာသီဥတုက အရမ်းကောင်းတယ်",
    "မင်းကို သတ်မယ်",
    "မင်းနဲ့ စကားပြောရတာ ဝမ်းသာတယ်",
]

print("\nBurmese Moderation V2 Predictions")
print("=" * 40)

for text in test_comments:

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

    print(f"\nComment: {text}")
    print(f"Prediction: {label_names[prediction]}")
    print(f"Confidence: {confidence * 100:.2f}%")