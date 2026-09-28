from transformers import pipeline

# Load our fine-tuned Burmese moderation model
classifier = pipeline(
    "text-classification",
    model="models/burmese-moderator/checkpoint-147"
)

# New Burmese examples for testing
test_comments = [
    "ဒီနေ့ အရမ်းပျော်တယ်",
    "မင်းက အရမ်းမိုက်တယ်",
    "မင်းကို ရိုက်မယ်",
    "မင်းက လူမိုက်ပဲ",
    "ဒီနေ့ ရာသီဥတုက အရမ်းကောင်းတယ်"
]

print("\nBurmese Model Predictions")
print("=" * 40)

for comment in test_comments:
    result = classifier(comment)

    print("\nComment:", comment)
    print("Prediction:", result[0]["label"])
    print("Confidence:", round(result[0]["score"] * 100, 2), "%")
    