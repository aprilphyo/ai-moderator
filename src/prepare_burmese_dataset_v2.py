import pandas as pd
from datasets import load_dataset

print("Preparing Burmese moderation dataset V2...")
# Load the Burmese harmful-content dataset
harmful_dataset = load_dataset("simbolo-ai/burmese-hatespeech")

harmful_df = harmful_dataset["train"].to_pandas()

print("\nHarmful dataset loaded!")
print("Total harmful examples:", len(harmful_df))

# Load the Burmese social-media sentiment dataset
safe_dataset = load_dataset(
    "chuuhtetnaing/myanmar-social-media-sentiment-analysis-dataset"
)

safe_df = safe_dataset["train"].to_pandas()

print("\nSafe/background dataset loaded!")
print("Total examples:", len(safe_df))

# Clean the safe/background dataset
safe_df = safe_df[["Text-MM", "Sentiment"]].copy()

safe_df["Text-MM"] = safe_df["Text-MM"].astype(str).str.strip()
safe_df["Sentiment"] = safe_df["Sentiment"].astype(str).str.strip()

# Remove empty texts
safe_df = safe_df[safe_df["Text-MM"].str.len() > 0]

print("\nCleaned safe/background dataset:")
print("Total examples:", len(safe_df))
print(safe_df[["Text-MM", "Sentiment"]].head())


# Sentiment categories we consider suitable as safe/background examples
safe_sentiments = [
    "Positive",
    "Joy",
    "Excitement",
    "Contentment",
    "Neutral",
    "Gratitude",
    "Curiosity",
    "Serenity",
    "Happy",
    "Nostalgia",
    "Awe",
    "Hopeful",
    "Acceptance",
    "Pride",
    "Elation",
    "Euphoria",
    "Enthusiasm",
    "Determination",
    "Surprise",
    "Playful",
    "Inspiration",
    "Happiness",
    "Hope",
    "Empowerment",
    "Inspired",
    "Admiration",
    "Calmness",
    "Compassion",
    "Tenderness",
    "Fulfillment",
    "Reverence",
    "Proud",
    "Grateful",
    "Compassionate",
    "Reflection",
    "Enchantment",
    "Love",
    "Amusement",
    "Anticipation",
    "Kind",
    "Empathetic",
    "Free-spirited",
    "Confident",
    "Satisfaction",
    "Accomplishment",
    "Harmony",
    "Creativity",
    "Wonder",
    "Adventure",
    "Enjoyment",
    "Affection",
    "Adoration",
    "Zest",
    "Yearning",
    "Exploration",
    "Captivation",
    "Tranquility",
    "Mischievous"
]

safe_df = safe_df[
    safe_df["Sentiment"].isin(safe_sentiments)
].copy()

print("\nSelected safe/background examples:")
print(safe_df["Sentiment"].value_counts())
print("Total selected:", len(safe_df))

# Select the same number of harmful examples as safe examples
harmful_df = harmful_df.sample(
    n=len(safe_df),
    random_state=42
).copy()

print("\nBalanced dataset:")
print("Safe examples:", len(safe_df))
print("Harmful examples:", len(harmful_df))
print("Total examples:", len(safe_df) + len(harmful_df))

# Prepare the safe examples
safe_final = safe_df[["Text-MM"]].copy()
safe_final = safe_final.rename(columns={"Text-MM": "text"})
safe_final["label"] = "safe"

# Prepare the harmful examples
harmful_final = harmful_df[["text"]].copy()
harmful_final["label"] = "harmful"

print("\nSafe examples ready:")
print(safe_final.head())

print("\nHarmful examples ready:")
print(harmful_final.head())

# Combine safe and harmful examples
combined_df = pd.concat(
    [safe_final, harmful_final],
    ignore_index=True
)

# Shuffle the dataset
combined_df = combined_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

print("\nCombined and shuffled dataset:")
print(combined_df.head(10))
print("\nLabel distribution:")
print(combined_df["label"].value_counts())
print("\nTotal examples:", len(combined_df))

# Save the V2 dataset
output_path = "data/burmese/burmese_moderation_dataset_v2.csv"

combined_df.to_csv(
    output_path,
    index=False
)

print("\nV2 dataset successfully saved!")
print("Saved to:", output_path)