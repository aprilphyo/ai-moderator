import pandas as pd

print("Splitting Burmese moderation dataset V2...")
# Load the V2 dataset
df = pd.read_csv(
    "data/burmese/burmese_moderation_dataset_v2.csv"
)

print("\nV2 dataset loaded!")
print("Total examples:", len(df))
print("\nLabel distribution:")
print(df["label"].value_counts())

from sklearn.model_selection import train_test_split

# First split: 70% training, 30% temporary
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["label"]
)

print("\nFirst split:")
print("Training:", len(train_df))
print("Temporary:", len(temp_df))

# Second split: divide the temporary data equally
validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label"]
)

print("\nFinal splits:")
print("Training:", len(train_df))
print("Validation:", len(validation_df))
print("Testing:", len(test_df))

print("\nTraining labels:")
print(train_df["label"].value_counts())

print("\nValidation labels:")
print(validation_df["label"].value_counts())

print("\nTest labels:")
print(test_df["label"].value_counts())

# Save the three V2 datasets
train_df.to_csv(
    "data/burmese/train_v2.csv",
    index=False
)

validation_df.to_csv(
    "data/burmese/validation_v2.csv",
    index=False
)

test_df.to_csv(
    "data/burmese/test_v2.csv",
    index=False
)

print("\nV2 datasets successfully saved!")
print("Training: data/burmese/train_v2.csv")
print("Validation: data/burmese/validation_v2.csv")
print("Testing: data/burmese/test_v2.csv")