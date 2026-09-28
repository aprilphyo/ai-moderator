from datasets import load_dataset


# ---------------------------------------
# 1. Load our combined Burmese dataset
# ---------------------------------------

dataset = load_dataset(
    "csv",
    data_files="data/burmese/burmese_moderation_dataset.csv"
)

dataset = dataset["train"]

print("Original dataset:")
print(dataset)


# ---------------------------------------
# 2. Split into 80% training and
#    20% temporary data
# ---------------------------------------

train_test = dataset.train_test_split(
    test_size=0.20,
    seed=42
)

train_dataset = train_test["train"]
temporary_dataset = train_test["test"]


# ---------------------------------------
# 3. Split the temporary 20% into
#    10% validation and 10% test
# ---------------------------------------

validation_test = temporary_dataset.train_test_split(
    test_size=0.50,
    seed=42
)

validation_dataset = validation_test["train"]
test_dataset = validation_test["test"]


# ---------------------------------------
# 4. Print dataset sizes
# ---------------------------------------

print("\nDataset sizes:")
print("----------------")
print("Training:", len(train_dataset))
print("Validation:", len(validation_dataset))
print("Testing:", len(test_dataset))


# ---------------------------------------
# 5. Save the datasets
# ---------------------------------------

train_dataset.to_csv(
    "data/burmese/train.csv",
    index=False
)

validation_dataset.to_csv(
    "data/burmese/validation.csv",
    index=False
)

test_dataset.to_csv(
    "data/burmese/test.csv",
    index=False
)


print("\nDatasets successfully saved!")

print("data/burmese/train.csv")
print("data/burmese/validation.csv")
print("data/burmese/test.csv")