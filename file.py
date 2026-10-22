import pandas as pd

# Combine train and test data with clean text and labels
combined = pd.concat([
    train_df[["clean_text", "label"]].assign(split="train"),
    test_df[["clean_text", "label"]].assign(split="test")
], ignore_index=True)

# Find cleaned texts that have multiple different labels
label_counts = combined.groupby("clean_text")["label"].nunique()
conflicts = label_counts[label_counts > 1]

print(f"Number of conflicting texts: {len(conflicts)}")

# Option 1: Remove conflicts from train and test
conflict_texts = conflicts.index.tolist()
clean_train_df = train_df[~train_df["clean_text"].isin(conflict_texts)].copy()
clean_test_df = test_df[~test_df["clean_text"].isin(conflict_texts)].copy()

print(f"Train size before: {len(train_df)}, after removing conflicts: {len(clean_train_df)}")
print(f"Test size before: {len(test_df)}, after removing conflicts: {len(clean_test_df)}")
