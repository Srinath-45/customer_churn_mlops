import os
import json
import pandas as pd


TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
OUTPUT_PATH = "artifacts/model/feature_columns.json"

TARGET_COLUMN = "Churn"


def build_features():
    """Validate processed features and save their column information."""

    print("Loading processed datasets...")

    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"Training file not found: {TRAIN_PATH}")

    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(f"Testing file not found: {TEST_PATH}")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # Check target column
    if TARGET_COLUMN not in train_df.columns:
        raise ValueError("Churn target column is missing from training data.")

    if TARGET_COLUMN not in test_df.columns:
        raise ValueError("Churn target column is missing from testing data.")

    # Separate feature columns
    train_features = train_df.drop(columns=[TARGET_COLUMN])
    test_features = test_df.drop(columns=[TARGET_COLUMN])

    # Ensure training and testing features match
    if list(train_features.columns) != list(test_features.columns):
        raise ValueError(
            "Training and testing feature columns do not match."
        )

    # Check missing values
    if train_features.isnull().any().any():
        raise ValueError("Missing values found in training features.")

    if test_features.isnull().any().any():
        raise ValueError("Missing values found in testing features.")

    # Save feature information
    feature_information = {
        "feature_count": len(train_features.columns),
        "features": list(train_features.columns)
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w") as file:
        json.dump(feature_information, file, indent=4)

    print("\nFeature validation completed successfully!")
    print("Number of features:", len(train_features.columns))
    print("Training shape:", train_features.shape)
    print("Testing shape:", test_features.shape)
    print("Feature information saved at:", OUTPUT_PATH)


if __name__ == "__main__":
    build_features()