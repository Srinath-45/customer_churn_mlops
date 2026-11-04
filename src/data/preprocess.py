
import os
import yaml
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def load_parameters():
    with open("params.yaml", "r") as file:
        return yaml.safe_load(file)


def preprocess_data():
    # 1. Load configuration
    params = load_parameters()

    raw_path = params["data"]["raw_path"]
    processed_dir = params["data"]["processed_dir"]

    # 2. Load raw dataset
    df = pd.read_csv(raw_path)

    print("Original dataset shape:", df.shape)

    # 3. Remove customer ID
    df = df.drop(columns=["customerID"])

    # 4. Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

    # 5. Separate features and target
    X = df.drop(columns=["Churn"])
    y = df["Churn"].map({"Yes": 1, "No": 0})

    if y.isna().any():
        raise ValueError("Unexpected values found in Churn column.")

    # 6. Identify numerical and categorical columns
    numerical_columns = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    # 7. Build preprocessing pipelines
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ])

    # 8. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params["preprocessing"]["test_size"],
        random_state=params["preprocessing"]["random_state"],
        stratify=y
    )

    # 9. Fit preprocessing only on training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # 10. Get feature names
    feature_names = preprocessor.get_feature_names_out()

    train_df = pd.DataFrame(
        X_train_processed,
        columns=feature_names,
        index=X_train.index
    )
    train_df["Churn"] = y_train

    test_df = pd.DataFrame(
        X_test_processed,
        columns=feature_names,
        index=X_test.index
    )
    test_df["Churn"] = y_test

    # 11. Create output directories
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs("artifacts/preprocessor", exist_ok=True)

    # 12. Save processed datasets
    train_df.to_csv(
        os.path.join(processed_dir, "train.csv"), index=False
    )
    test_df.to_csv(
        os.path.join(processed_dir, "test.csv"), index=False
    )

    # 13. Save fitted preprocessor for future predictions
    joblib.dump(
        preprocessor,
        "artifacts/preprocessor/preprocessor.joblib"
    )

    print("Preprocessing completed successfully!")
    print("Training data shape:", train_df.shape)
    print("Testing data shape:", test_df.shape)


if __name__ == "__main__":
    preprocess_data()
