"""
data.py

Handles everything to do with the UCI Adult dataset:
- downloading it
- combining the official train/test files into one dataset
- checking which features act as proxies for sex (Cramer's V)
- preprocessing it into train/val/test splits ready for a model
"""

import os
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Where the raw files get downloaded to / read from
DATA_DIR = "data"
TRAIN_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
TEST_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

# Column names, since the raw files don't include a header row
COLUMN_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
    "hours_per_week", "native_country", "income",
]

# All categorical columns that could plausibly go into the model. Whichever one
# is chosen as the sensitive attribute gets excluded from this list at preprocess time.
ALL_CATEGORICAL = [
    "workclass", "education", "marital_status", "occupation",
    "relationship", "race", "sex", "native_country",
]
NUMERIC_FEATURES = [
    "age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week",
]


def download():
    """Downloads the two raw Adult dataset files if they're not already on disk."""
    os.makedirs(DATA_DIR, exist_ok=True)
    train_path = os.path.join(DATA_DIR, "adult.data")
    test_path = os.path.join(DATA_DIR, "adult.test")

    if not os.path.exists(train_path):
        print("Downloading adult.data...")
        urllib.request.urlretrieve(TRAIN_URL, train_path)

    if not os.path.exists(test_path):
        print("Downloading adult.test...")
        urllib.request.urlretrieve(TEST_URL, test_path)

    return train_path, test_path


def load_adult():
    """Loads the Adult dataset and combines the train + test files into one DataFrame."""
    train_path, test_path = download()

    train_df = pd.read_csv(train_path, names=COLUMN_NAMES, sep=r",\s*", engine="python", na_values="?")

    # the test file has an extra header-like first line and a trailing "." on income labels
    test_df = pd.read_csv(test_path, names=COLUMN_NAMES, sep=r",\s*", engine="python", na_values="?", skiprows=1)
    test_df["income"] = test_df["income"].str.rstrip(".")

    df = pd.concat([train_df, test_df], ignore_index=True)
    return df


def proxy_check(df, sensitive_attr="sex"):
    """
    Checks how strongly each other categorical feature is associated with the
    given sensitive attribute, using Cramer's V (0 = no association, 1 = perfect).
    This is how we found that 'relationship' almost gives away sex directly.

    When sensitive_attr="race", groups into White / Non-White first, matching
    preprocess()'s grouping -- so the reported association numbers are
    consistent with what the actual race-focused experiment measures.
    """
    df = df.copy()
    if sensitive_attr == "race":
        df["race"] = df["race"].apply(lambda r: "White" if r == "White" else "Non-White")

    def cramers_v(col):
        table = pd.crosstab(df[col], df[sensitive_attr])
        chi2 = chi2_contingency(table)[0]
        n = table.sum().sum()
        r, k = table.shape
        return np.sqrt(chi2 / (n * (min(r, k) - 1)))

    other_cols = [c for c in ALL_CATEGORICAL if c != sensitive_attr]
    results = []
    for col in other_cols:
        results.append({"feature": col, "cramers_v": cramers_v(col)})

    results_df = pd.DataFrame(results).sort_values("cramers_v", ascending=False)
    return results_df.reset_index(drop=True)


def preprocess(df, seed, sensitive_attr="sex", train_size=0.6, val_size=0.2):
    """
    Cleans and splits the dataset into train/val/test sets.

    - drops rows with missing values (workclass, occupation, native_country)
    - keeps the chosen sensitive_attr as a separate array (the model never sees
      it as a feature); the other candidate attribute stays in as a normal feature
    - one-hot encodes the other categorical columns, scales the numeric ones
    - returns everything the model and fairness code need

    When sensitive_attr="race", the 5 raw categories are grouped into White /
    Non-White before splitting. This only happens here, not in load_adult(),
    so it never silently affects runs where race is just a regular input
    feature (e.g. the sex-focused pipeline) -- those keep the original 5
    categories. The grouping is needed because the smallest raw categories
    (Other: 406 rows, Amer-Indian-Eskimo: 470 rows total) are too small to
    give stable per-group fairness metrics once split into train/val/test --
    the smallest test-set group ends up with only ~65 people.
    """
    df = df.copy()
    if sensitive_attr == "race":
        df["race"] = df["race"].apply(lambda r: "White" if r == "White" else "Non-White")

    df = df.dropna().reset_index(drop=True)

    categorical_features = [c for c in ALL_CATEGORICAL if c != sensitive_attr]

    y = (df["income"] == ">50K").astype(int).values
    sensitive = df[sensitive_attr].values  # kept separate, not used as a model input

    X_raw = df[categorical_features + NUMERIC_FEATURES]

    # first split off the test set, then split the remainder into train/val
    X_temp, X_test, y_temp, y_test, sens_temp, sens_test = train_test_split(
        X_raw, y, sensitive, test_size=(1 - train_size - val_size), random_state=seed, stratify=y
    )
    val_ratio_of_temp = val_size / (train_size + val_size)
    X_train, X_val, y_train, y_val, sens_train, sens_val = train_test_split(
        X_temp, y_temp, sens_temp, test_size=val_ratio_of_temp, random_state=seed, stratify=y_temp
    )

    # one-hot encode categoricals, scale numeric features -- fit only on the training set
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ("num", StandardScaler(), NUMERIC_FEATURES),
    ])

    X_train_enc = preprocessor.fit_transform(X_train)
    X_val_enc = preprocessor.transform(X_val)
    X_test_enc = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    return {
        "X_train": X_train_enc, "X_val": X_val_enc, "X_test": X_test_enc,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "sensitive_train": sens_train, "sensitive_val": sens_val, "sensitive_test": sens_test,
        "feature_names": feature_names,
    }


if __name__ == "__main__":
    # quick manual check that everything works end to end, for both sensitive attributes
    df = load_adult()
    print("Loaded dataset:", df.shape)

    for attr in ["sex", "race"]:
        proxy_results = proxy_check(df, sensitive_attr=attr)
        print(f"\nProxy feature check (association with {attr}):")
        print(proxy_results.to_string(index=False))

        split = preprocess(df, seed=0, sensitive_attr=attr)
        print(f"\n[{attr}] Train/val/test sizes:", split["X_train"].shape[0], split["X_val"].shape[0], split["X_test"].shape[0])
        print(f"[{attr}] Number of encoded features:", split["X_train"].shape[1])
