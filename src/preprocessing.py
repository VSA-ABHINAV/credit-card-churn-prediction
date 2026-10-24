"""
Reusable Data Preprocessing Pipeline for Credit Card Churn Prediction

Encapsulates data loading, column mappings, feature selection,
ColumnTransformer creation, and train/test splitting.
"""

from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL_COLUMNS: List[str] = [
    "Education_Level",
    "Marital_Status",
    "Income_Category",
    "Card_Category",
]

NUMERIC_COLUMNS: List[str] = [
    "Customer_Age",
    "Gender",
    "Dependent_count",
    "Months_on_book",
    "Total_Relationship_Count",
    "Months_Inactive_12_mon",
    "Contacts_Count_12_mon",
    "Credit_Limit",
    "Total_Revolving_Bal",
    "Avg_Open_To_Buy",
    "Total_Amt_Chng_Q4_Q1",
    "Total_Trans_Amt",
    "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1",
    "Avg_Utilization_Ratio",
]

EXPECTED_INPUT_COLUMNS: List[str] = [
    "Customer_Age",
    "Gender",
    "Dependent_count",
    "Education_Level",
    "Marital_Status",
    "Income_Category",
    "Card_Category",
    "Months_on_book",
    "Total_Relationship_Count",
    "Months_Inactive_12_mon",
    "Contacts_Count_12_mon",
    "Credit_Limit",
    "Total_Revolving_Bal",
    "Avg_Open_To_Buy",
    "Total_Amt_Chng_Q4_Q1",
    "Total_Trans_Amt",
    "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1",
    "Avg_Utilization_Ratio",
]


def resolve_data_path(custom_path: Union[str, Path, None] = None) -> Path:
    """Resolve dataset path checking custom location, data/ directory, and root."""
    if custom_path:
        p = Path(custom_path)
        if p.exists():
            return p.resolve()

    # Common candidates
    candidates = [
        Path("data/BankChurners.csv"),
        Path("BankChurners.csv"),
        Path(__file__).resolve().parent.parent / "data" / "BankChurners.csv",
        Path(__file__).resolve().parent.parent / "BankChurners.csv",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()

    raise FileNotFoundError(
        "BankChurners.csv not found in data/ or root directory. Please specify path explicitly."
    )


def get_feature_names() -> List[str]:
    """Return list of expected input column names in order."""
    return list(EXPECTED_INPUT_COLUMNS)


def get_preprocessor(
    categorical_cols: Union[List[str], None] = None,
    numeric_cols: Union[List[str], None] = None,
) -> ColumnTransformer:
    """Return an unfitted ColumnTransformer configured for categorical OHE and numeric standard scaling."""
    cat_cols = categorical_cols if categorical_cols is not None else CATEGORICAL_COLUMNS
    num_cols = numeric_cols if numeric_cols is not None else NUMERIC_COLUMNS

    return ColumnTransformer(
        transformers=[
            ("categories", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("numbers", StandardScaler(), num_cols),
        ]
    )


def load_raw_data(path: Union[str, Path, None] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Load BankChurners CSV, apply gender mapping, drop CLIENTNUM, and return X, y."""
    data_path = resolve_data_path(path)
    df = pd.read_csv(data_path)

    # Encode Gender: M -> 1, F -> 0
    df["Gender"] = df["Gender"].replace({"M": 1, "F": 0}).astype(int)

    X = df.drop(columns=["CLIENTNUM", "Attrition_Flag"])
    y = df["Attrition_Flag"].map(
        {
            "Existing Customer": 0,
            "Attrited Customer": 1,
        }
    )
    return X, y


def load_and_split_data(
    path: Union[str, Path, None] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load dataset and return stratified train/test split (X_train, X_test, y_train, y_test)."""
    X, y = load_raw_data(path)
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
