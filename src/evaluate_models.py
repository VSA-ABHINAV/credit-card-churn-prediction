"""
Evaluation Script for Credit Card Churn Models

Evaluates Random Forest Classifier and Logistic Regression on BankChurners.csv
following the exact preprocessing and training pipelines used in the notebooks.
Saves metrics to results/evaluation_results.json and prints a formatted comparison table.
"""

import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def resolve_paths():
    """Resolve data and output paths relative to repository root."""
    src_dir = Path(__file__).resolve().parent
    repo_root = src_dir.parent
    data_path = repo_root / "data" / "BankChurners.csv"
    if not data_path.exists():
        data_path = repo_root / "BankChurners.csv"
    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return data_path, results_dir / "evaluation_results.json"


def load_and_preprocess_data(dataset_path: Path):
    """Load dataset, map target and gender, drop CLIENTNUM, and create preprocessing pipeline."""
    print(f"[1/5] Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)

    print("[2/5] Applying data transformations and column mappings...")
    # Encode Gender: M -> 1, F -> 0
    if df["Gender"].dtype == object or str(df["Gender"].dtype) == "string":
        df["Gender"] = df["Gender"].map({"M": 1, "F": 0})

    # Drop CLIENTNUM and separate target
    X = df.drop(columns=["CLIENTNUM", "Attrition_Flag"])
    y = df["Attrition_Flag"].map(
        {
            "Existing Customer": 0,
            "Attrited Customer": 1,
        }
    )

    # Identify categorical and numeric columns
    categorical_cols = X.select_dtypes(include=["object", "string"]).columns
    numeric_cols = X.select_dtypes(include=["number"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("categories", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("numbers", StandardScaler(), numeric_cols),
        ]
    )

    # 80/20 stratified split with random_state=42
    print("[3/5] Splitting data (80/20 stratified split, random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return preprocessor, X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test) -> dict:
    """Compute Accuracy, Precision, Recall, F1 Score, and ROC-AUC on the test set."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }
    return metrics


def print_comparison_table(results: dict):
    """Print a clean comparison table to the console."""
    headers = ["Metric", "Random Forest", "Logistic Regression"]
    metrics_display = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1 Score", "f1"),
        ("ROC-AUC", "roc_auc"),
    ]

    col_widths = [22, 18, 22]
    separator = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"

    print("\n" + "=" * 68)
    print(" " * 18 + "MODEL EVALUATION COMPARISON")
    print("=" * 68)
    print(separator)
    header_row = (
        "| "
        + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        + " |"
    )
    print(header_row)
    print(separator)

    rf_res = results["random_forest"]
    lr_res = results["logistic_regression"]

    for label, key in metrics_display:
        rf_val = f"{rf_res[key]:.4f}"
        lr_val = f"{lr_res[key]:.4f}"
        row = (
            f"| {label.ljust(col_widths[0])} "
            f"| {rf_val.ljust(col_widths[1])} "
            f"| {lr_val.ljust(col_widths[2])} |"
        )
        print(row)

    print(separator)
    print(f"\nTest Set Size: {results['test_set_size']}")
    print(
        f"Class Distribution (Test Set): Existing (0)={results['class_distribution_test']['0']}, "
        f"Attrited (1)={results['class_distribution_test']['1']}\n"
    )


def main():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    dataset_path, output_json_path = resolve_paths()

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    # Load and preprocess data
    preprocessor, X_train, X_test, y_train, y_test = load_and_preprocess_data(dataset_path)

    # Train models
    print("[4/5] Training Random Forest Classifier...")
    rf_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(n_estimators=100, random_state=42),
            ),
        ]
    )
    rf_pipeline.fit(X_train, y_train)

    print("[4/5] Training Logistic Regression...")
    lr_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
    lr_pipeline.fit(X_train, y_train)

    # Evaluate models
    print("[5/5] Evaluating models on test split...")
    rf_metrics = evaluate_model(rf_pipeline, X_test, y_test)
    lr_metrics = evaluate_model(lr_pipeline, X_test, y_test)

    # Test distribution
    test_dist = y_test.value_counts().to_dict()
    class_distribution_test = {
        "0": int(test_dist.get(0, 0)),
        "1": int(test_dist.get(1, 0)),
    }

    results = {
        "random_forest": rf_metrics,
        "logistic_regression": lr_metrics,
        "test_set_size": int(len(y_test)),
        "class_distribution_test": class_distribution_test,
    }

    # Save results
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results successfully saved to: {output_json_path}")

    print_comparison_table(results)


if __name__ == "__main__":
    main()
