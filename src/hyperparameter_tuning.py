"""
Hyperparameter Tuning for Random Forest Model

Performs RandomizedSearchCV (50 iterations, 5-fold CV, scoring='f1')
on BankChurners.csv with the standardized preprocessing pipeline.
Evaluates the best model on the test split and writes results to results/tuning_results.json.
"""

import json
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASELINE_METRICS = {
    "accuracy": 0.9580,
    "precision": 0.9412,
    "recall": 0.7877,
    "f1": 0.8576,
    "roc_auc": 0.9844,
}


def resolve_paths():
    """Resolve data and results paths relative to repository root."""
    src_dir = Path(__file__).resolve().parent
    repo_root = src_dir.parent
    data_path = repo_root / "data" / "BankChurners.csv"
    if not data_path.exists():
        data_path = repo_root / "BankChurners.csv"
    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return data_path, results_dir / "tuning_results.json", repo_root / "report.md"


def load_and_preprocess_data(dataset_path: Path):
    """Load dataset, encode target and gender, drop CLIENTNUM, and create preprocessing pipeline."""
    print(f"[1/5] Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)

    print("[2/5] Applying data transformations and mappings...")
    if df["Gender"].dtype == object or str(df["Gender"].dtype) == "string":
        df["Gender"] = df["Gender"].map({"M": 1, "F": 0})

    X = df.drop(columns=["CLIENTNUM", "Attrition_Flag"])
    y = df["Attrition_Flag"].map(
        {
            "Existing Customer": 0,
            "Attrited Customer": 1,
        }
    )

    categorical_cols = list(X.select_dtypes(include=["object", "string"]).columns)
    numeric_cols = list(X.select_dtypes(include=["number"]).columns)

    preprocessor = ColumnTransformer(
        transformers=[
            ("categories", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("numbers", StandardScaler(), numeric_cols),
        ]
    )

    print("[3/5] Splitting data (80/20 stratified split, random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return preprocessor, X_train, X_test, y_train, y_test


def run_hyperparameter_tuning(preprocessor, X_train, y_train):
    """Run RandomizedSearchCV on Random Forest pipeline."""
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )

    param_dist = {
        "classifier__n_estimators": [100, 200, 300, 400, 500],
        "classifier__max_depth": [None, 10, 20, 30, 40, 50],
        "classifier__min_samples_split": [2, 5, 10, 15, 20],
        "classifier__min_samples_leaf": [1, 2, 4, 6, 8],
        "classifier__max_features": ["sqrt", "log2", None],
        "classifier__bootstrap": [True, False],
    }

    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=50,
        cv=5,
        scoring="f1",
        random_state=42,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    print("[4/5] Running RandomizedSearchCV (50 iterations, 5 folds = 250 fits)...")
    start_time = time.time()
    random_search.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    print(f"Hyperparameter search completed in {elapsed_time:.2f} seconds.")

    return random_search, elapsed_time


def evaluate_model(model, X_test, y_test) -> dict:
    """Compute test set metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
    }
    return metrics


def print_comparison_table(best_params, cv_f1, cv_f1_std, test_metrics, baseline_metrics, improvements):
    """Print clean summary to console."""
    print("\n" + "=" * 76)
    print(" " * 20 + "RANDOM FOREST TUNING RESULTS")
    print("=" * 76)
    print(f"Best CV F1 Score: {cv_f1:.4f} (±{cv_f1_std:.4f})")
    print("\nBest Parameters:")
    for k, v in best_params.items():
        print(f"  - {k}: {v}")

    headers = ["Metric", "Baseline RF", "Tuned RF", "Improvement"]
    col_widths = [16, 14, 14, 14]
    separator = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"

    print("\n" + separator)
    header_row = (
        "| "
        + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        + " |"
    )
    print(header_row)
    print(separator)

    metric_labels = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1 Score", "f1"),
        ("ROC-AUC", "roc_auc"),
    ]

    for label, k in metric_labels:
        base_v = f"{baseline_metrics[k]:.4f}"
        tuned_v = f"{test_metrics[k]:.4f}"
        imp_v = improvements[k]
        imp_str = f"+{imp_v:.2f}%" if imp_v >= 0 else f"{imp_v:.2f}%"
        row = (
            f"| {label.ljust(col_widths[0])} "
            f"| {base_v.rjust(col_widths[1])} "
            f"| {tuned_v.rjust(col_widths[2])} "
            f"| {imp_str.rjust(col_widths[3])} |"
        )
        print(row)

    print(separator + "\n")


def main():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    dataset_path, output_json_path, _ = resolve_paths()

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    # Step 1: Preprocess Data
    preprocessor, X_train, X_test, y_train, y_test = load_and_preprocess_data(dataset_path)

    # Step 2: Run RandomizedSearchCV
    random_search, search_time = run_hyperparameter_tuning(preprocessor, X_train, y_train)

    # Step 3: Extract Best Results
    best_index = random_search.best_index_
    best_cv_f1 = round(float(random_search.best_score_), 4)
    best_cv_f1_std = round(float(random_search.cv_results_["std_test_score"][best_index]), 4)

    raw_best_params = random_search.best_params_
    best_params = {
        "n_estimators": raw_best_params["classifier__n_estimators"],
        "max_depth": raw_best_params["classifier__max_depth"],
        "min_samples_split": raw_best_params["classifier__min_samples_split"],
        "min_samples_leaf": raw_best_params["classifier__min_samples_leaf"],
        "max_features": raw_best_params["classifier__max_features"],
        "bootstrap": raw_best_params["classifier__bootstrap"],
    }

    # Step 4: Evaluate Best Model on Test Set
    print("[5/5] Evaluating best model on test set...")
    best_model = random_search.best_estimator_
    test_metrics = evaluate_model(best_model, X_test, y_test)

    improvements = {}
    for metric_name, base_val in BASELINE_METRICS.items():
        tuned_val = test_metrics[metric_name]
        pct_improvement = ((tuned_val - base_val) / base_val) * 100
        improvements[metric_name] = round(pct_improvement, 2)

    results_json_data = {
        "best_params": best_params,
        "best_cv_f1": best_cv_f1,
        "best_cv_f1_std": best_cv_f1_std,
        "test_metrics": test_metrics,
        "baseline_metrics": BASELINE_METRICS,
        "improvement": improvements,
        "search_time_seconds": round(search_time, 2),
        "n_iter": 50,
        "cv_folds": 5,
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results_json_data, f, indent=2)
    print(f"Results successfully saved to: {output_json_path}")

    print_comparison_table(
        best_params, best_cv_f1, best_cv_f1_std, test_metrics, BASELINE_METRICS, improvements
    )


if __name__ == "__main__":
    main()
