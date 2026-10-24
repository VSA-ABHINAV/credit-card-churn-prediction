"""
Feature Importance Analysis for Random Forest Model

Loads BankChurners.csv, trains Random Forest model with ColumnTransformer preprocessing,
aggregates one-hot encoded feature importances back to original features,
prints a ranked table, saves results to results/feature_importance.json,
and generates charts/feature_importance.png for the top 10 features.
"""

from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def resolve_paths():
    """Resolve data, results, and charts directories relative to repository root."""
    src_dir = Path(__file__).resolve().parent
    repo_root = src_dir.parent
    data_path = repo_root / "data" / "BankChurners.csv"
    if not data_path.exists():
        data_path = repo_root / "BankChurners.csv"
    results_dir = repo_root / "results"
    charts_dir = repo_root / "charts"
    results_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    return data_path, results_dir / "feature_importance.json", charts_dir / "feature_importance.png"


def load_and_preprocess_data(dataset_path: Path):
    """Load dataset, encode targets, and prepare feature splits."""
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

    return preprocessor, X_train, X_test, y_train, y_test, categorical_cols, numeric_cols, list(X.columns)


def aggregate_feature_importances(pipeline, categorical_cols, original_cols):
    """Extract and aggregate one-hot encoded feature importances back to original features."""
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    transformed_feature_names = preprocessor.get_feature_names_out()
    raw_importances = classifier.feature_importances_

    agg_importances = defaultdict(float)
    sorted_cat_cols = sorted(categorical_cols, key=len, reverse=True)

    for feat_name, imp in zip(transformed_feature_names, raw_importances):
        if feat_name.startswith("numbers__"):
            orig = feat_name[len("numbers__"):]
            agg_importances[orig] += float(imp)
        elif feat_name.startswith("categories__"):
            clean_name = feat_name[len("categories__"):]
            matched = False
            for cat_col in sorted_cat_cols:
                if clean_name.startswith(f"{cat_col}_"):
                    agg_importances[cat_col] += float(imp)
                    matched = True
                    break
            if not matched:
                agg_importances[clean_name] += float(imp)
        else:
            agg_importances[feat_name] += float(imp)

    for col in original_cols:
        if col not in agg_importances:
            agg_importances[col] = 0.0

    sorted_features = sorted(agg_importances.items(), key=lambda x: x[1], reverse=True)
    total_imp = sum(agg_importances.values())
    ranked_list = []
    cumulative = 0.0

    for feat, imp in sorted_features:
        pct = (imp / total_imp) * 100 if total_imp > 0 else 0.0
        cumulative += pct
        ranked_list.append(
            {
                "feature": feat,
                "importance": round(imp, 4),
                "percentage": round(pct, 2),
                "cumulative_percentage": round(cumulative, 2),
            }
        )

    return ranked_list, len(transformed_feature_names), len(original_cols)


def print_ranked_table(ranked_features: list):
    """Print formatted ranking table to console."""
    headers = ["Rank", "Feature Name", "Importance", "Percentage (%)", "Cumulative (%)"]
    col_widths = [6, 26, 14, 16, 16]
    separator = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"

    print("\n" + "=" * 86)
    print(" " * 24 + "RANDOM FOREST FEATURE IMPORTANCE")
    print("=" * 86)
    print(separator)
    header_row = (
        "| "
        + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        + " |"
    )
    print(header_row)
    print(separator)

    for i, item in enumerate(ranked_features, start=1):
        feat = item["feature"]
        imp_str = f"{item['importance']:.4f}"
        pct_str = f"{item['percentage']:.2f}%"
        cum_str = f"{item['cumulative_percentage']:.2f}%"
        row = (
            f"| {str(i).rjust(col_widths[0])} "
            f"| {feat.ljust(col_widths[1])} "
            f"| {imp_str.rjust(col_widths[2])} "
            f"| {pct_str.rjust(col_widths[3])} "
            f"| {cum_str.rjust(col_widths[4])} |"
        )
        print(row)

    print(separator + "\n")


def save_visualization(ranked_features: list, output_image_path: Path):
    """Save horizontal bar chart of top 10 features as PNG."""
    top_10 = ranked_features[:10]
    features = [item["feature"] for item in top_10][::-1]
    importances = [item["importance"] for item in top_10][::-1]
    percentages = [item["percentage"] for item in top_10][::-1]

    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(features, importances, color="#1f77b4", edgecolor="#0b3c61")

        for bar, pct in zip(bars, percentages):
            width = bar.get_width()
            ax.text(
                width + 0.003,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.4f} ({pct:.1f}%)",
                va="center",
                fontsize=9,
                color="#333333",
            )

        ax.set_title("Top 10 Feature Importances (Random Forest)", fontsize=13, pad=15)
        ax.set_xlabel("Importance Score", fontsize=10)
        ax.set_xlim(0, max(importances) * 1.25)
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_image_path, dpi=150)
        plt.close()
        print(f"Chart successfully saved to: {output_image_path}")
    except Exception as e:
        print(f"Error generating chart: {e}")


def main():
    dataset_path, output_json_path, output_image_path = resolve_paths()

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    # Load and preprocess data
    (
        preprocessor,
        X_train,
        X_test,
        y_train,
        y_test,
        categorical_cols,
        numeric_cols,
        original_cols,
    ) = load_and_preprocess_data(dataset_path)

    # Train Random Forest
    print("[4/5] Training Random Forest model (n_estimators=100, random_state=42)...")
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(n_estimators=100, random_state=42),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    # Extract and aggregate feature importances
    print("[5/5] Extracting and aggregating feature importances...")
    ranked_features, total_encoded_features, original_features_count = (
        aggregate_feature_importances(pipeline, categorical_cols, original_cols)
    )

    top_5 = [item["feature"] for item in ranked_features[:5]]

    results = {
        "feature_importances": ranked_features,
        "top_5_features": top_5,
        "total_features_after_encoding": int(total_encoded_features),
        "original_features_count": int(original_features_count),
    }

    # Save to JSON
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results successfully saved to: {output_json_path}")

    # Save visualization
    save_visualization(ranked_features, output_image_path)

    # Print ranked comparison table
    print_ranked_table(ranked_features)


if __name__ == "__main__":
    main()
