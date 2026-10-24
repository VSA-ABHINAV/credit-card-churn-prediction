"""
Inference Script for Credit Card Churn Prediction

Loads the production Random Forest model and metadata configuration,
validates input columns, computes churn probabilities, and outputs
predictions according to the approved decision threshold.

Usage:
    python src/predict.py --input path/to/customers.csv --output predictions.csv
"""

import argparse
import json
from pathlib import Path
import sys

import joblib
import pandas as pd


def resolve_model_files(model_path=None, config_path=None):
    """Locate final_model.joblib and final_model_config.json."""
    base_dir = Path(__file__).resolve().parent.parent

    if not model_path:
        candidates = [
            base_dir / "models" / "final_model.joblib",
            base_dir / "final_model.joblib",
            Path("models/final_model.joblib"),
            Path("final_model.joblib"),
        ]
        for c in candidates:
            if c.exists():
                model_path = c
                break

    if not config_path:
        candidates = [
            base_dir / "models" / "final_model_config.json",
            base_dir / "final_model_config.json",
            Path("models/final_model_config.json"),
            Path("final_model_config.json"),
        ]
        for c in candidates:
            if c.exists():
                config_path = c
                break

    if not model_path or not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found. Checked default models/final_model.joblib")
    if not config_path or not Path(config_path).exists():
        raise FileNotFoundError(f"Configuration file not found. Checked default models/final_model_config.json")

    return Path(model_path), Path(config_path)


def predict(input_path: str, output_path: str, model_file=None, config_file=None):
    """Run model inference on input CSV and save enriched predictions."""
    model_p, config_p = resolve_model_files(model_file, config_file)

    print(f"Loading production model from: {model_p}")
    model = joblib.load(model_p)

    print(f"Loading configuration from: {config_p}")
    with open(config_p, "r", encoding="utf-8") as f:
        config = json.load(f)

    threshold = float(config.get("decision_threshold", 0.30))
    expected_cols = config.get("expected_input_columns", [])

    print(f"Reading input data from: {input_path}")
    df_raw = pd.read_csv(input_path)
    df_features = df_raw.copy()

    # Preprocess Gender if provided as string
    if "Gender" in df_features.columns:
        df_features["Gender"] = df_features["Gender"].replace({"M": 1, "F": 0})
        try:
            df_features["Gender"] = df_features["Gender"].astype(int)
        except Exception:
            pass

    # Validate required columns
    missing_cols = [c for c in expected_cols if c not in df_features.columns]
    if missing_cols:
        raise ValueError(f"Input data is missing required feature columns: {missing_cols}")

    # Reorder columns to expected schema
    X_input = df_features[expected_cols]

    # Predict probabilities and apply threshold
    probabilities = model.predict_proba(X_input)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    # Attach results to output DataFrame
    output_df = df_raw.copy()
    output_df["churn_probability"] = probabilities.round(4)
    output_df["churn_prediction"] = predictions

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(out_p, index=False)
    print(f"Predictions successfully written to: {out_p}")

    total_count = len(output_df)
    churn_count = int(predictions.sum())
    churn_pct = (churn_count / total_count * 100) if total_count > 0 else 0.0

    print("\nInference Summary:")
    print(f"  - Total Customers Processed: {total_count}")
    print(f"  - Flagged for Attrition (Churn): {churn_count} ({churn_pct:.1f}%)")
    print(f"  - Classified as Existing: {total_count - churn_count} ({100 - churn_pct:.1f}%)")
    print(f"  - Decision Threshold Applied: {threshold:.2f}\n")


def main():
    parser = argparse.ArgumentParser(description="Predict Credit Card Churn Risk")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input CSV containing customer records")
    parser.add_argument("--output", "-o", type=str, default="predictions.csv", help="Path to write output predictions CSV")
    parser.add_argument("--model", "-m", type=str, default=None, help="Optional custom path to model.joblib")
    parser.add_argument("--config", "-c", type=str, default=None, help="Optional custom path to config.json")

    args = parser.parse_args()
    predict(args.input, args.output, args.model, args.config)


if __name__ == "__main__":
    main()
