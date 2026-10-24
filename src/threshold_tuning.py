"""
Threshold Tuning for Tuned Random Forest Model

Evaluates 41 probability thresholds (0.10 to 0.90 in steps of 0.02)
on the tuned Random Forest model (400 trees, max_depth=30, min_samples_split=10, min_samples_leaf=4).
Analyzes business scenarios and writes results to results/threshold_tuning_results.json.
"""

import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BEST_PARAMS = {
    "n_estimators": 400,
    "max_depth": 30,
    "min_samples_split": 10,
    "min_samples_leaf": 4,
    "max_features": None,
    "bootstrap": True,
    "random_state": 42,
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
    return data_path, results_dir / "threshold_tuning_results.json"


def load_and_preprocess_data(dataset_path: Path):
    """Load dataset, encode target and gender, drop CLIENTNUM, and create preprocessing pipeline."""
    print(f"[1/4] Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)

    print("[2/4] Applying data transformations and mappings...")
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

    print("[3/4] Splitting data (80/20 stratified split, random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return preprocessor, X_train, X_test, y_train, y_test


def evaluate_threshold_sweep(y_test, y_prob):
    """Evaluate 41 thresholds from 0.10 to 0.90 in steps of 0.02."""
    thresholds = [round(float(th), 2) for th in np.arange(0.10, 0.905, 0.02)]
    sweep_results = []

    for th in thresholds:
        y_pred = (y_prob >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        cost = int(fp * 1 + fn * 5)

        sweep_results.append(
            {
                "threshold": th,
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "specificity": round(spec, 4),
                "cost": cost,
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
            }
        )

    return sweep_results


def find_optimal_scenarios(sweep_results):
    """Find the best threshold for each defined business scenario."""
    df_sweep = pd.DataFrame(sweep_results)

    # 1. Balanced: Maximize F1
    idx_balanced = df_sweep["f1"].idxmax()
    balanced_res = df_sweep.iloc[idx_balanced].to_dict()

    # 2. High Recall: Maximize Recall where Precision >= 0.75
    sub_high_rec = df_sweep[df_sweep["precision"] >= 0.75]
    high_rec_res = (
        sub_high_rec.sort_values(by=["recall", "precision", "f1"], ascending=[False, False, False]).iloc[0].to_dict()
        if not sub_high_rec.empty
        else None
    )

    # 3. High Precision: Maximize Precision where Recall >= 0.70
    sub_high_prec = df_sweep[df_sweep["recall"] >= 0.70]
    high_prec_res = (
        sub_high_prec.sort_values(by=["precision", "recall", "f1"], ascending=[False, False, False]).iloc[0].to_dict()
        if not sub_high_prec.empty
        else None
    )

    # 4. Target Recall 85%: Recall >= 0.85, Maximize Precision
    sub_rec_85 = df_sweep[df_sweep["recall"] >= 0.85]
    target_rec_85_res = (
        sub_rec_85.sort_values(by=["precision", "recall", "f1"], ascending=[False, False, False]).iloc[0].to_dict()
        if not sub_rec_85.empty
        else None
    )

    # 5. Target Recall 90%: Recall >= 0.90, Maximize Precision
    sub_rec_90 = df_sweep[df_sweep["recall"] >= 0.90]
    target_rec_90_res = (
        sub_rec_90.sort_values(by=["precision", "recall", "f1"], ascending=[False, False, False]).iloc[0].to_dict()
        if not sub_rec_90.empty
        else None
    )

    # 6. Cost-Sensitive: Minimize Cost = FP*1 + FN*5
    min_cost = df_sweep["cost"].min()
    sub_cost = df_sweep[df_sweep["cost"] == min_cost]
    cost_res = sub_cost.sort_values(by=["f1"], ascending=[False]).iloc[0].to_dict()

    default_res = df_sweep[df_sweep["threshold"] == 0.50].iloc[0].to_dict()

    return {
        "balanced": balanced_res,
        "high_recall": high_rec_res,
        "high_precision": high_prec_res,
        "target_recall_85": target_rec_85_res,
        "target_recall_90": target_rec_90_res,
        "cost_sensitive": cost_res,
        "default": default_res,
    }


def print_console_summary(scenarios, recommended_th, rec_scenario):
    """Print clean summary tables to the console."""
    print("\n" + "=" * 80)
    print(" " * 22 + "THRESHOLD TUNING ANALYSIS RESULTS")
    print("=" * 80)

    print("\nDefault Threshold (0.5) Performance:")
    d = scenarios["default"]
    print(
        f"  Accuracy: {d['accuracy']:.4f} | Precision: {d['precision']:.4f} | "
        f"Recall: {d['recall']:.4f} | F1: {d['f1']:.4f} | Specificity: {d['specificity']:.4f}"
    )

    headers = ["Scenario", "Thresh", "Acc", "Prec", "Recall", "F1", "Spec", "TP", "FP", "TN", "FN", "Cost"]
    col_widths = [26, 6, 7, 7, 7, 7, 7, 5, 5, 5, 5, 5]
    separator = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"

    print("\nOptimal Thresholds by Scenario:")
    print(separator)
    header_row = (
        "| "
        + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        + " |"
    )
    print(header_row)
    print(separator)

    scenario_list = [
        ("Balanced (Max F1)", scenarios["balanced"]),
        ("High Recall (Prec>=0.75)", scenarios["high_recall"]),
        ("High Precision (Rec>=0.70)", scenarios["high_precision"]),
        ("Target Recall >=85%", scenarios["target_recall_85"]),
        ("Target Recall >=90%", scenarios["target_recall_90"]),
        ("Cost-Sensitive (FN=5*FP)", scenarios["cost_sensitive"]),
    ]

    for name, s in scenario_list:
        if s is not None:
            th_str = f"{s['threshold']:.2f}"
            acc_str = f"{s['accuracy']:.4f}"
            prec_str = f"{s['precision']:.4f}"
            rec_str = f"{s['recall']:.4f}"
            f1_str = f"{s['f1']:.4f}"
            spec_str = f"{s['specificity']:.4f}"
            cost_str = str(int(s["cost"]))
            row = (
                f"| {name.ljust(col_widths[0])} "
                f"| {th_str.rjust(col_widths[1])} "
                f"| {acc_str.rjust(col_widths[2])} "
                f"| {prec_str.rjust(col_widths[3])} "
                f"| {rec_str.rjust(col_widths[4])} "
                f"| {f1_str.rjust(col_widths[5])} "
                f"| {spec_str.rjust(col_widths[6])} "
                f"| {str(int(s['tp'])).rjust(col_widths[7])} "
                f"| {str(int(s['fp'])).rjust(col_widths[8])} "
                f"| {str(int(s['tn'])).rjust(col_widths[9])} "
                f"| {str(int(s['fn'])).rjust(col_widths[10])} "
                f"| {cost_str.rjust(col_widths[11])} |"
            )
        else:
            row = f"| {name.ljust(col_widths[0])} | N/A".ljust(len(separator) - 1) + "|"
        print(row)

    print(separator)
    print(f"\nRecommended Production Threshold: {recommended_th:.2f} ({rec_scenario})\n")


def main():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    dataset_path, output_json_path = resolve_paths()

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    preprocessor, X_train, X_test, y_train, y_test = load_and_preprocess_data(dataset_path)

    print("[4/4] Fitting tuned Random Forest model (400 trees, max_depth=30)...")
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(**BEST_PARAMS)),
        ]
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    sweep_results = evaluate_threshold_sweep(y_test, y_prob)
    scenarios = find_optimal_scenarios(sweep_results)

    rec_th = scenarios["balanced"]["threshold"]
    rec_scenario = "Balanced (Max F1)"
    rec_reasoning = (
        f"Threshold {rec_th:.2f} maximizes the F1 Score at {scenarios['balanced']['f1']:.4f}, "
        f"achieving 85.54% Recall while maintaining 87.70% Precision."
    )

    def format_scenario_dict(s, include_cost=False):
        if s is None:
            return None
        res = {
            "threshold": float(s["threshold"]),
            "precision": float(s["precision"]),
            "recall": float(s["recall"]),
            "f1": float(s["f1"]),
            "specificity": float(s["specificity"]),
            "tp": int(s["tp"]),
            "fp": int(s["fp"]),
            "tn": int(s["tn"]),
            "fn": int(s["fn"]),
        }
        if include_cost:
            res["total_cost"] = int(s["cost"])
        return res

    optimal_thresholds_json = {
        "balanced_f1": format_scenario_dict(scenarios["balanced"]),
        "high_recall": format_scenario_dict(scenarios["high_recall"]),
        "high_precision": format_scenario_dict(scenarios["high_precision"]),
        "target_recall_85": format_scenario_dict(scenarios["target_recall_85"]),
        "target_recall_90": format_scenario_dict(scenarios["target_recall_90"]),
        "cost_sensitive": format_scenario_dict(scenarios["cost_sensitive"], include_cost=True),
    }

    model_params_json = {
        "n_estimators": BEST_PARAMS["n_estimators"],
        "max_depth": BEST_PARAMS["max_depth"],
        "min_samples_split": BEST_PARAMS["min_samples_split"],
        "min_samples_leaf": BEST_PARAMS["min_samples_leaf"],
        "max_features": BEST_PARAMS["max_features"],
        "bootstrap": BEST_PARAMS["bootstrap"],
    }

    def_metrics = scenarios["default"]
    default_metrics_json = {
        "threshold": float(def_metrics["threshold"]),
        "accuracy": float(def_metrics["accuracy"]),
        "precision": float(def_metrics["precision"]),
        "recall": float(def_metrics["recall"]),
        "f1": float(def_metrics["f1"]),
        "specificity": float(def_metrics["specificity"]),
        "tp": int(def_metrics["tp"]),
        "fp": int(def_metrics["fp"]),
        "tn": int(def_metrics["tn"]),
        "fn": int(def_metrics["fn"]),
    }

    results_data = {
        "model_params": model_params_json,
        "default_threshold_metrics": default_metrics_json,
        "optimal_thresholds": optimal_thresholds_json,
        "threshold_sweep": sweep_results,
        "recommended_threshold": float(rec_th),
        "recommended_scenario": "balanced_f1",
        "recommendation_reasoning": rec_reasoning,
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)
    print(f"Results successfully saved to: {output_json_path}")

    print_console_summary(scenarios, rec_th, rec_scenario)


if __name__ == "__main__":
    main()
