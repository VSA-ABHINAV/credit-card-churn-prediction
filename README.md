# Credit Card Customer Churn Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.5+-orange.svg)](https://scikit-learn.org/)

An end-to-end machine learning project for predicting credit card customer churn, optimizing retention intervention thresholds, and exporting production-ready inference artifacts.

---

## 🎯 Business Problem

Customer attrition is one of the most significant profit drains for retail banking and consumer credit divisions. When a credit card customer closes their account or ceases card usage, the bank permanently loses recurring interchange fees, annual membership charges, net interest margins on revolving debt, and cross-selling potential. Industry research consistently confirms that acquiring a new credit card customer is **5 to 10 times more expensive** than retaining an existing account.

Effective retention requires identifying at-risk cardholders *before* they initiate account closure, while their behavioral patterns first start diverging from normal account activity. However, standard off-the-shelf classification algorithms calibrated at an arbitrary 50% probability threshold often miss a catastrophic number of departing customers because churn is inherently an imbalanced outcome (~16% base rate).

This project develops, validates, and deploys a production-grade machine learning solution that accurately forecasts credit card customer churn. By carefully calibrating decision thresholds around actual business economics, the resulting model captures **90.2% of all at-risk customers** while maintaining high operational precision (81.6%), providing relationship managers with high-confidence targets for personalized retention outreach.

---

## 📊 Dataset

- **Source**: `BankChurners.csv` (10,127 customer accounts)
- **Target Variable**: `Attrition_Flag`
  - `Existing Customer`: **8,500** accounts (83.9%)
  - `Attrited Customer`: **1,627** accounts (16.1%)
- **Class Balance**: Moderately imbalanced binary classification (~16.1% churn rate)
- **Features**: 21 raw columns spanning customer demographics (age, gender, education, income category, marital status), relationship tenure, credit lines, revolving balances, credit utilization, and 6-month rolling transaction activity.

---

## 🔬 Approach

1. **Preprocessing Pipeline**: 
   - Excluded account identifiers (`CLIENTNUM`) and isolated target.
   - Applied one-hot encoding with unknown category handling for multi-class strings (`Education_Level`, `Marital_Status`, `Income_Category`, `Card_Category`).
   - Mapped binary gender (`M`→1, `F`→0).
   - Standardized all 15 numeric features using `StandardScaler`.
   - Stratified 80/20 train/test split (`random_state=42`) preserving class ratios.
2. **Baseline Models**: 
   - Built pipelines comparing 100-tree `RandomForestClassifier` against L2-regularized `LogisticRegression`.
3. **Feature Importance Analysis**: 
   - Extracted and aggregated one-hot encoded tree importances back to their 19 original parent features.
   - Identified rolling transaction metrics as the dominant churn predictors (~62% of total signal).
4. **Hyperparameter Tuning**: 
   - Conducted 50-iteration `RandomizedSearchCV` with 5-fold stratified cross-validation optimizing specifically for F1 score.
   - Determined optimal parameters: 400 trees, `max_depth=30`, `min_samples_split=10`, `min_samples_leaf=4`, `max_features=None`, `bootstrap=True`.
5. **Threshold Optimization**: 
   - Swept 41 probability thresholds (0.10 to 0.90 in increments of 0.02) to evaluate business scenarios (Balanced F1, High Recall, High Precision, Target Recall, Cost-Sensitive).
   - Selected **0.30** as the approved production threshold for proactive customer retention.
6. **Cross-Validation Verification**: 
   - Verified generalization stability via 5-fold stratified cross-validation on the training set (Tuned RF achieved 86.7% ± 2.2% F1).
7. **Production Model Export**: 
   - Retrained the approved pipeline on the entire 10,127 dataset.
   - Exported production binary (`models/final_model.joblib`), schema metadata (`models/final_model_config.json`), and standalone CLI inference tooling (`src/predict.py`).

---

## 🏆 Results Summary

Evaluated on the untouched held-out test set (2,026 customers: 1,701 Existing, 325 Attrited):

| Model | Threshold | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Specificity |
|---|---|---|---|---|---|---|---|
| **Tuned RF (Production)** | **0.30** | **95.2%** | **81.6%** | **90.2%** | **85.7%** | **98.4%** | **96.1%** |
| Tuned RF (Max F1) | 0.42 | 95.8% | 87.7% | 85.5% | 86.6% | 98.4% | 97.7% |
| Baseline RF | 0.50 | 95.8% | 93.5% | 79.4% | 85.9% | 98.4% | 98.9% |
| Logistic Regression | 0.50 | 90.0% | 76.8% | 53.8% | 63.3% | 91.7% | 96.9% |

**Key Insight**: The production Tuned Random Forest at threshold 0.30 captures **90.2% of all churners (293 out of 325)** compared to only **53.8% (175 out of 325)** for standard Logistic Regression — identifying **118 additional at-risk customers** per 2,000 accounts while maintaining an 81.6% precision rate.

---

## 📈 Top Predictive Features

Aggregation of Random Forest feature importances reveals that behavioral interaction signals overwhelmingly drive churn probability:

| Rank | Feature Name | Importance | Share (%) | Cumulative (%) | Interpretation |
|---|---|---|---|---|---|
| 1 | `Total_Trans_Amt` | 0.1813 | 18.1% | 18.1% | Aggregate 12-month transaction spend volume |
| 2 | `Total_Trans_Ct` | 0.1608 | 16.1% | 34.2% | Frequency of card swipes/transactions |
| 3 | `Total_Ct_Chng_Q4_Q1` | 0.1032 | 10.3% | 44.5% | Drop-off in transaction counts over time |
| 4 | `Total_Revolving_Bal` | 0.1001 | 10.0% | 54.5% | Active carrying credit balance |
| 5 | `Avg_Utilization_Ratio`| 0.0706 | 7.1% | 61.6% | Proportion of available credit actively used |

*The top 5 behavioral signals account for **61.6%** of total predictive importance. Demographic attributes (Education, Income, Marital Status, Gender) contribute less than 6% combined.*

---

## 📊 Visualizations

All visual evaluations are generated as publication-ready PNGs in the `charts/` directory:

| Visualization | Description |
|---|---|
| [`charts/model_comparison_bars.png`](charts/model_comparison_bars.png) | Metric-by-metric head-to-head comparison between Tuned RF @ 0.30 and Logistic Regression |
| [`charts/roc_curves.png`](charts/roc_curves.png) | Comparative ROC curves with AUC annotations (RF: 0.9844 vs LR: 0.9166) and operating point |
| [`charts/confusion_matrices.png`](charts/confusion_matrices.png) | Side-by-side confusion matrices highlighting reduction in false negatives |
| [`charts/cv_score_distribution.png`](charts/cv_score_distribution.png) | Boxplots of 5-fold cross-validation F1 score distributions across all candidate models |
| [`charts/feature_importance.png`](charts/feature_importance.png) | Horizontal bar chart of the top 10 most influential features |

---

## 🚀 Quick Start

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/VSA-ABHINAV/credit-card-churn-prediction.git
cd credit-card-churn-prediction
pip install -r requirements.txt
```

### 2. Run Batch Inference with Pretrained Model
Run predictions on new customer records using the CLI:
```bash
python src/predict.py --input data/BankChurners.csv --output predictions.csv
```

### 3. Programmatic Python Inference
```python
import joblib, json, pandas as pd

# Load pipeline and metadata
model = joblib.load("models/final_model.joblib")
config = json.load(open("models/final_model_config.json"))

# Read new accounts
new_customers = pd.read_csv("new_customers.csv")

# Encode Gender if passed as string ('M'/'F')
if "Gender" in new_customers.columns:
    new_customers["Gender"] = new_customers["Gender"].replace({"M": 1, "F": 0}).astype(int)

# Extract expected features in exact order
X = new_customers[config["expected_input_columns"]]

# Score and classify
probabilities = model.predict_proba(X)[:, 1]
churn_predictions = (probabilities >= config["decision_threshold"]).astype(int)

new_customers["churn_probability"] = probabilities.round(4)
new_customers["churn_risk_flag"] = churn_predictions
```

---

## 🌐 API Usage (FastAPI)

A high-performance production REST API is implemented in `src/api.py` using FastAPI and Pydantic. It provides real-time single and batch predictions, health checks, metadata inspection, and bulk CSV scoring.

### 1. Start the Server
Start the API locally using Uvicorn:
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
# Or run with python directly
python src/api.py
```
Interactive Swagger UI documentation is available at `http://localhost:8000/docs`.

### 2. Available Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness & readiness probe (reports model load status and version) |
| `GET` | `/model-info` | Returns model hyperparameters, decision threshold (0.30), and feature schema |
| `POST` | `/predict` | Predict churn probability, risk flag, and risk level (`LOW`, `MEDIUM`, `HIGH`) for single customer or list |
| `POST` | `/predict/batch` | Bulk scoring via CSV file upload (`multipart/form-data`), returns downloadable CSV |

### 3. Example `curl` Commands

#### Single Customer Prediction:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Customer_Age": 45,
    "Gender": "M",
    "Dependent_count": 3,
    "Education_Level": "High School",
    "Marital_Status": "Married",
    "Income_Category": "$60K - $80K",
    "Card_Category": "Blue",
    "Months_on_book": 39,
    "Total_Relationship_Count": 5,
    "Months_Inactive_12_mon": 1,
    "Contacts_Count_12_mon": 3,
    "Credit_Limit": 12691.0,
    "Total_Revolving_Bal": 777,
    "Avg_Open_To_Buy": 11914.0,
    "Total_Amt_Chng_Q4_Q1": 1.335,
    "Total_Trans_Amt": 1144,
    "Total_Trans_Ct": 42,
    "Total_Ct_Chng_Q4_Q1": 1.625,
    "Avg_Utilization_Ratio": 0.061
  }'
```

**Response (`200 OK`):**
```json
{
  "predictions": [
    {
      "churn_probability": 0.0713,
      "churn_risk_flag": 0,
      "risk_level": "LOW"
    }
  ],
  "threshold_used": 0.3,
  "model_version": "2026-10-24T14:41:48.514082+00:00"
}
```

#### Batch JSON Array Prediction:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '[
    {"Customer_Age": 45, "Gender": "M", "Dependent_count": 3, ...},
    {"Customer_Age": 55, "Gender": "F", "Dependent_count": 1, ...}
  ]'
```

#### Bulk CSV Scoring:
```bash
curl -X POST http://localhost:8000/predict/batch \
  -F "file=@data/BankChurners.csv" \
  --output scored_customers.csv
```
*Appends `churn_probability`, `churn_risk_flag`, and `risk_level` directly to the returned CSV.*

### 4. Python `requests` Example
```python
import requests

customer_data = {
    "Customer_Age": 45,
    "Gender": "M",
    "Dependent_count": 3,
    "Education_Level": "High School",
    "Marital_Status": "Married",
    "Income_Category": "$60K - $80K",
    "Card_Category": "Blue",
    "Months_on_book": 39,
    "Total_Relationship_Count": 5,
    "Months_Inactive_12_mon": 1,
    "Contacts_Count_12_mon": 3,
    "Credit_Limit": 12691.0,
    "Total_Revolving_Bal": 777,
    "Avg_Open_To_Buy": 11914.0,
    "Total_Amt_Chng_Q4_Q1": 1.335,
    "Total_Trans_Amt": 1144,
    "Total_Trans_Ct": 42,
    "Total_Ct_Chng_Q4_Q1": 1.625,
    "Avg_Utilization_Ratio": 0.061
}

response = requests.post("http://localhost:8000/predict", json=customer_data)
result = response.json()
print("Risk Level:", result["predictions"][0]["risk_level"])
print("Churn Probability:", result["predictions"][0]["churn_probability"])
print("Flag:", result["predictions"][0]["churn_risk_flag"])
```

---

## 📁 Repository Structure

```
credit-card-churn-prediction/
├── README.md                    # Project overview, results, and usage guide
├── LICENSE                      # MIT Open Source License
├── .gitignore                   # Standard Python/IDE/model exclusions
├── requirements.txt             # Pinned environment dependencies
├── data/
│   └── BankChurners.csv         # Complete customer dataset (10,127 records)
├── static/
│   └── index.html               # Single-page web UI dashboard
├── src/
│   ├── __init__.py              # Package initializer
│   ├── api.py                   # Production FastAPI REST API application
│   ├── preprocessing.py         # Modular preprocessing and column transformer logic
│   ├── evaluate_models.py       # Baseline evaluation script (RF vs Logistic Regression)
│   ├── feature_importance.py    # Feature importance extraction and aggregation
│   ├── hyperparameter_tuning.py # RandomizedSearchCV hyperparameter optimization
│   ├── threshold_tuning.py      # Multi-scenario decision threshold optimization
│   ├── final_model.py           # Production cross-validation, test evaluation, and export
│   └── predict.py               # Standalone batch inference CLI script
├── models/
│   ├── final_model.joblib       # Fully-trained production pipeline (trained on all 10,127 rows)
│   └── final_model_config.json  # Model hyperparameters, decision threshold, and schema metadata
├── results/
│   ├── evaluation_results.json  # Baseline evaluation metrics
│   ├── feature_importance.json  # Ranked feature importances and cumulative percentages
│   ├── tuning_results.json      # Hyperparameter tuning cross-validation results
│   ├── threshold_tuning_results.json # 41-threshold sweep and business scenario metrics
│   └── final_model_results.json # Full cross-validation and test set results
├── charts/
│   ├── feature_importance.png   # Top 10 feature importance bar chart
│   ├── model_comparison_bars.png# Head-to-head test set metrics comparison
│   ├── roc_curves.png           # Comparative ROC curves with operational threshold
│   ├── confusion_matrices.png   # Side-by-side confusion matrices
│   └── cv_score_distribution.png# 5-fold CV stability boxplots
└── notebooks/
    ├── CreditCardRFC.ipynb      # Original exploratory notebook (Random Forest)
    └── CreditCardLogistic.ipynb # Original exploratory notebook (Logistic Regression)
```

---

## 📋 Reproducing Results

Each pipeline phase can be independently executed via modular scripts in `src/`:

| Script | Purpose | Output Artifacts |
|---|---|---|
| `python src/evaluate_models.py` | Baseline comparison of Random Forest vs Logistic Regression | `results/evaluation_results.json` |
| `python src/feature_importance.py` | Feature ranking and parent feature aggregation | `results/feature_importance.json`, `charts/feature_importance.png` |
| `python src/hyperparameter_tuning.py` | 50-iteration RandomizedSearchCV tuning on 5-fold CV | `results/tuning_results.json` |
| `python src/threshold_tuning.py` | Multi-scenario probability threshold evaluation (41 thresholds) | `results/threshold_tuning_results.json` |
| `python src/final_model.py` | 5-fold CV, held-out test evaluation, charts, and model export | `models/final_model.joblib`, `models/final_model_config.json`, `results/final_model_results.json`, all 4 charts |
| `python src/predict.py --input <csv>` | Batch prediction on incoming customer accounts | Enriched predictions CSV with churn probabilities and flags |

---

## ⚠️ Caveats & Operational Considerations

1. **Static Dataset & Concept Drift**: The dataset represents a historical snapshot without longitudinal time stamps. In production, transaction patterns shift seasonally and macroeconomically. Monthly or quarterly data drift monitoring on key behavioral metrics (`Total_Trans_Amt`, `Total_Trans_Ct`, `Total_Ct_Chng_Q4_Q1`) is recommended.
2. **Economic Calibration of Threshold**: The operational threshold of `0.30` was selected under the premise that customer acquisition/replacement cost substantially exceeds the cost of a retention incentive ($5\times$ to $10\times$). If retention campaign costs increase (e.g. expensive loyalty gifts vs automated zero-cost fee waivers), the decision threshold should be re-calibrated using `src/threshold_tuning.py`.
3. **Account Age Requirements**: The model heavily relies on rolling transaction trends over the past 6 to 12 months. New cardholders (tenure < 6 months) may lack sufficient transaction velocity history for accurate risk assessment; specialized onboarding churn models should be developed for accounts under 6 months old.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
