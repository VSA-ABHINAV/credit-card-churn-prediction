# Package Project into Public GitHub Repository

## Objective
Create a clean, professional GitHub repository containing the complete Credit Card Churn Prediction project. The Code agent should initialize git, create a clean directory structure, commit all artifacts, and push to a new public GitHub repository.

## Repository Structure to Create

```
credit-card-churn-prediction/
├── README.md                    # Main documentation (see detailed spec below)
├── LICENSE                      # MIT License
├── .gitignore                   # Python/data/model exclusions
├── requirements.txt             # Pinned dependencies
├── data/
│   └── BankChurners.csv         # Dataset (or download script if too large)
├── src/
│   ├── __init__.py
│   ├── preprocessing.py         # Reusable preprocessing pipeline
│   ├── evaluate_models.py       # Baseline evaluation script
│   ├── feature_importance.py    # Feature importance analysis
│   ├── hyperparameter_tuning.py # RandomizedSearchCV script
│   ├── threshold_tuning.py      # Threshold optimization script
│   ├── final_model.py           # Final training + export script
│   └── predict.py               # Simple inference script using exported model
├── models/
│   ├── final_model.joblib       # Production model (from final_model.py)
│   └── final_model_config.json  # Model metadata + threshold
├── results/
│   ├── evaluation_results.json
│   ├── feature_importance.json
│   ├── tuning_results.json
│   ├── threshold_tuning_results.json
│   └── final_model_results.json
├── charts/
│   ├── feature_importance.png
│   ├── model_comparison_bars.png
│   ├── roc_curves.png
│   ├── confusion_matrices.png
│   └── cv_score_distribution.png
└── notebooks/
    ├── CreditCardRFC.ipynb      # Original exploratory notebook
    └── CreditCardLogistic.ipynb # Original exploratory notebook
```

## Detailed Requirements

### 1. README.md (Comprehensive, Business-Focused)
Write a professional README.md with these sections:

```markdown
# Credit Card Customer Churn Prediction

## 🎯 Business Problem
[2-3 paragraphs: Banks lose revenue when credit card customers leave. Proactive retention is 5-10x cheaper than acquisition. This project predicts which customers will churn using behavioral and demographic data, enabling targeted retention campaigns.]

## 📊 Dataset
- **Source**: BankChurners.csv (public dataset, ~10K customers)
- **Target**: Attrition_Flag (Existing vs Attrited Customer)
- **Class Balance**: ~16% churn rate (imbalanced)
- **Features**: 21 columns (demographics, account info, 6-month behavioral metrics)

## 🔬 Approach
1. **Preprocessing**: One-hot encode categoricals, standardize numerics, stratified 80/20 split
2. **Baseline Models**: Random Forest (100 trees) vs Logistic Regression
3. **Feature Importance**: Identified transaction behavior as primary signal (62% of importance)
4. **Hyperparameter Tuning**: RandomizedSearchCV (50 iter, 5-fold CV, F1 optimization)
5. **Threshold Tuning**: Selected 0.30 to prioritize recall (90% churn capture) for retention use case
6. **Cross-Validation**: Verified stability (Tuned RF: 86.7% ± 2.2% F1)
7. **Production Export**: Full-dataset model + config + threshold packaged for deployment

## 🏆 Results Summary

| Model | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|-----------|----------|-----------|--------|-----|---------|
| **Tuned RF (Production)** | **0.30** | **95.2%** | **81.6%** | **90.2%** | **85.7%** | **98.4%** |
| Tuned RF (Max F1) | 0.42 | 95.8% | 87.7% | 85.5% | 86.6% | 98.4% |
| Baseline RF | 0.50 | 95.8% | 93.5% | 79.4% | 85.9% | 98.4% |
| Logistic Regression | 0.50 | 90.0% | 76.8% | 53.8% | 63.3% | 91.7% |

**Key Insight**: Random Forest captures 90% of churners vs 54% for Logistic Regression — 118 more at-risk customers identified per 2,000.

## 📈 Top Predictive Features
1. **Total_Trans_Amt** (18.1%) — Total transaction amount
2. **Total_Trans_Ct** (16.1%) — Transaction frequency
3. **Total_Ct_Chng_Q4_Q1** (10.3%) — Quarterly usage change
4. **Total_Revolving_Bal** (10.0%) — Revolving balance
5. **Avg_Utilization_Ratio** (7.1%) — Credit utilization

*Behavioral signals dominate; demographics contribute <6% combined.*

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (reproduces all results)
python src/final_model.py

# Or just run inference with exported model
python src/predict.py --input new_customers.csv --output predictions.csv
```

### Inference Example
```python
import joblib, json, pandas as pd

model = joblib.load("models/final_model.joblib")
config = json.load(open("models/final_model_config.json"))

new_data = pd.read_csv("new_customers.csv")  # Must have 19 feature columns
proba = model.predict_proba(new_data)[:, 1]
churn_risk = (proba >= config["decision_threshold"]).astype(int)
```

## 📁 Repository Structure
[Table or tree view as above]

## 📋 Reproducing Results
| Script | Purpose | Output |
|--------|---------|--------|
| `python src/evaluate_models.py` | Baseline comparison | `results/evaluation_results.json` |
| `python src/feature_importance.py` | Feature ranking + chart | `results/feature_importance.json`, `charts/feature_importance.png` |
| `python src/hyperparameter_tuning.py` | RandomizedSearchCV | `results/tuning_results.json`, `report.md` |
| `python src/threshold_tuning.py` | Threshold optimization | `results/threshold_tuning_results.json`, `report.md` |
| `python src/final_model.py` | Full CV + export | `models/final_model.joblib`, `models/final_model_config.json`, `results/final_model_results.json`, all charts |

## ⚠️ Caveats
- Static dataset — no temporal validation; monitor drift in production
- Threshold 0.30 assumes retention offer cost << churn cost; adjust with real economics
- Features dominated by 6-month behavioral window; may not generalize to new customers with <6 months history

## 📄 License
MIT License — see LICENSE file.
```

### 2. LICENSE
Standard MIT License.

### 3. .gitignore
```gitignore
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Virtual environments
venv/
env/
.venv/

# Jupyter
.ipynb_checkpoints/

# Data (if large - keep BankChurners.csv if <10MB, otherwise use download script)
# data/*.csv

# Models (large binary files - track with Git LFS or exclude)
# models/*.joblib

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
*.swp
```

### 4. requirements.txt
Pinned versions based on current environment:
```
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.9.1
matplotlib==3.9.1
joblib==1.4.2
scipy==1.13.1
```

### 5. src/preprocessing.py
Extract the shared preprocessing logic into a reusable module:
```python
def get_preprocessor():
    """Return fitted ColumnTransformer for the churn dataset."""
    ...

def load_and_split_data(path, test_size=0.2, random_state=42):
    """Load CSV, apply encoding, return X_train, X_test, y_train, y_test."""
    ...

def get_feature_names():
    """Return list of expected input column names in order."""
    ...
```

### 6. src/predict.py
Simple CLI for inference:
```bash
python src/predict.py --input path/to/customers.csv --output predictions.csv
```
- Loads `models/final_model.joblib` and `models/final_model_config.json`
- Validates input columns match `expected_input_columns`
- Outputs CSV with original columns + `churn_probability` + `churn_prediction`

### 7. Git Operations
The Code agent must:
1. `git init` in project root
2. Create all files/directories per structure above
3. Copy existing artifacts to correct locations:
   - Move `*.joblib`, `*_config.json` → `models/`
   - Move `*_results.json`, `evaluation_results.json` → `results/`
   - Move `*.png` → `charts/`
   - Move `*.ipynb` → `notebooks/`
   - Move/create `src/` scripts (refactor from existing standalone scripts)
4. `git add .`
5. `git commit -m "Initial commit: Credit Card Churn Prediction project with production-ready Random Forest model"`
6. `git branch -M main`
7. **Create GitHub repo** via `gh repo create credit-card-churn-prediction --public --source=. --push`
   - Requires `gh` CLI authenticated (assume it is)
   - If `gh` not available, create repo manually and push with `git push -u origin main`

## Success Criteria
- Repository created on GitHub (public)
- All files committed with clean history
- README.md renders correctly on GitHub
- `requirements.txt` installs cleanly in fresh environment
- `python src/predict.py` works with exported model
- All charts present in `charts/`
- All results JSONs in `results/`
- Notebooks preserved in `notebooks/`
- No large files (>100MB) committed without LFS
- .gitignore excludes appropriate files