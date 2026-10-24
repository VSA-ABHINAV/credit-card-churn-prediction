# Final Model Report - Credit Card Churn Prediction

## Executive Summary
Following rigorous hyperparameter tuning, cross-validation, and decision threshold analysis, the **Tuned Random Forest Classifier** operated at a decision threshold of **0.30** is selected as the production model for credit card churn prediction. Operating at threshold 0.30 captures **90.15% of all churners** (293 out of 325 test churners) while maintaining **81.62% precision**, **96.12% specificity**, and an exceptional **0.9844 ROC-AUC**. Compared to standard Logistic Regression which misses 150 churners (recall 53.85%), the tuned Random Forest at threshold 0.30 leaves only 32 churners undetected, perfectly fulfilling the high-impact business retention mandate.

## Model Configuration
- **Selected Model**: Tuned Random Forest
- **Production Threshold**: 0.30
- **Rationale**: Prioritizes proactive customer retention by capturing ~90% of attrited customers while limiting false alarms to less than 19% of flagged accounts.
- **Training rows**: 10,127 (model export) / 8,101 (test evaluation split)
- **Test rows**: 2,026 (1,701 Existing, 325 Attrited)

## 5-Fold Stratified Cross-Validation (training set)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| Baseline RF | 0.9547 ± 0.0078 | 0.9235 ± 0.0166 | 0.7827 ± 0.0391 | 0.8470 ± 0.0286 | 0.9871 ± 0.0033 |
| Tuned RF | 0.9585 ± 0.0063 | 0.8963 ± 0.0216 | 0.8395 ± 0.0328 | 0.8666 ± 0.0216 | 0.9858 ± 0.0038 |
| Logistic Regression | 0.9042 ± 0.0028 | 0.7653 ± 0.0178 | 0.5837 ± 0.0214 | 0.6619 ± 0.0122 | 0.9246 ± 0.0075 |

## Held-Out Test Performance
| Model / Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | Specificity | TP | FP | TN | FN |
|-------------------|----------|-----------|--------|-----|---------|-------------|----|----|----|----|
| Tuned RF @ 0.30 (production) | 0.9516 | 0.8162 | 0.9015 | 0.8567 | 0.9844 | 0.9612 | 293 | 66 | 1635 | 32 |
| Tuned RF @ 0.42 (max F1) | 0.9576 | 0.8770 | 0.8554 | 0.8660 | 0.9844 | 0.9771 | 278 | 39 | 1662 | 47 |
| Baseline RF @ 0.50 | 0.9580 | 0.9348 | 0.7938 | 0.8586 | 0.9844 | 0.9894 | 258 | 18 | 1683 | 67 |
| Logistic Regression @ 0.50 | 0.8998 | 0.7675 | 0.5385 | 0.6329 | 0.9166 | 0.9688 | 175 | 53 | 1648 | 150 |

## Visualizations
- `model_comparison_bars.png` — metric-by-metric head-to-head comparison
- `roc_curves.png` — ROC curves with AUC (Tuned RF: 0.9844 vs Logistic Regression: 0.9166)
- `confusion_matrices.png` — side-by-side confusion matrices (Tuned RF @ 0.30 vs Logistic Regression @ 0.50)
- `cv_score_distribution.png` — 5-fold cross-validation F1 score stability boxplots

## Final Verdict
| Criterion | Winner | Notes |
|-----------|--------|-------|
| Overall F1 | Tuned Random Forest | Consistently dominates (0.8666 CV F1, 0.8567 Test F1 @ 0.30 vs 0.6329 for LR) |
| Recall (churn capture) | Tuned Random Forest (@0.30) | Captures 90.15% of churners (293/325) vs 53.85% (175/325) for Logistic Regression |
| ROC-AUC | Tuned Random Forest | Superior class discrimination (0.9844 vs 0.9166) |
| Cross-validation stability | Tuned Random Forest | Low variance across 5 folds (std ±0.0216 on F1) |
| Interpretability | Logistic Regression | Linear coefficients provide direct directional impact, but at a severe 36.3% recall penalty |

**Recommended for deployment: Tuned Random Forest at Threshold 0.30** — Provides the optimal balance for credit card churn intervention, capturing 9 out of 10 churners while preserving high precision (81.62%) and keeping retention team outreach costs highly efficient.

## Exported Artifacts
- `final_model.joblib` — fitted pipeline (preprocessor + classifier), trained on all 10,127 rows
- `final_model_config.json` — threshold, feature schema, version metadata
- Usage:
```python
import joblib, json, pandas as pd
model = joblib.load("final_model.joblib")
config = json.load(open("final_model_config.json"))
proba = model.predict_proba(new_customers_df)[:, 1]
churn_flag = (proba >= config["decision_threshold"]).astype(int)
```

## Caveats & Next Steps
- Data is a static public dataset; no temporal validation was performed, so quarterly drift monitoring is advised.
- Threshold 0.30 should be dynamically revisited once specific retention campaign cost and customer lifetime value (LTV) metrics are established.
- Monitor feature distributions and model drift especially on behavioral features (`Total_Trans_Amt`, `Total_Trans_Ct`, `Total_Ct_Chng_Q4_Q1`, `Total_Revolving_Bal`), which account for the majority of predictive power.
