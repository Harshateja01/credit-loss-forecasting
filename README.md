# Credit Risk Forecasting

An end-to-end machine learning project for predicting credit-card payment default risk.

The project covers data quality analysis, feature engineering, baseline and challenger modeling, probability calibration, business-cost-based threshold selection, SHAP explainability, model governance, and deployment through an interactive Streamlit application.

## Live Application

The model is deployed as an interactive Streamlit credit-risk assessment dashboard.

**Live Demo:** https://credit-loss-forecasting-ejlaywhcwbrdseokrpxb2v.streamlit.app/

> This project is intended for research and portfolio demonstration purposes and is not a production credit-decision system.

---

## Business Problem

Credit-risk models help financial institutions estimate the likelihood that a borrower will fail to meet payment obligations.

The objective of this project is to estimate the probability of credit-card payment default and demonstrate how a machine-learning challenger model can be evaluated against a statistical baseline.

The project also considers an important practical question:

**At what probability threshold should a customer be classified as high risk when false positives and missed defaults have different business costs?**

---

## Dataset

The original dataset contains:

- 30,000 observations
- 23 predictor variables
- Binary default target
- Default rate: 22.12%

After duplicate analysis and cleaning:

- 29,965 observations remained
- 35 duplicate observations were removed
- No conflicting targets were found among duplicate groups

The variables contain information about:

- Credit limits
- Customer demographics
- Repayment status
- Monthly bill amounts
- Monthly payment amounts
- Historical delinquency

---

## Data Quality Analysis

The project investigates several data-quality issues, including:

- Duplicate observations
- Unusual categorical codes
- Negative bill balances
- Extreme credit utilization
- Zero-denominator payment ratios
- Extreme payment-ratio values

Rather than automatically deleting unusual financial observations, potentially meaningful values were investigated and robust features were created where appropriate.

---

## Feature Engineering

Behavioral risk features include:

- `max_delinquency`
- `delinquent_months`
- `recent_delinquency`
- `avg_pay_status`
- `credit_utilization`

Additional robustness features include:

- `payment_ratio_missing`
- `payment_ratio_capped`
- `credit_utilization_capped`
- `high_utilization`
- `negative_bill_flag`

Extreme ratios were capped using percentile-based limits instead of removing the observations.

---

## Model Development

Two primary models were evaluated.

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| Logistic Regression | 0.7474 | 0.5157 |
| Random Forest | 0.7770 | 0.5558 |
| Calibrated Random Forest | 0.7766 | 0.5560 |

The Random Forest challenger improved discrimination relative to the Logistic Regression baseline.

---

## Cross-Validation

Five-fold cross-validation produced:

| Metric | Mean |
|---|---:|
| ROC-AUC | 0.7847 ± 0.0085 |
| PR-AUC | 0.5615 ± 0.0118 |
| F1 | 0.5411 ± 0.0101 |
| Recall | 0.6196 ± 0.0125 |
| Precision | 0.4803 ± 0.0108 |

The relatively small variation between folds suggests reasonably stable validation performance.

---

## Probability Calibration

The original Random Forest produced:

- ROC-AUC: **0.7770**
- PR-AUC: **0.5558**
- Brier Score: **0.1766**

After probability calibration:

- ROC-AUC: **0.7766**
- PR-AUC: **0.5560**
- Brier Score: **0.1361**

Calibration substantially improved the Brier score while maintaining similar discriminatory performance.

---

## Business Threshold Analysis

Classification thresholds were evaluated using asymmetric error costs.

For the primary scenario:

- False-positive cost = 1
- False-negative cost = 3

The lowest modeled cost occurred at:

**Threshold = 0.55**

At this threshold:

- False positives: **713**
- False negatives: **584**
- Total modeled cost: **2,465**

Sensitivity analysis demonstrated that the optimal threshold changes as the assumed cost of missed defaults changes.

---

## Model Explainability

SHAP was used to understand the Random Forest's predictions.

Important global model drivers included:

1. `max_delinquency`
2. `delinquent_months`
3. `pay_0`
4. `recent_delinquency`
5. `avg_pay_status`
6. `credit_utilization_capped`
7. `limit_bal`

The strongest predictors are therefore primarily associated with repayment behavior, delinquency history, and credit utilization.

Individual customer explanations are also available.

---

## Streamlit Risk Assessment Application

The deployed application allows a user to select a customer and view:

- Predicted probability of default
- Risk category
- Decision threshold
- Recommended monitoring decision
- Customer financial information
- Credit utilization
- Delinquency information
- Risk signals
- SHAP-based model explanations
- Model performance information
- Governance disclaimer

The production artifact used by the application is a persisted calibrated Random Forest model.

---

## Project Structure

```text
credit-loss-forecasting/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── calibrated_random_forest.pkl
│
├── reports/
│   └── model_governance.md
│
├── src/
│   ├── app.py
│   ├── clean_data.py
│   ├── create_features.py
│   ├── create_robust_features.py
│   ├── train_baseline.py
│   ├── train_challenger.py
│   ├── validate_model.py
│   ├── evaluate_thresholds.py
│   ├── evaluate_business_cost.py
│   ├── evaluate_cost_sensitivity.py
│   ├── calibrate_model.py
│   ├── explain_model.py
│   ├── explain_customer_risk.py
│   └── customer_risk_assessment.py
│
├── requirements.txt
└── README.md
```

---

## Running Locally

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

Start the Streamlit application:

```bash
python -m streamlit run src/app.py
```

Then open the local address displayed by Streamlit, normally:

```text
http://localhost:8501
```

---

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- Random Forest
- Logistic Regression
- SHAP
- Streamlit
- Git
- GitHub
- Git LFS

---

## Model Governance

This model should be treated as a **research / portfolio challenger model**, not a production credit-decision system.

Before production use, additional work would be required, including:

- Out-of-time validation
- Population stability monitoring
- Data drift monitoring
- Model performance monitoring
- Fairness analysis
- Stress testing
- Regulatory review
- Model assumption documentation
- Production controls and monitoring

See `reports/model_governance.md` for additional governance documentation.

---

## Key Takeaway

The project demonstrates that improving a credit-risk model involves more than maximizing predictive accuracy.

A usable risk-modeling workflow also requires:

**data quality → feature engineering → validation → calibration → business threshold selection → explainability → governance → deployment**