# Credit Risk Model Governance Report

## 1. Model Purpose

The objective of this project is to estimate the probability that a credit card customer will default on their payment.

The model is intended to support credit-risk analysis and demonstrate how a challenger machine-learning model can be evaluated against a baseline statistical model.

---

## 2. Dataset

Dataset size before cleaning: 30,000 rows

Dataset size after removing duplicate rows: 29,965 rows

Target variable:

- default_flag

Target distribution:

- Non-default: 77.88%
- Default: 22.12%

No missing values were identified in the original dataset.

---

## 3. Data Quality Findings

The analysis identified:

- 70 rows belonging to duplicate groups
- 35 unique duplicate groups
- No duplicate groups with conflicting target values
- Invalid or unusual categorical codes requiring investigation
- Negative bill amounts
- Extremely high credit utilization
- Missing payment ratios caused by zero denominators
- Extreme payment-ratio outliers

Duplicate rows were investigated before modeling.

35 duplicate rows were removed, leaving 29,965 observations.

---

## 4. Feature Engineering

The following behavioral features were created:

- max_delinquency
- delinquent_months
- recent_delinquency
- avg_pay_status
- credit_utilization

Additional robustness features included:

- payment_ratio_missing
- payment_ratio_capped
- credit_utilization_capped
- high_utilization
- negative_bill_flag

Extreme ratios were capped using percentile-based limits rather than simply deleting observations.

---

## 5. Models Evaluated

### Logistic Regression Baseline

ROC-AUC: 0.7474

PR-AUC: 0.5157

The logistic regression model provides a simple statistical benchmark.

### Random Forest Challenger

ROC-AUC: 0.7745

PR-AUC: 0.5536

The Random Forest improved both ROC-AUC and PR-AUC relative to the baseline.

---

## 6. Cross-Validation

Five-fold cross-validation produced:

ROC-AUC: 0.7847 +/- 0.0085

PR-AUC: 0.5615 +/- 0.0118

F1: 0.5411 +/- 0.0101

Recall: 0.6196 +/- 0.0125

Precision: 0.4803 +/- 0.0108

The relatively small variation between folds provides evidence that model performance is reasonably stable across the validation folds.

---

## 7. Threshold Selection

The default classification threshold was evaluated against business costs.

The lowest modeled cost occurred at a threshold of:

0.55

At this threshold:

- False positives: 713
- False negatives: 584
- Total modeled cost: 2,465

However, the optimal threshold depends on the relative business cost of false positives and false negatives.

Sensitivity analysis showed that increasing the cost of missed defaults causes the optimal threshold to decrease.

---

## 8. Model Explainability

SHAP analysis identified the following major model drivers:

1. max_delinquency
2. delinquent_months
3. pay_0
4. recent_delinquency
5. avg_pay_status
6. credit_utilization_capped
7. limit_bal

These features primarily represent repayment behavior, delinquency history, and credit utilization.

---

## 9. Individual Customer Explainability

SHAP was also used to explain predictions for individual customers.

For example, customer 15,000 received an estimated default probability of 63.09%.

The strongest risk-increasing factors included:

- max_delinquency
- recent_delinquency
- delinquent_months
- pay_2
- credit_utilization_capped
- lack of recent payment

Individual explanations help analysts understand why a particular customer received a high-risk prediction.

---
---

## Probability Calibration

The Random Forest produced useful ranking performance but its raw probabilities were poorly calibrated.

The original model achieved a Brier Score of 0.1766.

After probability calibration, the Brier Score improved to 0.1361 while ROC-AUC remained approximately unchanged:

- Original ROC-AUC: 0.7770
- Calibrated ROC-AUC: 0.7766

The calibration curve showed that the original Random Forest systematically overestimated default probabilities in several probability ranges.

After calibration, predicted probabilities were substantially closer to observed default rates.

For example:

- Predicted 0.050 vs actual 0.042
- Predicted 0.104 vs actual 0.107
- Predicted 0.195 vs actual 0.192
- Predicted 0.281 vs actual 0.274
- Predicted 0.673 vs actual 0.700

This demonstrates that calibration improved the reliability of the model's probability estimates without materially changing its ranking performance.

This distinction is important because a credit-risk model may be used not only to rank customers by risk but also to estimate the probability of default.
---

## 11. Model Risks and Limitations

Potential model risks include:

- Dataset may not represent current portfolio behavior
- Historical relationships may change over time
- Extreme observations may affect model behavior
- Categorical codes require careful interpretation
- Random Forest predictions may not be perfectly calibrated probabilities
- Model performance may change under economic stress
- Correlation should not be interpreted as causation
- The dataset does not contain all information used in a real credit-risk environment

The model should therefore not be treated as a production-ready credit decision system.

---

## 12. Monitoring Recommendations

A production implementation should monitor:

### Data quality

- Missing values
- Invalid values
- Unexpected categorical codes
- Distribution changes

### Model performance

- ROC-AUC
- PR-AUC
- Recall
- Precision
- Calibration
- Default-rate stability

### Population stability

Feature distributions should be monitored over time to identify population drift.

### Business performance

The organization should monitor the financial cost associated with false positives and false negatives.

---

## 13. Governance Recommendation

The Random Forest challenger demonstrates improved discriminatory performance relative to the logistic regression baseline.

However, additional validation is required before production use, including:

- Probability calibration
- Out-of-time validation
- Stability testing
- Drift analysis
- Fairness analysis
- Stress testing
- Documentation of model assumptions
- Ongoing production monitoring

The model should therefore be considered a research/portfolio challenger model rather than a production credit decision model.