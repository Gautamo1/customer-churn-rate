## Experiment 001 — Random Forest Baseline

**Date:** 2026-08-09

### Objective

Establish a baseline Random Forest model for predicting customer churn.

### Model

- Algorithm: Random Forest
- Number of trees: 100
- Class weighting: `balanced`
- Random state: 42
- Train/test split: 80/20
- Stratification: Yes
- Classification threshold: 0.50

### Results

| Metric | Stayed | Churned |
|---|---:|---:|
| Precision | 0.86 | 0.45 |
| Recall | 0.92 | 0.30 |
| F1-score | 0.89 | 0.36 |

**Overall Accuracy:** 0.81

### Confusion Matrix

| | Predicted Stayed | Predicted Churned |
|---|---:|---:|
| Actual Stayed | 36,593 | 3,328 |
| Actual Churned | 6,166 | 2,671 |

### Observations

The model achieves 81% accuracy and performs well at identifying
customers who stay.

However, performance on the churn class is considerably weaker:

- Churn precision: 0.45
- Churn recall: 0.30
- Churn F1-score: 0.36

The model correctly identifies 2,671 of 8,837 churned customers and
misses 6,166 churned customers.

Class imbalance was explicitly addressed using
`class_weight='balanced'`. Therefore, the low churn recall cannot be
attributed to class imbalance alone.

### Hypotheses for Further Investigation

Possible reasons for the low churn recall include:

- Classification threshold of 0.50 may be too conservative.
- Random Forest may not be the best model for this dataset.
- Feature relationships may require a more powerful model.
- Class weighting may not be sufficient to optimize the desired
  precision/recall trade-off.

### Next Steps

1. Tune the classification threshold.
2. Evaluate XGBoost.
3. Investigate SMOTE as an alternative imbalance-handling strategy.
4. Compare models using churn recall, precision, F1-score and PR-AUC.

## Experiment 002 — Classification Threshold Tuning

### Objective

The Random Forest baseline achieved a churn recall of only 0.30
using the default classification threshold of 0.50.

The objective was to determine whether lowering the threshold could
improve the model's ability to identify customers at risk of churn.

### Results

| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.50 | 0.450 | 0.300 | 0.360 |
| 0.36 | 0.335 | 0.605 | 0.431 |
| 0.35 | 0.329 | 0.625 | 0.431 |
| 0.34 | 0.322 | 0.644 | 0.430 |
| 0.33 | 0.317 | 0.666 | 0.430 |
| 0.32 | 0.312 | 0.685 | 0.428 |
| 0.31 | 0.306 | 0.707 | 0.427 |
| 0.30 | 0.300 | 0.728 | 0.425 |

### Observations

Lowering the classification threshold substantially increases recall
for the churn class, but reduces precision.

At a threshold of 0.35, churn recall increases from 0.30 to 0.625,
while precision decreases from 0.45 to 0.329.

The F1-score also improves from 0.36 to approximately 0.431.

### Conclusion

The default threshold of 0.50 is too conservative for the objective
of identifying customers at risk of churn.

A threshold around 0.35 appears to provide a better balance between
churn recall and precision and will be considered as a candidate
threshold for the decision-support system.

### Important Limitation

The threshold was explored using the test data during experimentation.
For the final model, the threshold should be selected using validation
data and then evaluated once on the untouched test set.

### Next Step

Evaluate XGBoost and compare its performance against the Random Forest,
using churn recall, precision, F1-score, and PR-AUC as key metrics.


## Experiment 003 — XGBoost Threshold Tuning

### Objective

Determine whether adjusting the XGBoost classification threshold can
improve the precision-recall trade-off for churn prediction.

### Results

| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.55 | 0.348 | 0.607 | 0.443 |
| 0.54 | 0.343 | 0.624 | 0.442 |
| 0.53 | 0.339 | 0.641 | 0.442 |
| 0.52 | 0.332 | 0.655 | 0.441 |
| 0.51 | 0.328 | 0.668 | 0.439 |
| 0.50 | 0.328 | 0.668 | 0.439 |
| 0.49 | 0.323 | 0.682 | 0.439 |
| 0.48 | 0.319 | 0.696 | 0.437 |
| 0.47 | 0.315 | 0.713 | 0.437 |
| 0.46 | 0.305 | 0.737 | 0.431 |

### Observations

Lowering the threshold increases churn recall but decreases precision.
The highest F1-score in the evaluated range is approximately 0.443
at a threshold of 0.55.

### Conclusion

XGBoost provides a better balance between churn precision and recall
than the Random Forest baseline. A threshold around 0.55 is a strong
candidate operating point if F1 is prioritized.

A lower threshold such as 0.46 could be considered if the business
prioritizes identifying as many potential churners as possible.

### Next Step

Select the threshold using validation data and perform a final evaluation
on the untouched test set.


## Experiment 004 — Model Comparison

### Objective

Compare Random Forest and XGBoost to identify the most suitable model
for the customer churn decision-support system.

### Results

| Model | Threshold | Accuracy | Churn Precision | Churn Recall | Churn F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.35 | 0.805 | 0.445 | 0.302 | 0.360 | 0.738 |
| XGBoost | 0.55 | 0.722 | 0.348 | 0.609 | 0.443 | 0.750 |

### Conclusion

Random Forest achieved higher overall accuracy and churn precision.
However, its churn recall was only 30.2%, meaning that the majority
of customers who actually churned were not identified.

XGBoost achieved a substantially higher churn recall of 60.9% and a
higher churn F1-score of 0.443. It also achieved a slightly higher
ROC-AUC of 0.750.

Since the objective of the system is to identify customers at risk of
churn and support retention decisions, identifying potential churners
is more important than maximizing overall accuracy.

Therefore, XGBoost with a threshold of 0.55 is selected as the current
candidate model.

The lower precision of XGBoost will be considered when designing the
retention strategy layer.


## Experiment 005 — XGBoost RandomizedSearch (Recall@Top30%)

**Date:** 2026-08-12

### Notebook

- `notebooks/07_hyperparameter_tuningXGB.ipynb`

### Objective

Optimize recall for churners among the top 30% of predicted probabilities (business targeting constraint). Hyperparameter search uses a custom scorer that computes recall when the top 30% highest-probability customers are treated as positives.

### Best search results

- Best parameters:

```
{'subsample': 0.8, 'reg_lambda': 0.1, 'reg_alpha': 1, 'n_estimators': 300, 'min_child_weight': 5, 'max_depth': 4, 'learning_rate': 0.03, 'gamma': 0.3, 'colsample_bytree': 0.9}
```

- Best cross-validation score (recall@top30%): `0.5863064082614231`

### Test-set evaluation (threshold = 0.50 as in notebook)

Confusion matrix:

| | Predicted Stayed | Predicted Churned |
|---|---:|---:|
| Actual Stayed | 26,751 | 13,170 |
| Actual Churned | 2,657 | 6,180 |

Classification report:

| class | precision | recall | f1-score | support |
|---:|---:|---:|---:|---:|
| 0 | 0.91 | 0.67 | 0.77 | 39,921 |
| 1 | 0.32 | 0.70 | 0.44 | 8,837 |

Overall:

- Accuracy: 0.68
- Macro avg (precision, recall, f1): 0.61, 0.68, 0.61
- Weighted avg (precision, recall, f1): 0.80, 0.68, 0.71

ROC-AUC: `0.7527266898123568`

PR-AUC: `0.4063486644365407`

### Notes & next steps

- The model delivers substantially higher recall for the churn class at
  the evaluated threshold, but precision for churn is low (0.32). This
  matches the objective prioritizing recall within a top-30% outreach
  budget.
- Save `best_model` to `models/` with a timestamped filename for
  reproducibility and add the saved artifact path to this log entry.
- Consider validating the chosen operating point on a held-out fold or
  across multiple random splits to ensure stability of recall@top30%.

---

*Entry appended with supplied metrics.*