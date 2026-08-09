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