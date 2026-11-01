# SECOM Fault Prediction

Predicting pass/fail outcomes of semiconductor production units from sensor readings, with a focus on class imbalance, missing data, and honest evaluation.

## Problem

A semiconductor fab records hundreds of sensor signals while each unit moves through production, and every unit is later tested as pass or fail. If failures could be predicted from the sensor data, a fab could catch doomed units earlier and save processing time and cost.

This project uses the public **SECOM** dataset to test how well that works, and what makes it hard.

## Dataset

[SECOM, UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/179/secom)

- 1,567 production units, each with 590 sensor measurements and a pass/fail label
- Only 104 failures (about 6.6%), so the classes are heavily imbalanced
- Many missing values, and many sensors that are constant or mostly empty
- Each unit has a timestamp, so the data is ordered in time

To run the notebooks, download `secom.data` and `secom_labels.data` from the link above and place them in a `data/` folder (not included in this repo).

## Approach

1. **Explored the data** (class balance, missing values, constant columns).
2. **Split by time**, not randomly: the earliest 80% of units for training (1,253 units) and the latest 20% for testing (314 units, 17 failures). A real fab predicts the future from the past, and a random split would let the model see data from the same period as the test rows.
3. **Preprocessed using training data only**: removed constant columns and columns with more than 50% missing values (590 → 444 sensors), filled the remaining gaps with the median, and scaled the features.
4. **Trained a naive baseline** (logistic regression) to see the imbalance problem directly.
5. **Tried imbalance fixes**: class weights and SMOTE.
6. **Compared models with time-ordered cross-validation** on the training data only (`TimeSeriesSplit`, 5 folds), using ROC-AUC and PR-AUC. Feature selection and imputation were done inside each fold to avoid leakage.
7. **Ran diagnostics**: per-fold failure counts, a drift test (early vs late data), a rolling training window, and extra models.
8. **Evaluated the chosen model once** on the held-out test set.

## Why accuracy is not used

94.6% of the test units pass, so a model that always predicts "pass" scores 94.6% accuracy while catching zero failures. This project reports recall, precision, F1, ROC-AUC and PR-AUC instead. PR-AUC for random guessing equals the failure rate (about 5 to 7%).

## Results

### Test set (314 units, 17 failures)

| Model | Failures caught | False alarms | Recall | Precision | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Always predict "pass" (reference) | 0 / 17 | 0 | 0.00 | n/a | 0.50 | about 0.05 |
| Baseline: logistic regression, all features | 3 / 17 | 7 | 0.18 | 0.30 | 0.758 | 0.181 |
| Final: logistic regression, top 50 features, class-weighted | 6 / 17 | 58 | 0.35 | 0.09 | 0.598 | 0.135 |

The final model's 95% bootstrap intervals on the test set are ROC-AUC [0.449, 0.747] and PR-AUC [0.052, 0.327].

### Cross-validation on the training period (time-ordered, 5 folds)

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Logistic regression, all features | 0.514 | 0.075 |
| Logistic regression, top 20 features | 0.570 | 0.110 |
| Logistic regression, top 50 features | 0.569 | 0.118 |
| Logistic regression, L1 | 0.551 | 0.090 |
| Random forest | 0.597 | 0.119 |
| Extra trees | 0.592 | 0.100 |

Differences between most of these rows are within the fold-to-fold noise (about ±0.05 to ±0.13 AUC).

## Key findings

- **The signal is weak.** Cross-validated AUC stayed around 0.5 to 0.6 for every model I tried.
- **Resampling barely helped.** Class weights and SMOTE changed how many failures were caught by one or two units, while increasing false alarms.
- **Using fewer sensors beat using all of them** in cross-validation (about 0.57 vs 0.51 ROC-AUC for logistic regression).
- **The process changed over time.** The failure rate was about 13% in the first 400 units and about 3 to 4% afterwards, and 281 of the 444 sensors showed a statistically significant shift between the first and second half of the training data.
- **Training only on recent data did not help** (rolling windows of 400 and 600 units).
- **On the test set, the simple baseline ranked failures better than the final model** (ROC-AUC 0.758 vs 0.598). I did not switch models after seeing this, because the final model was chosen before looking at the test set.

## Limitations

- The test set has only **17 failures**, so every test metric is noisy. "Caught 6 of 17" and "caught 3 of 17" are hard to distinguish statistically.
- The baseline-vs-final comparison on the test set is not conclusive for the same reason. Possible explanations (small-sample noise, process drift between periods, the cutoff effect of class weights) were not tested.
- The imbalance comparison in notebook 04 was evaluated on the test set. I stopped doing that afterwards and used cross-validation on the training data for all later model choices.
- The baseline configuration (C=1, unweighted) was not included in the cross-validation comparison.
- Gradient boosting was not evaluated (the run was too slow in my setup).
- Hyperparameters were only lightly explored, and median imputation was the only imputation method tried.


## Tools

Python, pandas, NumPy, scikit-learn, imbalanced-learn, SciPy, Jupyter.


