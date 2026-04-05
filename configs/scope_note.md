# HyperFast Robustness Scope (Frozen)

## Main Claim

HyperFast provides very fast tabular inference, but robustness under realistic data corruption is conditional and must be measured explicitly.

## In-Scope Extension

- Noise robustness on continuous numeric features
- Missingness robustness with train-only imputer fitting
- Reduced-data robustness with stratified subsampling

## Out of Scope

- HyperFast architecture modification
- Reproducing full 70-dataset OpenML benchmark
- Novel model invention

## Primary Metric

- Balanced accuracy (primary)

## Secondary Metrics

- F1, precision, recall
- AUROC (only when probability outputs are available)
- Runtime: fit, predict, total

## Dataset Lock

- UCI Heart Disease
- UCI Adult Income
- UCI Default of Credit Card Clients

## Baseline Lock

- HyperFast (default and tuned operating points)
- Logistic Regression
- Random Forest
- Optional boosting baseline (only if setup remains stable)

## Fairness Rules

- Same splits for all models
- Same preprocessing pipeline logic for all models
- Split first, preprocessing fit on train only
- Validation-only tuning; no test tuning

## Completion Gate for Next Phase

Proceed to baseline runs only after these are present:

1. Pinned dependencies
2. Saved split files with fixed test set
3. Dataset manifest with paths/targets/features
