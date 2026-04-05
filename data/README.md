# Dataset Layout

This folder stores benchmark datasets selected for the paper.

## Structure

- data/archives: Original downloaded zip files (immutable source copies)
- data/raw: Extracted working files grouped by dataset
- data/metadata: Dataset manifest and schema/feature metadata artifacts
- data/splits: Fixed stratified train/validation/test split index files
- data/processed: Reserved for cleaned and transformed modeling inputs

## Extracted Datasets

1. Heart Disease (UCI)

- Extracted folder: data/raw/heart_disease
- Common baseline file: data/raw/heart_disease/processed.cleveland.data

1. Adult Income (UCI)

- Extracted folder: data/raw/adult_income
- Training file: data/raw/adult_income/adult.data
- Test file: data/raw/adult_income/adult.test

1. Credit Default (UCI)

- Extracted folder: data/raw/credit_default
- Main file: data/raw/credit_default/default of credit card clients.xls

## Notes

- Keep files in data/archives unchanged for reproducibility.
- Use data/raw for ingestion and preprocessing.
- Use data/splits as the single source of split indices for all models.
- Use data/metadata/dataset_manifest.json for target and feature-type lookup.
- Any cleaned/engineered outputs should be saved under data/processed.
