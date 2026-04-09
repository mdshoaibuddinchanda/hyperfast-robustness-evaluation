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

1. Banknote Authentication (UCI)

- Extracted folder: data/raw/banknote_authentication
- Main file: data/raw/banknote_authentication/data_banknote_authentication.txt

1. Breast Cancer Wisconsin (Diagnostic, UCI)

- Extracted folder: data/raw/breast_cancer_wisconsin_diagnostic
- Main file: data/raw/breast_cancer_wisconsin_diagnostic/wdbc.data

1. Haberman Survival (UCI)

- Extracted folder: data/raw/haberman_survival
- Main file: data/raw/haberman_survival/haberman.data

1. Ionosphere (UCI)

- Extracted folder: data/raw/ionosphere
- Main file: data/raw/ionosphere/ionosphere.data

1. Mushroom (UCI)

- Extracted folder: data/raw/mushroom
- Main file: data/raw/mushroom/agaricus-lepiota.data

1. Pima Diabetes (UCI)

- Extracted folder: data/raw/pima_diabetes
- Main file: data/raw/pima_diabetes/pima-indians-diabetes.data

1. Sonar Mines vs Rocks (UCI)

- Extracted folder: data/raw/sonar_mines_rocks
- Main file: data/raw/sonar_mines_rocks/sonar.all-data

## Notes

- Keep files in data/archives unchanged for reproducibility.
- Use data/raw for ingestion and preprocessing.
- Use data/splits as the single source of split indices for all models.
- Use data/metadata/dataset_manifest.json for target and feature-type lookup.
- Any cleaned/engineered outputs should be saved under data/processed.
