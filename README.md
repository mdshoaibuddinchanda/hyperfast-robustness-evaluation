# HyperFast Robustness Benchmark (Tabular Classification)

A reproducible evaluation of training-free tabular models (HyperFast) under real-world conditions including noise, missing data, and limited training regimes.

This repository runs a reproducible robustness benchmark for tabular binary classification using:

- HyperFast
- Logistic Regression
- Random Forest

Robustness is evaluated under:

- Additive Gaussian noise
- MCAR missingness
- Reduced training data fractions

Primary metric: balanced accuracy.

## Reproducibility Policy

To keep this repository lightweight and reproducible:

- Do not version-control datasets
- Do not version-control split files
- Do not version-control generated results, plots, reports, or logs
- Do not version-control large checkpoints
- Keep `PROGRESS.md` locally only

All of these are excluded in `.gitignore`.

## Project Structure

- `src/`: experiment and analysis scripts
- `configs/`: frozen experiment configuration
- `data/README.md`: required dataset file layout
- `requirements.txt`: Python dependencies
- `environment.yml`: optional environment descriptor

## Environment Setup

Use conda for environment management and install Python packages from `requirements.txt`.

```powershell
conda create -n hyperfast-robustness python=3.12 -y
conda activate hyperfast-robustness
uv pip install -r requirements.txt
```

## Data Setup (Local Only)

Place dataset files locally under `data/raw/`:

- Heart Disease: `data/raw/heart_disease/processed.cleveland.data`
- Adult Income: `data/raw/adult_income/adult.data`
- Credit Default: `data/raw/credit_default/default of credit card clients.xls`

Reference dataset links:

- <https://archive.ics.uci.edu/dataset/45/heart+disease>
- <https://archive.ics.uci.edu/ml/datasets/Adult>
- <https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients>

## Commands (End-to-End)

1) Download HyperFast checkpoint (local file `hyperfast.ckpt`):

```powershell
python src/download_hyperfast_checkpoint.py
```

1) Generate fixed train/val/test splits:

```powershell
python src/generate_splits.py
```

1) Run full benchmark:

```powershell
python src/run_full_comparison.py
```

Optional quick smoke run:

```powershell
python src/run_full_comparison.py --datasets heart_disease --seeds 1
```

1) Generate analysis artifacts:

```powershell
python src/generate_analysis_artifacts.py
```

1) Verify lineage from raw run outputs to tables/plots:

```powershell
python src/verify_artifact_lineage.py
```

## What Gets Generated (Local)

- `runs/`
- `results/`
- `plots/`
- `report/`
- `error_analysis/`
- `logs/`

These are intentionally ignored in git.

## License

This project is licensed under the Non-Commercial Research and Private Use
License (NCRPU-1.0). See `LICENSE` for full terms.

In short:

- Allowed: private use, education, and non-commercial research
- Not allowed: any commercial use
- Warranty: provided "AS IS" with no guarantees or liability

## Citation

### HyperFast (software used)

Package metadata used in this project:

- Name: HyperFast
- Version: 1.0.6
- Summary: HyperFast: Instant Classification for Tabular Data
- URL: <https://github.com/AI-sandbox/hyperfast>

Suggested software citation (BibTeX):

```bibtex
@software{hyperfast_2026,
  title   = {HyperFast: Instant Classification for Tabular Data},
  author  = {AI-sandbox},
  year    = {2026},
  version = {1.0.6},
  url     = {https://github.com/AI-sandbox/hyperfast}
}
```

### Dataset citations

- UCI Machine Learning Repository: Dua, D. and Graff, C. (2019).
- Heart Disease (Cleveland): UCI dataset page above.
- Adult Income (Census Income): UCI Adult dataset page above.
- Default of Credit Card Clients: Yeh, I.-C., and Lien, C.-H. (2009), Expert Systems with Applications.

If your venue requires formal BibTeX for each dataset, add those exact records in your paper repository.
