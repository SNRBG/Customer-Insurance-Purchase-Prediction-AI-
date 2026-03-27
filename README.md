# Customer Insurance Purchase Prediction

A Python project for predicting whether customers will purchase insurance based on demographic and behavioral features.

## ?? Project Overview

- Dataset: `data/Social_Network_Ads.csv`
- Modules:
  - `src/data_preprocessing.py` — clean, split, and encode features
  - `src/model_training.py` — train and save machine learning model
  - `src/prediction.py` — load model and predict new customer outcomes
  - `src/evaluation.py` — calculate metrics (accuracy, precision, recall, F1)
  - `src/visualization.py` — charts and model performance plots
- Entry point: `main.py`

## ?? Setup Instructions

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## ?? Run the whole pipeline

```powershell
python main.py
```

## ?? Recommended workflow

1. Confirm dataset available: `data/Social_Network_Ads.csv`
2. Inspect and adjust settings in `main.py`
3. Run pipeline output results and plots
4. Evaluate metrics with `src/evaluation.py`

## ?? GitHub updates

1. Commit changes:

```powershell
git add README.md
git commit -m "Update README with project details"
```

2. Push to remote:

```powershell
git push
```
```

## ?? Contribution

- Fork repo and open PR
- Add tests and update docs
- Use `git branch` for feature work

## ?? Cleanup (optional)

```powershell
git rm --cached src/__pycache__/*.pyc
git commit -m "Remove cached bytecode files"
git push
```
