# Student Performance Prediction and Early Warning Benchmark

Reproducible binary classification project using the UCI Student Performance **Mathematics** dataset (395 records). Target: `at_risk = (G3 < 10)`.

## Research questions
1. Can final-grade risk be predicted using available student attributes?
2. How does performance change when G1 and G2 grades are included?
3. Which features most affect held-out F1 performance?

## Setup
Python 3.10+ recommended.

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
python src/train.py
```

The script downloads `student-mat.csv` from the [UCI Student Performance dataset](https://archive.ics.uci.edu/dataset/320/student+performance) on first run. Alternatively, download the original CSV manually and place it in `data/student-mat.csv`.

## Method
- Positive label `1`: final grade G3 below 10 (0–20 scale).
- Identical 80/20 stratified train/test split across both scenarios (random seed 42).
- Five-fold stratified CV **on training data only** to choose the highest-F1 model from Dummy, Logistic Regression, Random Forest, and Histogram Gradient Boosting.
- Numeric imputation and standardization; categorical imputation and one-hot encoding, all inside a pipeline fitted separately per fold.
- One untouched test set for each selected model; report precision, recall, F1, ROC-AUC, average precision, confusion matrix.
- Permutation importance on held-out test data, measured as F1 decrease. Exploratory only; not causal evidence.

## Files
- `src/train.py`: end-to-end pipeline.
- `reports/metrics.csv`: model-selection and held-out test metrics after running.
- `reports/*_classification_report.txt`: class-wise results.
- `reports/*_confusion_matrix.png`, `*_roc_curve.png`: figures.
- `reports/*_permutation_importance.csv`: exploratory feature importance.
- `reports/run_metadata.json`: configuration and sample counts.

## Limitations
- Small observational dataset from two Portuguese secondary schools; not representative of university students or other education systems.
- `without_prior_grades` does **not** mean all predictors were available on day one; `absences` and other variables require availability review.
- G1 and G2 are earlier grades, not future leakage if available at prediction time; they are inappropriate for an early-term prediction.
- One test split has substantial sampling uncertainty; model selection and feature exploration may overfit.
- Classifier predicts final-grade threshold, **not** dropout, need for intervention, or causal effects.
- Sensitive/social attributes require ethical review and fairness analysis before operational use.
- No external validation, deployment, or intervention-effect evaluation is provided.

## Citation
Cortez, P. and Silva, A. (2008). *Using Data Mining to Predict Secondary School Student Performance*. Proceedings of FUBUTEC 2008. Dataset hosted by UCI Machine Learning Repository.

## License
Code: MIT (see LICENSE). Dataset remains subject to UCI's dataset terms and attribution requirements.
