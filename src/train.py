"""Reproducible UCI Student Performance risk-classification benchmark."""
from pathlib import Path
from io import BytesIO
import json
import urllib.request
import zipfile

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay, f1_score,
                             precision_score, recall_score, roc_auc_score, RocCurveDisplay)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
REPORTS = ROOT / 'reports'
SEED = 42
URL = 'https://archive.ics.uci.edu/static/public/320/student+performance.zip'


def find_csv_in_zip(blob):
    """Locate student-mat.csv, including within nested ZIP archives."""
    with zipfile.ZipFile(BytesIO(blob)) as z:
        for name in z.namelist():
            if name.lower().endswith('student-mat.csv'):
                return z.read(name)
        for name in z.namelist():
            if name.lower().endswith('.zip'):
                try:
                    return find_csv_in_zip(z.read(name))
                except (FileNotFoundError, zipfile.BadZipFile):
                    pass
    raise FileNotFoundError('student-mat.csv not found in UCI archive')


def load_data():
    DATA.mkdir(exist_ok=True)
    csv_file = DATA / 'student-mat.csv'
    if not csv_file.exists():
        print('Downloading original dataset from UCI...')
        with urllib.request.urlopen(URL, timeout=45) as response:
            csv_file.write_bytes(find_csv_in_zip(response.read()))
    df = pd.read_csv(csv_file, sep=';')
    required = {'G1', 'G2', 'G3'}
    if not required.issubset(df.columns):
        raise ValueError(f'Missing required columns: {required - set(df.columns)}')
    return df


def build_pipeline(X, model):
    numeric = X.select_dtypes(include='number').columns.tolist()
    categorical = X.select_dtypes(exclude='number').columns.tolist()
    prep = ColumnTransformer([
        ('num', Pipeline([('impute', SimpleImputer(strategy='median')),
                          ('scale', StandardScaler())]), numeric),
        ('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                          ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical)
    ])
    return Pipeline([('prep', prep), ('model', model)])


def main():
    REPORTS.mkdir(exist_ok=True)
    df = load_data()
    y = (df['G3'] < 10).astype(int)
    print(f'Records: {len(df)} | At-risk: {int(y.sum())} ({y.mean():.1%})')
    # Identical row indices for both scenarios to make comparisons meaningful.
    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.2,
                                           stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    models = {
        'dummy': lambda: DummyClassifier(strategy='prior'),
        'logistic_regression': lambda: LogisticRegression(max_iter=3000, class_weight='balanced', random_state=SEED),
        'random_forest': lambda: RandomForestClassifier(n_estimators=250, min_samples_leaf=2, class_weight='balanced', random_state=SEED, n_jobs=-1),
        'gradient_boosting': lambda: HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, random_state=SEED)
    }
    rows = []
    best_by_scenario = {}
    for scenario, excluded in [('without_prior_grades', ['G1', 'G2', 'G3']),
                               ('with_prior_grades', ['G3'])]:
        X = df.drop(columns=excluded)
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        candidates = []
        for name, factory in models.items():
            pipe = build_pipeline(X_train, factory())
            scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='f1', n_jobs=1)
            candidates.append((name, float(scores.mean()), float(scores.std())))
            print(f'{scenario} | {name}: CV F1={scores.mean():.3f} +/- {scores.std():.3f}')
        # Select model ONLY using training-set cross-validation.
        selected_name, cv_mean, cv_std = max(candidates, key=lambda item: item[1])
        winner = build_pipeline(X_train, models[selected_name]())
        winner.fit(X_train, y_train)
        pred = winner.predict(X_test)
        prob = winner.predict_proba(X_test)[:, 1]
        row = {
            'scenario': scenario, 'selected_model': selected_name,
            'cv_f1_mean': cv_mean, 'cv_f1_std': cv_std,
            'test_precision': precision_score(y_test, pred, zero_division=0),
            'test_recall': recall_score(y_test, pred, zero_division=0),
            'test_f1': f1_score(y_test, pred, zero_division=0),
            'test_roc_auc': roc_auc_score(y_test, prob),
            'test_average_precision': average_precision_score(y_test, prob),
            'test_n': len(y_test)
        }
        rows.append(row)
        best_by_scenario[scenario] = row
        joblib.dump(winner, REPORTS / f'{scenario}_model.joblib')
        (REPORTS / f'{scenario}_classification_report.txt').write_text(
            classification_report(y_test, pred, target_names=['not_at_risk', 'at_risk'], zero_division=0))
        disp = ConfusionMatrixDisplay(confusion_matrix(y_test, pred),
                                      display_labels=['Not at risk', 'At risk'])
        disp.plot(cmap='Blues', values_format='d')
        plt.title(f'Confusion matrix: {scenario}')
        plt.tight_layout()
        plt.savefig(REPORTS / f'{scenario}_confusion_matrix.png', dpi=150)
        plt.close()
        RocCurveDisplay.from_predictions(y_test, prob)
        plt.title(f'ROC curve: {scenario}')
        plt.tight_layout()
        plt.savefig(REPORTS / f'{scenario}_roc_curve.png', dpi=150)
        plt.close()
        perm = permutation_importance(winner, X_test, y_test, scoring='f1',
                                      n_repeats=10, random_state=SEED)
        importance = pd.DataFrame({'feature': X.columns,
                                   'importance_mean': perm.importances_mean,
                                   'importance_std': perm.importances_std})
        importance.sort_values('importance_mean', ascending=False).to_csv(
            REPORTS / f'{scenario}_permutation_importance.csv', index=False)
    pd.DataFrame(rows).to_csv(REPORTS / 'metrics.csv', index=False)
    (REPORTS / 'run_metadata.json').write_text(json.dumps({
        'seed': SEED, 'test_size': 0.2, 'cv_folds': 5,
        'target': 'at_risk = (G3 < 10)', 'positive_label': 1,
        'n_records': len(df), 'n_at_risk': int(y.sum()),
        'model_selection': 'highest training 5-fold CV F1',
        'scenarios': best_by_scenario
    }, indent=2))
    print('\nSaved outputs in reports/. Review metrics.csv and figures.')


if __name__ == '__main__':
    main()
