"""
train.py
--------
Full training pipeline:
  1. Load & engineer features
  2. Train/test split (stratified)
  3. SMOTE oversampling on train fold only
  4. Cross-validate three models
  5. Final fit + evaluation on held-out test set
  6. Save best model + preprocessor to models/

Usage:
    python train.py                      # uses data/churn_data.csv (auto-generates if missing)
    python train.py --data path/to.csv   # custom dataset
    python train.py --n-customers 10000  # regenerate with N rows
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from xgboost import XGBClassifier

# Local imports (resolve path regardless of where script is called from)
sys.path.insert(0, str(Path(__file__).parent))
from utils.preprocessing import build_preprocessor, load_and_prepare

DATA_PATH   = Path(__file__).parent / "data" / "churn_data.csv"
MODELS_DIR  = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ── Model definitions ─────────────────────────────────────────────────────────

def get_model_definitions() -> dict:
    return {
        "logistic_regression": LogisticRegression(
            C=0.5,
            max_iter=1000,
            class_weight="balanced",  # alternative to SMOTE alone
            solver="lbfgs",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=10,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        "xgboost": XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
    }


# ── Evaluation helpers ────────────────────────────────────────────────────────

def evaluate(clf, X_test, y_test, model_name: str) -> dict:
    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "model":     model_name,
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred,    zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred,        zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba),                   4),
    }

    cm = confusion_matrix(y_test, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    print(f"\n{'-'*50}")
    print(f"  {model_name.upper()}")
    print(f"{'-'*50}")
    print(classification_report(y_test, y_pred, target_names=["Stay", "Churn"]))
    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")

    return metrics


# ── Cross-validation ──────────────────────────────────────────────────────────

def cross_validate_model(
    model_name: str,
    clf,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor,
    n_splits: int = 5,
) -> dict:
    """
    5-fold stratified CV using an imbalanced-learn Pipeline so SMOTE is applied
    ONLY inside each training fold — preventing data leakage.
    """
    cv_pipeline = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote",        SMOTE(random_state=42, k_neighbors=5)),
        ("classifier",   clf),
    ])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    scoring = ["precision", "recall", "f1", "roc_auc"]
    cv_results = cross_validate(
        cv_pipeline,
        X_train, y_train,
        cv=skf,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )

    summary = {}
    for metric in scoring:
        key    = f"test_{metric}"
        scores = cv_results[key]
        summary[metric] = {
            "mean":  round(float(scores.mean()), 4),
            "std":   round(float(scores.std()),  4),
            "folds": [round(float(s), 4) for s in scores],
        }

    print(f"\n  {model_name} — {n_splits}-fold CV")
    for m, v in summary.items():
        print(f"    {m:12s}: {v['mean']:.4f} ± {v['std']:.4f}")

    return summary


# ── Main pipeline ─────────────────────────────────────────────────────────────

def train(data_path: Path = DATA_PATH, n_customers: int = 5000) -> dict:
    # 1. Generate data if missing
    if not data_path.exists():
        print(f"Dataset not found at {data_path}. Generating {n_customers:,} rows...")
        sys.path.insert(0, str(data_path.parent))
        from generate_data import generate_dataset
        df = generate_dataset(n_customers)
        df.to_csv(data_path, index=False)
        print(f"Saved to {data_path}")

    # 2. Load & feature engineer
    print(f"\nLoading dataset from {data_path}")
    X, y = load_and_prepare(str(data_path))
    print(f"Shape: {X.shape}  |  Churn rate: {y.mean()*100:.1f}%")

    # 3. Stratified train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    print(f"Train churn rate: {y_train.mean()*100:.1f}%  |  "
          f"Test churn rate:  {y_test.mean()*100:.1f}%")

    # 4. Build preprocessor (fit on train data only)
    preprocessor = build_preprocessor()
    X_train_pp = preprocessor.fit_transform(X_train)
    X_test_pp  = preprocessor.transform(X_test)
    print(f"\nFeatures after preprocessing: {X_train_pp.shape[1]}")

    # 5. Apply SMOTE to training set
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_res, y_train_res = smote.fit_resample(X_train_pp, y_train)
    print(f"\nAfter SMOTE:")
    print(f"  Class 0 (Stay):  {(y_train_res == 0).sum():,}")
    print(f"  Class 1 (Churn): {(y_train_res == 1).sum():,}")

    # 6. Cross-validate each model
    print("\n" + "="*50)
    print("  CROSS-VALIDATION (5-fold stratified)")
    print("="*50)

    model_defs = get_model_definitions()
    cv_results_all = {}

    for name, clf in model_defs.items():
        cv_results_all[name] = cross_validate_model(
            name, clf, X_train, y_train, build_preprocessor()
        )

    # 7. Final training on full resampled train set
    print("\n" + "="*50)
    print("  FINAL TRAINING + TEST EVALUATION")
    print("="*50)

    final_results = []
    trained_models = {}

    for name, clf in model_defs.items():
        print(f"\nFitting {name}...")
        clf.fit(X_train_res, y_train_res)
        metrics = evaluate(clf, X_test_pp, y_test, name)
        metrics["cv"] = cv_results_all[name]
        final_results.append(metrics)
        trained_models[name] = clf

    # 8. Pick best model by F1 score
    best = max(final_results, key=lambda r: r["f1"])
    print(f"\n{'='*50}")
    print(f"  Best model: {best['model']}  (F1={best['f1']:.4f})")
    print(f"{'='*50}")

    # 9. Persist artefacts
    joblib.dump(preprocessor,                  MODELS_DIR / "preprocessor.joblib")
    joblib.dump(trained_models["logistic_regression"], MODELS_DIR / "lr_model.joblib")
    joblib.dump(trained_models["random_forest"],       MODELS_DIR / "rf_model.joblib")
    joblib.dump(trained_models["xgboost"],             MODELS_DIR / "xgb_model.joblib")

    results_payload = {
        "models":     final_results,
        "best_model": best["model"],
        "feature_names": list(X_train.columns),
    }
    with open(MODELS_DIR / "results.json", "w") as f:
        json.dump(results_payload, f, indent=2)

    print(f"\nArtefacts saved to {MODELS_DIR}/")
    return results_payload


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train churn prediction models")
    parser.add_argument("--data",        type=str, default=str(DATA_PATH))
    parser.add_argument("--n-customers", type=int, default=5000)
    args = parser.parse_args()
    train(Path(args.data), args.n_customers)
