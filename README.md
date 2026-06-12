# Customer Churn Prediction System

End-to-end ML pipeline predicting customer churn, with a Streamlit dashboard.

## Project structure

```
churn_prediction/
├── data/
│   └── generate_data.py     # Synthetic telecom dataset generator
├── models/                  # Auto-created by train.py
│   ├── preprocessor.joblib
│   ├── lr_model.joblib
│   ├── rf_model.joblib
│   ├── xgb_model.joblib
│   └── results.json
├── utils/
│   └── preprocessing.py     # Feature engineering + ColumnTransformer
├── train.py                 # Full training pipeline
├── app.py                   # Streamlit dashboard
└── requirements.txt
```

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

## Usage

### Step 1 — Train models

```bash
python train.py
```

This will:
- Auto-generate `data/churn_data.csv` (5,000 synthetic telecom customers)
- Engineer features (num_services, charge_per_month_ratio, tenure_bucket)
- Apply a stratified 80/20 train/test split
- Apply SMOTE oversampling to the training fold only (no leakage)
- Run 5-fold stratified cross-validation for all three models
- Train final models on the full resampled train set
- Evaluate on the held-out test set (Precision, Recall, F1, ROC-AUC)
- Save all artefacts to `models/`

Optional flags:
```bash
python train.py --n-customers 10000    # larger dataset
python train.py --data path/to/my.csv  # your own dataset
```

### Step 2 — Launch the dashboard

```bash
streamlit run app.py
```

Dashboard pages:
1. **Overview** — KPIs, churn by contract/tenure/charge
2. **Model Comparison** — CV results, confusion matrix, radar chart
3. **Predict Customer** — real-time churn probability per model
4. **Feature Insights** — importances + business recommendations

## Using your own dataset

Your CSV needs these columns (or a subset — the pipeline handles missings):

| Column | Type | Description |
|--------|------|-------------|
| `tenure_months` | int | Months as a customer |
| `monthly_charges` | float | Monthly billing amount |
| `total_charges` | float | Lifetime billing |
| `support_calls` | int | Support contacts in last 6 months |
| `contract` | int | 0=M-t-M, 1=One year, 2=Two year |
| `internet_service` | int | 0=None, 1=DSL, 2=Fiber |
| `payment_method` | int | 0–3 (electronic/mailed/bank/card) |
| `senior_citizen` | 0/1 | |
| `partner` | 0/1 | |
| `dependents` | 0/1 | |
| `phone_service` | 0/1 | |
| `multiple_lines` | 0/1 | |
| `online_security` | 0/1 | |
| `online_backup` | 0/1 | |
| `device_protection` | 0/1 | |
| `tech_support` | 0/1 | |
| `streaming_tv` | 0/1 | |
| `streaming_movies` | 0/1 | |
| `paperless_billing` | 0/1 | |
| `churn` | 0/1 | **Target variable** |

## Key design decisions

### SMOTE placement (critical)
SMOTE is applied **inside** the cross-validation loop using `imblearn.Pipeline`,
not before splitting. This prevents data leakage from synthetic samples appearing
in both train and validation folds.

```python
cv_pipeline = ImbPipeline([
    ("preprocessor", preprocessor),
    ("smote",        SMOTE(random_state=42)),
    ("classifier",   clf),
])
```

### Class imbalance strategy
Two complementary techniques are combined:
- SMOTE oversampling on the training set
- `class_weight="balanced"` in Logistic Regression and Random Forest

### Evaluation metric choice
F1 score is the primary ranking metric because churn is imbalanced and the cost
of false negatives (missed churners) is higher than false positives (unnecessary
retention calls). ROC-AUC is reported as a secondary ranking-quality measure.
