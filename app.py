"""
app.py  —  Streamlit Churn Prediction Dashboard
------------------------------------------------
Run:  streamlit run app.py

Pages
  1. Overview    — dataset stats, class balance, churn by cohort
  2. Model comparison — CV scores, confusion matrices, ROC curves
  3. Predict     — real-time single-customer churn probability
  4. Insights    — feature importances, business recommendations

Requires models/ artefacts to exist.  Run `python train.py` first.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from utils.preprocessing import engineer_features, NUMERIC_FEATURES, CATEGORICAL_FEATURES, BINARY_FEATURES

MODELS_DIR = ROOT / "models"
DATA_PATH  = ROOT / "data" / "churn_data.csv"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-container { background: #f8f9fa; border-radius: 8px; padding: 16px; text-align: center; }
    .metric-value { font-size: 2rem; font-weight: 600; margin: 0; }
    .metric-label { font-size: 0.8rem; color: #6c757d; text-transform: uppercase; letter-spacing: 0.05em; }
    .risk-high   { color: #dc3545; font-weight: 600; }
    .risk-medium { color: #fd7e14; font-weight: 600; }
    .risk-low    { color: #198754; font-weight: 600; }
    [data-testid="stMetricValue"] { font-size: 1.8rem; }
</style>
""", unsafe_allow_html=True)


# ── Loaders (cached) ──────────────────────────────────────────────────────────

@st.cache_resource
def load_artefacts():
    """Load preprocessor + all three trained models."""
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
    models = {
        "Logistic Regression": joblib.load(MODELS_DIR / "lr_model.joblib"),
        "Random Forest":       joblib.load(MODELS_DIR / "rf_model.joblib"),
        "XGBoost":             joblib.load(MODELS_DIR / "xgb_model.joblib"),
    }
    with open(MODELS_DIR / "results.json") as f:
        results = json.load(f)
    return preprocessor, models, results


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return engineer_features(df)


def check_artefacts_exist() -> bool:
    required = [
        MODELS_DIR / "preprocessor.joblib",
        MODELS_DIR / "lr_model.joblib",
        MODELS_DIR / "rf_model.joblib",
        MODELS_DIR / "xgb_model.joblib",
        MODELS_DIR / "results.json",
    ]
    return all(p.exists() for p in required)


# ── Sidebar navigation ────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📉 Churn Prediction")
    st.divider()
    page = st.radio(
        "Navigate",
        ["🏠 Overview", "🤖 Model Comparison", "🎯 Predict Customer", "💡 Feature Insights"],
        label_visibility="collapsed",
    )
    st.divider()

    if not check_artefacts_exist():
        st.error("Models not found. Run `python train.py` first.")
        st.stop()

    preprocessor, models, results = load_artefacts()
    df = load_data()

    st.caption(f"Dataset: {len(df):,} customers")
    st.caption(f"Churn rate: {df['churn'].mean()*100:.1f}%")
    st.caption(f"Best model: {results['best_model'].replace('_', ' ').title()}")


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("Customer Churn Overview")
    st.markdown("Dataset-level statistics and cohort-level churn patterns.")
    st.divider()

    # ── Top KPIs ──────────────────────────────────────────────────────────────
    total      = len(df)
    churned    = df["churn"].sum()
    retained   = total - churned
    churn_rate = churned / total * 100
    avg_tenure = df.groupby("churn")["tenure_months"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total customers",  f"{total:,}")
    c2.metric("Churned",          f"{churned:,}",   f"{churn_rate:.1f}%")
    c3.metric("Retained",         f"{retained:,}",  f"{100-churn_rate:.1f}%")
    c4.metric("Avg tenure (churned)", f"{avg_tenure[1]:.0f} mo",
              f"vs {avg_tenure[0]:.0f} mo (retained)")

    st.divider()

    col_a, col_b = st.columns(2)

    # Class distribution
    with col_a:
        st.subheader("Class distribution")
        fig = px.pie(
            values=[retained, churned],
            names=["Retained", "Churned"],
            color_discrete_sequence=["#2563eb", "#dc2626"],
            hole=0.55,
        )
        fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=280)
        st.plotly_chart(fig, width='stretch')

    # Churn by contract type
    with col_b:
        st.subheader("Churn rate by contract type")
        contract_map = {0: "Month-to-month", 1: "One year", 2: "Two year"}
        df["contract_label"] = df["contract"].map(contract_map)
        cr_contract = (
            df.groupby("contract_label")["churn"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "churn_rate", "count": "n"})
        )
        cr_contract["churn_rate_pct"] = (cr_contract["churn_rate"] * 100).round(1)
        fig2 = px.bar(
            cr_contract,
            x="contract_label",
            y="churn_rate_pct",
            text="churn_rate_pct",
            color="churn_rate_pct",
            color_continuous_scale="RdYlGn_r",
        )
        fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig2.update_layout(
            yaxis_title="Churn rate (%)",
            xaxis_title="",
            coloraxis_showscale=False,
            margin=dict(t=20, b=20),
            height=280,
        )
        st.plotly_chart(fig2, width='stretch')

    col_c, col_d = st.columns(2)

    # Churn rate by tenure band
    with col_c:
        st.subheader("Churn rate by tenure")
        bins   = [0, 12, 24, 36, 48, 60, 72]
        labels = ["0–12", "13–24", "25–36", "37–48", "49–60", "61–72"]
        df["tenure_band"] = pd.cut(df["tenure_months"], bins=bins, labels=labels)
        cr_tenure = (
            df.groupby("tenure_band", observed=True)["churn"]
            .mean()
            .reset_index()
        )
        cr_tenure["churn_pct"] = (cr_tenure["churn"] * 100).round(1)
        fig3 = px.line(
            cr_tenure, x="tenure_band", y="churn_pct",
            markers=True,
            labels={"tenure_band": "Tenure (months)", "churn_pct": "Churn rate (%)"},
        )
        fig3.update_traces(line_color="#2563eb", marker_size=8)
        fig3.update_layout(margin=dict(t=20, b=20), height=260)
        st.plotly_chart(fig3, width='stretch')

    # Monthly charges distribution by churn
    with col_d:
        st.subheader("Monthly charges distribution")
        fig4 = px.histogram(
            df, x="monthly_charges", color="churn",
            barmode="overlay",
            nbins=40,
            opacity=0.7,
            color_discrete_map={0: "#2563eb", 1: "#dc2626"},
            labels={"churn": "Churned", "monthly_charges": "Monthly charges ($)"},
        )
        fig4.update_layout(
            legend=dict(title="", x=0.75, y=0.95),
            margin=dict(t=20, b=20),
            height=260,
        )
        st.plotly_chart(fig4, width='stretch')


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — MODEL COMPARISON
# ════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Comparison":
    st.title("Model Comparison")
    st.markdown("5-fold cross-validation results and held-out test set performance.")
    st.divider()

    # ── Summary table ─────────────────────────────────────────────────────────
    rows = []
    for r in results["models"]:
        rows.append({
            "Model":     r["model"].replace("_", " ").title(),
            "Precision": r["precision"],
            "Recall":    r["recall"],
            "F1":        r["f1"],
            "ROC-AUC":   r["roc_auc"],
            "CV F1 (mean ± std)": (
                f"{r['cv']['f1']['mean']:.3f} ± {r['cv']['f1']['std']:.3f}"
            ),
        })
    summary_df = pd.DataFrame(rows).set_index("Model")

    best_model_name = results["best_model"].replace("_", " ").title()
    st.dataframe(
        summary_df.style.highlight_max(
            subset=["Precision", "Recall", "F1", "ROC-AUC"],
            color="#bbf7d0",
            axis=0,
        ).format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}", "ROC-AUC": "{:.3f}"}),
        width='stretch',
    )
    st.caption(f"✅ Best model: **{best_model_name}** (highest F1 on test set)")

    st.divider()
    col_left, col_right = st.columns(2)

    # ── CV F1 box comparison ──────────────────────────────────────────────────
    with col_left:
        st.subheader("Cross-validation F1 (5-fold)")
        cv_data = []
        for r in results["models"]:
            for fold_score in r["cv"]["f1"]["folds"]:
                cv_data.append({
                    "Model": r["model"].replace("_", " ").title(),
                    "F1":    fold_score,
                })
        cv_df = pd.DataFrame(cv_data)
        fig_cv = px.box(
            cv_df, x="Model", y="F1",
            color="Model",
            points="all",
            color_discrete_sequence=["#2563eb", "#16a34a", "#ea580c"],
        )
        fig_cv.update_layout(showlegend=False, margin=dict(t=20, b=20), height=300)
        st.plotly_chart(fig_cv, width='stretch')

    # ── Confusion matrices ────────────────────────────────────────────────────
    with col_right:
        st.subheader("Confusion matrix")
        selected_model = st.selectbox(
            "Select model",
            [r["model"].replace("_", " ").title() for r in results["models"]],
        )
        cm_data = next(
            r["confusion_matrix"] for r in results["models"]
            if r["model"].replace("_", " ").title() == selected_model
        )
        cm = np.array(cm_data)
        cm_labels = ["Stay", "Churn"]
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            x=["Pred: Stay", "Pred: Churn"],
            y=["Actual: Stay", "Actual: Churn"],
            color_continuous_scale="Blues",
            aspect="auto",
        )
        fig_cm.update_layout(margin=dict(t=20, b=20), height=300)
        st.plotly_chart(fig_cm, width='stretch')

    # ── Metric radar chart ────────────────────────────────────────────────────
    st.subheader("Metric radar comparison")
    metrics = ["Precision", "Recall", "F1", "ROC-AUC"]
    colors  = ["#2563eb", "#16a34a", "#ea580c"]
    fig_radar = go.Figure()

    for i, r in enumerate(results["models"]):
        values = [r["precision"], r["recall"], r["f1"], r["roc_auc"]]
        values += [values[0]]  # close loop
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics + [metrics[0]],
            fill="toself",
            name=r["model"].replace("_", " ").title(),
            line_color=colors[i],
            opacity=0.6,
        ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0.5, 1.0])),
        height=380,
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig_radar, width='stretch')


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — PREDICT CUSTOMER
# ════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Predict Customer":
    st.title("Predict Churn Risk")
    st.markdown("Adjust customer attributes to get a real-time churn probability from each model.")
    st.divider()

    col_form, col_result = st.columns([1, 1])

    with col_form:
        st.subheader("Customer attributes")

        tenure         = st.slider("Tenure (months)",         1,  72,  12)
        monthly_charge = st.slider("Monthly charges ($)",    18, 120,  65)
        support_calls  = st.slider("Support calls (6 mo)",    0,  10,   2)
        contract       = st.selectbox("Contract", [0, 1, 2],
                                       format_func=lambda x: {0:"Month-to-month",1:"One year",2:"Two year"}[x])
        internet       = st.selectbox("Internet service", [0, 1, 2],
                                       format_func=lambda x: {0:"None",1:"DSL",2:"Fiber optic"}[x])

        st.markdown("**Add-on services**")
        c1, c2 = st.columns(2)
        with c1:
            online_sec  = int(st.checkbox("Online security"))
            online_bk   = int(st.checkbox("Online backup"))
            device_prot = int(st.checkbox("Device protection"))
            tech_sup    = int(st.checkbox("Tech support"))
        with c2:
            stream_tv   = int(st.checkbox("Streaming TV"))
            stream_mv   = int(st.checkbox("Streaming movies"))
            senior      = int(st.checkbox("Senior citizen"))
            partner     = int(st.checkbox("Has partner"))

        payment        = st.selectbox("Payment method", [0, 1, 2, 3],
                                       format_func=lambda x:
                                       {0:"Electronic check",1:"Mailed check",
                                        2:"Bank transfer",3:"Credit card"}[x])
        paperless      = int(st.checkbox("Paperless billing", value=True))
        dependents     = int(st.checkbox("Has dependents"))
        phone_svc      = int(st.checkbox("Phone service", value=True))
        multi_lines    = int(st.checkbox("Multiple lines"))

    # Build feature row
    total_charges = round(monthly_charge * tenure * np.random.uniform(0.98, 1.02), 2)
    num_services  = sum([online_sec, online_bk, device_prot, tech_sup,
                         stream_tv, stream_mv, multi_lines, phone_svc])

    input_df = pd.DataFrame([{
        "tenure_months":     tenure,
        "monthly_charges":   monthly_charge,
        "total_charges":     total_charges,
        "support_calls":     support_calls,
        "num_services":      num_services,
        "charge_per_month_ratio": round(monthly_charge / (tenure + 1), 4),
        "tenure_bucket":     min(3, tenure // 13),
        "contract":          contract,
        "internet_service":  internet,
        "payment_method":    payment,
        "senior_citizen":    senior,
        "partner":           partner,
        "dependents":        dependents,
        "phone_service":     phone_svc,
        "multiple_lines":    multi_lines,
        "online_security":   online_sec,
        "online_backup":     online_bk,
        "device_protection": device_prot,
        "tech_support":      tech_sup,
        "streaming_tv":      stream_tv,
        "streaming_movies":  stream_mv,
        "paperless_billing": paperless,
    }])

    input_pp = preprocessor.transform(input_df)

    with col_result:
        st.subheader("Churn probability by model")
        st.markdown("")

        probabilities = {}
        for model_name, clf in models.items():
            prob = clf.predict_proba(input_pp)[0][1]
            probabilities[model_name] = prob

        # Gauge-style display for each model
        for model_name, prob in probabilities.items():
            pct = round(prob * 100, 1)
            if prob >= 0.65:
                color = "#dc2626"
                risk  = "HIGH RISK"
            elif prob >= 0.35:
                color = "#d97706"
                risk  = "MEDIUM RISK"
            else:
                color = "#16a34a"
                risk  = "LOW RISK"

            st.markdown(f"**{model_name}**")
            col_val, col_bar = st.columns([1, 3])
            with col_val:
                st.markdown(
                    f"<span style='font-size:1.8rem;font-weight:600;color:{color}'>{pct}%</span>"
                    f"<br><span style='font-size:0.7rem;color:{color}'>{risk}</span>",
                    unsafe_allow_html=True,
                )
            with col_bar:
                st.progress(float(prob))
            st.markdown("")

        st.divider()

        # Ensemble average
        avg_prob = np.mean(list(probabilities.values()))
        avg_pct  = round(avg_prob * 100, 1)

        if avg_prob >= 0.65:
            rec_color = "#dc2626"
            recommendation = "🚨 Immediate action recommended — offer retention incentive or escalate to account manager."
        elif avg_prob >= 0.35:
            rec_color = "#d97706"
            recommendation = "⚠️ Monitor closely — consider proactive check-in or targeted discount."
        else:
            rec_color = "#16a34a"
            recommendation = "✅ Customer appears stable — continue standard engagement."

        st.markdown(f"### Ensemble average: **{avg_pct}%**")
        st.info(recommendation)


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — FEATURE INSIGHTS
# ════════════════════════════════════════════════════════════════════════════
elif page == "💡 Feature Insights":
    st.title("Feature Insights")
    st.markdown("What drives churn — and what business levers to pull.")
    st.divider()

    # ── Feature importances (Random Forest + XGBoost) ─────────────────────────
    rf_clf  = models["Random Forest"]
    xgb_clf = models["XGBoost"]

    all_features = (
        NUMERIC_FEATURES
        + [f"cat_{c}" for c in CATEGORICAL_FEATURES]
        + BINARY_FEATURES
    )

    rf_importances  = rf_clf.feature_importances_
    xgb_importances = xgb_clf.feature_importances_
    n_feats = min(len(rf_importances), len(xgb_importances), len(all_features))

    feat_df = pd.DataFrame({
        "Feature":       all_features[:n_feats],
        "Random Forest": rf_importances[:n_feats],
        "XGBoost":       xgb_importances[:n_feats],
    }).sort_values("Random Forest", ascending=False).head(12)

    col_rf, col_xgb = st.columns(2)

    with col_rf:
        st.subheader("Random Forest — top 12 features")
        fig_rf = px.bar(
            feat_df.sort_values("Random Forest"),
            x="Random Forest", y="Feature",
            orientation="h",
            color="Random Forest",
            color_continuous_scale="Blues",
        )
        fig_rf.update_layout(
            coloraxis_showscale=False,
            margin=dict(t=10, b=10),
            height=380,
        )
        st.plotly_chart(fig_rf, width='stretch')

    with col_xgb:
        st.subheader("XGBoost — top 12 features")
        xgb_top = feat_df.sort_values("XGBoost")
        fig_xgb = px.bar(
            xgb_top,
            x="XGBoost", y="Feature",
            orientation="h",
            color="XGBoost",
            color_continuous_scale="Oranges",
        )
        fig_xgb.update_layout(
            coloraxis_showscale=False,
            margin=dict(t=10, b=10),
            height=380,
        )
        st.plotly_chart(fig_xgb, width='stretch')

    st.divider()

    # ── Business recommendations ──────────────────────────────────────────────
    st.subheader("Business recommendations")

    recs = [
        ("📄 Contract upgrade campaigns",
         "Month-to-month customers churn at 3–4× the rate of annual subscribers. "
         "Offer a discount or bonus incentive to move customers to 1-year contracts "
         "during their first 3 months — the highest-risk window."),
        ("🔧 Proactive support outreach",
         "Each additional support call in the last 6 months correlates with a "
         "~7 percentage-point increase in churn probability. Flag customers with "
         "2+ unresolved tickets for a proactive callback before they escalate."),
        ("💰 Charge sensitivity tiers",
         "Customers paying over $80/month are significantly more likely to churn. "
         "Consider loyalty discounts at the 12-month mark for high-spend customers "
         "or bundle add-ons to increase perceived value."),
        ("🛡️ Security & support bundle",
         "Online security and tech support are the top retention services. "
         "Offering these as a free trial in month 1 significantly reduces early churn — "
         "consider an auto-enroll then opt-out model."),
        ("📊 Predictive retention scoring",
         "Run the XGBoost model daily on your CRM to generate a churn score for "
         "every active customer. Route the top 5% to your retention team for "
         "outbound contact — a cost-effective targeting strategy."),
    ]

    for title, body in recs:
        with st.expander(title):
            st.markdown(body)
