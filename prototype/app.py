"""
Steel Plates Defect IDSS — Streamlit Prototype
AIS431 Final Project — Mohamed Sherif (221000142) & Mohamed Osama (221001647)
"""
import sys
import os
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import shap
import streamlit as st

from src.config import MODELS_DIR, DATA_PROCESSED, FIGURES_DIR

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Steel Plates Defect IDSS",
    page_icon="🏭",
    layout="wide",
)

# ── Load model artefacts (cached) ─────────────────────────────────────────────
@st.cache_resource
def load_artefacts():
    model        = joblib.load(MODELS_DIR / "best_model.pkl")
    pipeline     = joblib.load(MODELS_DIR / "preprocessing_pipeline.pkl")
    label_classes = joblib.load(MODELS_DIR / "label_classes.pkl")
    le           = joblib.load(MODELS_DIR / "label_encoder.pkl")
    feat_names   = joblib.load(MODELS_DIR / "feature_names.pkl")
    # Extract RF from stacking for SHAP
    rf = model.named_estimators_["rf"]
    explainer = shap.TreeExplainer(rf)
    return model, pipeline, label_classes, le, feat_names, rf, explainer

model, pipeline, label_classes, le, feat_names, rf, explainer = load_artefacts()
final_classes = list(label_classes)

# ── Training-set medians for defaults ──────────────────────────────────────────
@st.cache_data
def get_train_medians():
    X_raw = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "features.csv")
    return X_raw.median()

train_medians = get_train_medians()

# ── Risk tier ─────────────────────────────────────────────────────────────────
def risk_tier(max_prob):
    if max_prob >= 0.70:
        return "High", "#e74c3c"
    elif max_prob >= 0.40:
        return "Medium", "#e67e22"
    return "Low", "#27ae60"

# ── Action map ────────────────────────────────────────────────────────────────
ACTION_MAP = {
    "Pastry":       ("Reject — send to scrap line",              "red"),
    "Z_Scratch":    ("Reject — deep scratch, unusable",          "red"),
    "K_Scatch":     ("Reject — cross-direction scratch",         "red"),
    "Stains":       ("Hold — chemical cleaning possible",        "orange"),
    "Dirtiness":    ("Hold — surface cleaning + re-inspect",     "orange"),
    "Bumps":        ("Conditional accept — check tolerance spec","orange"),
    "Other_Faults": ("Human review — ambiguous category",        "blue"),
}

# ── Sidebar: input mode ────────────────────────────────────────────────────────
st.sidebar.title("🏭 Steel Plates IDSS")
st.sidebar.markdown("**Input Mode**")
input_mode = st.sidebar.radio("", ["Manual Entry", "CSV Upload"], horizontal=True)

# ── Raw feature columns expected by the pipeline ──────────────────────────────
RAW_COLS = [c for c in train_medians.index.tolist() if c != "TypeOfSteel_A300"]
# Make sure TypeOfSteel_A300 is present for the pipeline (it gets dropped internally)
ALL_RAW_COLS = train_medians.index.tolist()


def collect_manual_input():
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Sensor Readings** (use training-set medians as defaults)")
    vals = {}
    for col in ALL_RAW_COLS:
        default = float(train_medians.get(col, 0.0))
        if col in ["TypeOfSteel_A300", "TypeOfSteel_A400"]:
            vals[col] = st.sidebar.selectbox(col, [0, 1], index=int(default))
        elif train_medians.get(col, 0.0) <= 1.0 and train_medians.get(col, 0.0) >= 0.0 and col.startswith("Type"):
            vals[col] = st.sidebar.selectbox(col, [0, 1], index=int(default))
        else:
            vals[col] = st.sidebar.number_input(col, value=default, format="%.4f")
    return pd.DataFrame([vals])


def run_prediction(raw_df):
    """Run pipeline → model → return probabilities and predictions."""
    # Ensure column order matches what pipeline was fitted on
    for c in ALL_RAW_COLS:
        if c not in raw_df.columns:
            raw_df[c] = train_medians.get(c, 0.0)
    raw_df = raw_df[ALL_RAW_COLS]

    X_scaled = pipeline.transform(raw_df)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feat_names)

    proba = model.predict_proba(X_scaled_df)
    preds = model.predict(X_scaled_df)
    return X_scaled_df, proba, preds


# ── Main area ──────────────────────────────────────────────────────────────────
st.title("🏭 Steel Plates Defect Decision-Support System")
st.markdown(
    "Upload a CSV or enter sensor readings manually. "
    "The system classifies the defect type, shows confidence, and explains the prediction."
)

if input_mode == "Manual Entry":
    raw_df = collect_manual_input()
    submitted = st.sidebar.button("🔍 Predict", type="primary", use_container_width=True)

    if submitted:
        X_scaled_df, proba, preds = run_prediction(raw_df)
        predicted_class = preds[0]
        max_prob = proba[0].max()
        tier, tier_color = risk_tier(max_prob)
        action, action_color = ACTION_MAP.get(predicted_class, ("Manual review", "blue"))

        # ── Results ────────────────────────────────────────────────────────────
        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Defect", predicted_class)
        col2.metric("Confidence", f"{max_prob:.1%}")
        col3.markdown(
            f"<div style='background:{tier_color};padding:12px;border-radius:8px;"
            f"color:white;text-align:center;font-size:1.1em;font-weight:bold;'>"
            f"Risk: {tier}</div>",
            unsafe_allow_html=True
        )

        st.markdown("---")
        action_badge_color = {"red": "#c0392b", "orange": "#d35400", "blue": "#2980b9"}[action_color]
        st.markdown(
            f"<div style='background:{action_badge_color};padding:10px;border-radius:6px;"
            f"color:white;font-size:1.05em;'><b>Recommended Action:</b> {action}</div>",
            unsafe_allow_html=True
        )
        st.markdown("---")

        # ── Probability bar chart ───────────────────────────────────────────────
        st.subheader("Class Probabilities")
        prob_df = pd.DataFrame({
            "Class": final_classes,
            "Probability": proba[0]
        }).sort_values("Probability", ascending=True)

        fig_prob, ax = plt.subplots(figsize=(8, 4))
        colors_bar = ["#e74c3c" if c == predicted_class else "#3498db"
                      for c in prob_df["Class"]]
        ax.barh(prob_df["Class"], prob_df["Probability"], color=colors_bar)
        ax.axvline(0.5, color="grey", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Probability")
        ax.set_xlim(0, 1)
        for i, (v, c) in enumerate(zip(prob_df["Probability"], prob_df["Class"])):
            ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_prob)
        plt.close(fig_prob)

        # ── SHAP waterfall ──────────────────────────────────────────────────────
        st.subheader("SHAP Explanation")
        try:
            sv_row = explainer.shap_values(X_scaled_df)
            if isinstance(sv_row, np.ndarray) and sv_row.ndim == 3:
                sv_list = [sv_row[:, :, k] for k in range(sv_row.shape[2])]
            elif isinstance(sv_row, list):
                sv_list = sv_row
            else:
                sv_list = [sv_row]

            rf_classes_list = list(rf.classes_)
            cls_idx = rf_classes_list.index(predicted_class) if predicted_class in rf_classes_list else 0
            sv_cls = sv_list[cls_idx][0]
            base_val = (explainer.expected_value[cls_idx]
                        if isinstance(explainer.expected_value, (list, np.ndarray))
                        else explainer.expected_value)

            explanation = shap.Explanation(
                values=sv_cls,
                base_values=float(base_val),
                data=X_scaled_df.values[0],
                feature_names=feat_names,
            )
            fig_shap, ax_shap = plt.subplots(figsize=(9, 5))
            shap.waterfall_plot(explanation, max_display=10, show=False)
            plt.title(f"SHAP — Why '{predicted_class}'?", fontsize=12, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_shap)
            plt.close(fig_shap)

            # Top-5 feature contributions table
            st.subheader("Top Feature Contributions")
            contrib_df = pd.DataFrame({
                "Feature": feat_names,
                "SHAP Value": sv_cls,
                "Feature Value": X_scaled_df.values[0],
            }).sort_values("SHAP Value", key=abs, ascending=False).head(5).reset_index(drop=True)
            contrib_df["Direction"] = contrib_df["SHAP Value"].apply(
                lambda v: "↑ Increases probability" if v > 0 else "↓ Decreases probability"
            )
            st.dataframe(contrib_df[["Feature", "SHAP Value", "Feature Value", "Direction"]],
                         use_container_width=True)

        except Exception as e:
            st.warning(f"SHAP explanation unavailable: {e}")

    else:
        st.info("Configure sensor readings in the sidebar and click **Predict**.")

else:
    # ── CSV Upload ─────────────────────────────────────────────────────────────
    st.subheader("CSV Upload")
    uploaded = st.file_uploader(
        "Upload a CSV with original feature columns (27 raw features)",
        type=["csv"]
    )
    if uploaded is not None:
        raw_df = pd.read_csv(uploaded)
        st.write(f"Loaded {len(raw_df)} rows × {raw_df.shape[1]} columns.")
        st.dataframe(raw_df.head(5))

        X_scaled_df, proba, preds = run_prediction(raw_df)
        max_proba = proba.max(axis=1)

        def assign_tier(p):
            if p >= 0.70: return "High"
            if p >= 0.40: return "Medium"
            return "Low"

        results_df = pd.DataFrame({
            "Predicted_Class": preds,
            "Confidence": max_proba.round(4),
            "Risk_Tier": [assign_tier(p) for p in max_proba],
            "Recommended_Action": [ACTION_MAP.get(c, ("Manual review", ""))[0] for c in preds],
        })
        for cls, col in zip(final_classes, proba.T):
            results_df[f"P({cls})"] = col.round(4)

        st.subheader("Predictions")
        st.dataframe(results_df, use_container_width=True)

        # Summary bar
        tier_counts = results_df["Risk_Tier"].value_counts()
        fig_tier, ax_t = plt.subplots(figsize=(6, 3))
        colors_t = {"High": "#e74c3c", "Medium": "#e67e22", "Low": "#27ae60"}
        for tier in ["High", "Medium", "Low"]:
            if tier in tier_counts:
                ax_t.bar(tier, tier_counts[tier], color=colors_t[tier])
        ax_t.set_title("Risk Tier Distribution", fontweight="bold")
        ax_t.set_ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig_tier)
        plt.close(fig_tier)

        # Download
        csv_out = results_df.to_csv(index=False)
        st.download_button("⬇ Download Predictions CSV", csv_out, "predictions.csv", "text/csv")
    else:
        st.info("Upload a CSV file to get batch predictions.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "AIS431 Final Project — Mohamed Sherif (221000142) & Mohamed Osama (221001647) | "
    f"Model: Stacking Ensemble | Macro AUC: 0.9545"
)
