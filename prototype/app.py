"""
Steel Plates Defect IDSS — Streamlit Prototype
AIS431 Final Project — Mohamed Sherif (221000142) & Mohamed Osama (221001647)
"""
import sys
import os
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap
import streamlit as st

from src.config import MODELS_DIR, DATA_PROCESSED, FIGURES_DIR
from chatbot import chat, get_suggested_questions

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Steel Plates Defect IDSS",
    page_icon="🏭",
    layout="wide",
)

# ── Load model artefacts (cached) ─────────────────────────────────────────────
@st.cache_resource
def load_artefacts():
    model         = joblib.load(MODELS_DIR / "best_model.pkl")
    pipeline      = joblib.load(MODELS_DIR / "preprocessing_pipeline.pkl")
    label_classes = joblib.load(MODELS_DIR / "label_classes.pkl")
    le            = joblib.load(MODELS_DIR / "label_encoder.pkl")
    feat_names    = joblib.load(MODELS_DIR / "feature_names.pkl")
    rf            = model.named_estimators_["rf"]
    explainer     = shap.TreeExplainer(rf)
    return model, pipeline, label_classes, le, feat_names, rf, explainer

model, pipeline, label_classes, le, feat_names, rf, explainer = load_artefacts()
final_classes = list(label_classes)

# ── Training-set medians for defaults ─────────────────────────────────────────
@st.cache_data
def get_train_medians():
    X_raw = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "features.csv")
    return X_raw.median()

train_medians = get_train_medians()

# ── Raw feature column order (matches original CSV) ───────────────────────────
ALL_RAW_COLS = [
    'X_Minimum', 'X_Maximum', 'Y_Minimum', 'Y_Maximum', 'Pixels_Areas',
    'X_Perimeter', 'Y_Perimeter', 'Sum_of_Luminosity', 'Minimum_of_Luminosity',
    'Maximum_of_Luminosity', 'Length_of_Conveyer', 'TypeOfSteel_A300',
    'TypeOfSteel_A400', 'Steel_Plate_Thickness', 'Edges_Index', 'Empty_Index',
    'Square_Index', 'Outside_X_Index', 'Edges_X_Index', 'Edges_Y_Index',
    'Outside_Global_Index', 'LogOfAreas', 'Log_X_Index', 'Log_Y_Index',
    'Orientation_Index', 'Luminosity_Index', 'SigmoidOfAreas'
]

# ── Risk tier ─────────────────────────────────────────────────────────────────
def risk_tier(max_prob):
    if max_prob >= 0.70:
        return "High", "#e74c3c"
    elif max_prob >= 0.40:
        return "Medium", "#e67e22"
    return "Low", "#27ae60"

# ── Action map ────────────────────────────────────────────────────────────────
ACTION_MAP = {
    "Pastry":       ("Reject — send to scrap line",               "red"),
    "Z_Scratch":    ("Reject — deep scratch, unusable",           "red"),
    "K_Scatch":     ("Reject — cross-direction scratch",          "red"),
    "Stains":       ("Hold — chemical cleaning possible",         "orange"),
    "Dirtiness":    ("Hold — surface cleaning + re-inspect",      "orange"),
    "Bumps":        ("Conditional accept — check tolerance spec", "orange"),
    "Other_Faults": ("Human review — ambiguous category",         "blue"),
}

# ── Session state init ────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "prediction_context" not in st.session_state:
    st.session_state.prediction_context = {}

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🏭 Steel Plates IDSS")
st.sidebar.markdown("**Input Mode**")
input_mode = st.sidebar.radio(
    "Input Mode",
    ["Manual Entry", "CSV Upload"],
    horizontal=True,
    label_visibility="collapsed",
)

def collect_manual_input():
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Sensor Readings** (use training-set medians as defaults)")
    vals = {}
    for col in ALL_RAW_COLS:
        default = float(train_medians.get(col, 0.0))
        if col in ["TypeOfSteel_A300", "TypeOfSteel_A400"]:
            vals[col] = st.sidebar.selectbox(col, [0, 1], index=int(default))
        else:
            vals[col] = st.sidebar.number_input(col, value=default, format="%.4f")
    return pd.DataFrame([vals])

def run_prediction(raw_df):
    for c in ALL_RAW_COLS:
        if c not in raw_df.columns:
            raw_df[c] = train_medians.get(c, 0.0)
    raw_df = raw_df[ALL_RAW_COLS]
    X_scaled = pipeline.transform(raw_df)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feat_names)
    proba = model.predict_proba(X_scaled_df)
    preds = model.predict(X_scaled_df)
    return X_scaled_df, proba, preds

def compute_shap(X_scaled_df, predicted_class):
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
    return sv_cls, base_val, sv_list, cls_idx

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🏭 Steel Plates Defect Decision-Support System")
st.markdown(
    "Upload a CSV or enter sensor readings manually. "
    "The system classifies the defect type, shows confidence, and explains the prediction."
)

tab_predict, tab_chat = st.tabs(["🔍 Prediction", "🤖 QC Assistant"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    if input_mode == "Manual Entry":
        raw_df    = collect_manual_input()
        submitted = st.sidebar.button("🔍 Predict", type="primary", use_container_width=True)

        if submitted:
            X_scaled_df, proba, preds = run_prediction(raw_df)
            predicted_class = preds[0]
            max_prob        = proba[0].max()
            tier, tier_color        = risk_tier(max_prob)
            action, action_color    = ACTION_MAP.get(predicted_class, ("Manual review", "blue"))

            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted Defect", predicted_class)
            col2.metric("Confidence", f"{max_prob:.1%}")
            col3.markdown(
                f"<div style='background:{tier_color};padding:12px;border-radius:8px;"
                f"color:white;text-align:center;font-size:1.1em;font-weight:bold;'>"
                f"Risk: {tier}</div>",
                unsafe_allow_html=True,
            )

            st.markdown("---")
            badge_color = {"red": "#c0392b", "orange": "#d35400", "blue": "#2980b9"}[action_color]
            st.markdown(
                f"<div style='background:{badge_color};padding:10px;border-radius:6px;"
                f"color:white;font-size:1.05em;'><b>Recommended Action:</b> {action}</div>",
                unsafe_allow_html=True,
            )
            st.markdown("---")

            # Probability bar chart
            st.subheader("Class Probabilities")
            prob_df = pd.DataFrame({
                "Class": final_classes,
                "Probability": proba[0],
            }).sort_values("Probability", ascending=True)

            fig_prob, ax = plt.subplots(figsize=(8, 4))
            colors_bar = ["#e74c3c" if c == predicted_class else "#3498db" for c in prob_df["Class"]]
            ax.barh(prob_df["Class"], prob_df["Probability"], color=colors_bar)
            ax.axvline(0.5, color="grey", linestyle="--", linewidth=0.8)
            ax.set_xlabel("Probability")
            ax.set_xlim(0, 1)
            for i, v in enumerate(prob_df["Probability"]):
                ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig_prob)
            plt.close(fig_prob)

            # SHAP
            st.subheader("SHAP Explanation")
            try:
                sv_cls, base_val, sv_list, cls_idx = compute_shap(X_scaled_df, predicted_class)

                explanation = shap.Explanation(
                    values=sv_cls,
                    base_values=float(base_val),
                    data=X_scaled_df.values[0],
                    feature_names=feat_names,
                )
                fig_shap, _ = plt.subplots(figsize=(9, 5))
                shap.waterfall_plot(explanation, max_display=10, show=False)
                plt.title(f"SHAP — Why '{predicted_class}'?", fontsize=12, fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig_shap)
                plt.close(fig_shap)

                st.subheader("Top Feature Contributions")
                contrib_df = pd.DataFrame({
                    "Feature":       feat_names,
                    "SHAP Value":    sv_cls,
                    "Feature Value": X_scaled_df.values[0],
                }).sort_values("SHAP Value", key=abs, ascending=False).head(5).reset_index(drop=True)
                contrib_df["Direction"] = contrib_df["SHAP Value"].apply(
                    lambda v: "↑ Increases probability" if v > 0 else "↓ Decreases probability"
                )
                st.dataframe(
                    contrib_df[["Feature", "SHAP Value", "Feature Value", "Direction"]],
                    use_container_width=True,
                )

                # Save context for chatbot
                top_feats = contrib_df.apply(
                    lambda r: {"name": r["Feature"], "shap": r["SHAP Value"], "value": r["Feature Value"]},
                    axis=1,
                ).tolist()
                st.session_state.prediction_context = {
                    "predicted_class": predicted_class,
                    "confidence":      max_prob,
                    "risk_tier":       tier,
                    "action":          action,
                    "top_features":    top_feats,
                    "all_probs":       dict(zip(final_classes, proba[0])),
                }
                st.success("✅ Prediction saved — switch to **🤖 QC Assistant** tab to ask questions.")

            except Exception as e:
                st.warning(f"SHAP explanation unavailable: {e}")

        else:
            st.info("Configure sensor readings in the sidebar and click **Predict**.")

    else:
        st.subheader("CSV Upload")
        uploaded = st.file_uploader(
            "Upload a CSV with original feature columns (27 raw features)", type=["csv"]
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
                "Predicted_Class":    preds,
                "Confidence":         max_proba.round(4),
                "Risk_Tier":          [assign_tier(p) for p in max_proba],
                "Recommended_Action": [ACTION_MAP.get(c, ("Manual review", ""))[0] for c in preds],
            })
            for cls, col in zip(final_classes, proba.T):
                results_df[f"P({cls})"] = col.round(4)

            st.subheader("Predictions")
            st.dataframe(results_df, use_container_width=True)

            tier_counts = results_df["Risk_Tier"].value_counts()
            fig_tier, ax_t = plt.subplots(figsize=(6, 3))
            colors_t = {"High": "#e74c3c", "Medium": "#e67e22", "Low": "#27ae60"}
            for t in ["High", "Medium", "Low"]:
                if t in tier_counts:
                    ax_t.bar(t, tier_counts[t], color=colors_t[t])
            ax_t.set_title("Risk Tier Distribution", fontweight="bold")
            ax_t.set_ylabel("Count")
            plt.tight_layout()
            st.pyplot(fig_tier)
            plt.close(fig_tier)

            # Save batch context for chatbot
            st.session_state.prediction_context = {
                "batch":          True,
                "total_rows":     len(results_df),
                "avg_confidence": results_df["Confidence"].mean(),
                "tier_summary":   tier_counts.to_dict(),
                "class_summary":  results_df["Predicted_Class"].value_counts().to_dict(),
            }
            st.success("✅ Batch results saved — switch to **🤖 QC Assistant** tab to ask questions.")

            st.download_button(
                "⬇ Download Predictions CSV",
                results_df.to_csv(index=False),
                "predictions.csv",
                "text/csv",
            )
        else:
            st.info("Upload a CSV file to get batch predictions.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — QC ASSISTANT CHATBOT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.subheader("🤖 QC Assistant — Ask me about any prediction")

    ctx = st.session_state.prediction_context
    if ctx:
        if ctx.get("batch"):
            st.info(
                f"📊 **Batch context loaded** — {ctx['total_rows']} plates | "
                f"Avg confidence: {ctx['avg_confidence']:.1%} | "
                f"Tiers: {ctx.get('tier_summary', {})}"
            )
        else:
            icon = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}.get(ctx.get("risk_tier", ""), "⚪")
            st.info(
                f"{icon} **Prediction context loaded** — "
                f"**{ctx.get('predicted_class')}** | "
                f"Confidence: {ctx.get('confidence', 0):.1%} | "
                f"Risk: {ctx.get('risk_tier')}"
            )
    else:
        st.warning(
            "💡 No prediction context yet. Run a prediction in the **🔍 Prediction** tab first, "
            "or ask general QC questions below."
        )

    st.markdown("---")

    # Suggested questions
    predicted_class = ctx.get("predicted_class") if ctx and not ctx.get("batch") else None
    suggestions     = get_suggested_questions(predicted_class)

    st.markdown("**Suggested questions:**")
    cols = st.columns(len(suggestions))
    for i, (col, q) in enumerate(zip(cols, suggestions)):
        with col:
            if st.button(q, key=f"suggestion_{i}", use_container_width=True):
                st.session_state["pending_question"] = q

    st.markdown("---")

    # Chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    pending    = st.session_state.pop("pending_question", None)
    user_input = st.chat_input("Ask about the prediction, defect types, or QC procedures...")
    question   = pending or user_input

    if question:
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    reply, updated_history = chat(
                        user_message=question,
                        history=st.session_state.chat_history,
                        prediction_context=st.session_state.prediction_context,
                        model="llama-3.3-70b-versatile",
                    )
                    st.markdown(reply)
                    st.session_state.chat_history = updated_history
                except Exception as e:
                    err = str(e)
                    if "401" in err or "authentication" in err.lower():
                        st.error("❌ Invalid API key.")
                    elif "429" in err:
                        st.error("❌ Rate limit hit. Wait a moment and try again.")
                    else:
                        st.error(f"❌ API error: {err}")

    if st.session_state.chat_history:
        st.markdown("---")
        if st.button("🗑️ Clear conversation", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "AIS431 Final Project — Mohamed Sherif (221000142) & Mohamed Osama (221001647) | "
    "Model: Stacking Ensemble | Macro AUC: 0.9545"
)