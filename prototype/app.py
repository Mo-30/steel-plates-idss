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
import plotly.express as px
import plotly.graph_objects as go
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

# ── Training-set medians + realistic ranges ───────────────────────────────────
@st.cache_data
def get_train_medians():
    X_raw = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "features.csv")
    return X_raw.median()

@st.cache_data
def get_feature_ranges():
    X_raw = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "features.csv")
    return X_raw.quantile(0.05).to_dict(), X_raw.quantile(0.95).to_dict()

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

CLASS_COLORS = {
    "Pastry":       "#e74c3c",
    "Z_Scratch":    "#9b59b6",
    "K_Scatch":     "#3498db",
    "Stains":       "#f39c12",
    "Dirtiness":    "#1abc9c",
    "Bumps":        "#e67e22",
    "Other_Faults": "#95a5a6",
}

# ── Session state init ────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "prediction_context" not in st.session_state:
    st.session_state.prediction_context = {}
if "batch_results" not in st.session_state:
    st.session_state.batch_results = None
if "batch_X_scaled" not in st.session_state:
    st.session_state.batch_X_scaled = None
if "batch_shap_computed" not in st.session_state:
    st.session_state.batch_shap_computed = False
if "batch_shap_importance" not in st.session_state:
    st.session_state.batch_shap_importance = None
if "batch_shap_values" not in st.session_state:
    st.session_state.batch_shap_values = None
if "batch_X_sample" not in st.session_state:
    st.session_state.batch_X_sample = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🏭 Steel Plates IDSS")
st.sidebar.markdown("**Input Mode**")
input_mode = st.sidebar.radio(
    "Input Mode",
    ["Manual Entry", "CSV Upload"],
    horizontal=True,
    label_visibility="collapsed",
)

def build_dashboard_summary(res_df):
    """Build a text-rich dict describing every dashboard panel — fed to the chatbot."""
    total = len(res_df)
    n_high = int((res_df["Risk_Tier"] == "High").sum())
    n_med  = int((res_df["Risk_Tier"] == "Medium").sum())
    n_low  = int((res_df["Risk_Tier"] == "Low").sum())

    class_counts = res_df["Predicted_Class"].value_counts()
    conf_stats   = res_df.groupby("Predicted_Class")["Confidence"].agg(["mean", "min", "max"])
    risk_cross   = (
        res_df.groupby(["Predicted_Class", "Risk_Tier"])
        .size().unstack(fill_value=0)
    )

    prob_cols = [c for c in res_df.columns if c.startswith("P(")]
    heatmap_means = res_df.groupby("Predicted_Class")[prob_cols].mean() if prob_cols else None

    defect_dist = {
        cls: f"{cnt} plates ({cnt/total:.0%})"
        for cls, cnt in class_counts.items()
    }
    conf_by_cls = {
        cls: f"avg {row['mean']:.1%}, range {row['min']:.1%}–{row['max']:.1%}"
        for cls, row in conf_stats.iterrows()
    }
    risk_by_cls = {}
    for cls in risk_cross.index:
        row = risk_cross.loc[cls]
        risk_by_cls[cls] = (
            f"High={int(row.get('High', 0))}, "
            f"Medium={int(row.get('Medium', 0))}, "
            f"Low={int(row.get('Low', 0))}"
        )

    heatmap_notes = []
    if heatmap_means is not None:
        for cls in heatmap_means.index:
            row_probs = heatmap_means.loc[cls].sort_values(ascending=False)
            self_col  = f"P({cls})"
            self_val  = heatmap_means.loc[cls].get(self_col, 0)
            # top 2 off-diagonal (confusion) entries
            off_diag = row_probs.drop(labels=[self_col], errors="ignore").head(2)
            confusion_str = ", ".join(
                f"'{c[2:-1]}'={v:.2f}" for c, v in off_diag.items() if v > 0.02
            )
            heatmap_notes.append(
                f"  - Predicted as {cls}: self-probability={self_val:.2f}"
                + (f", confused with: {confusion_str}" if confusion_str else " (no significant confusion)")
            )

    return {
        "kpi": (
            f"Total={total}, High={n_high} ({n_high/total:.0%}), "
            f"Medium={n_med} ({n_med/total:.0%}), Low={n_low} ({n_low/total:.0%}), "
            f"Top defect={class_counts.index[0]} ({class_counts.iloc[0]} plates), "
            f"Avg confidence={res_df['Confidence'].mean():.1%}"
        ),
        "defect_distribution": defect_dist,
        "risk_tier_breakdown": {
            "High":   f"{n_high} plates ({n_high/total:.0%})",
            "Medium": f"{n_med} plates ({n_med/total:.0%})",
            "Low":    f"{n_low} plates ({n_low/total:.0%})",
        },
        "confidence_by_class": conf_by_cls,
        "highest_conf_class": conf_stats["mean"].idxmax(),
        "lowest_conf_class":  conf_stats["mean"].idxmin(),
        "risk_by_class": risk_by_cls,
        "heatmap_notes": heatmap_notes,
        "actionable_summary": [
            {
                "class": cls,
                "count": int(class_counts.get(cls, 0)),
                "pct": f"{class_counts.get(cls, 0)/total:.0%}",
                "risk": risk_by_cls.get(cls, ""),
                "avg_conf": f"{conf_stats.loc[cls, 'mean']:.1%}" if cls in conf_stats.index else "—",
                "action": ACTION_MAP.get(cls, ("Manual review", ""))[0],
            }
            for cls in class_counts.index
        ],
    }


def collect_manual_input():
    st.sidebar.markdown("---")
    _rb, _rm = st.sidebar.columns(2)
    with _rb:
        if st.button("🎲 Randomize", use_container_width=True, help="Fill with realistic random sensor values"):
            _lo, _hi = get_feature_ranges()
            for _c in ALL_RAW_COLS:
                if _c in ("TypeOfSteel_A300", "TypeOfSteel_A400"):
                    st.session_state[f"inp_{_c}"] = int(np.random.randint(0, 2))
                else:
                    st.session_state[f"inp_{_c}"] = float(
                        np.random.uniform(_lo.get(_c, 0), _hi.get(_c, 1))
                    )
            st.rerun()
    with _rm:
        if st.button("↺ Medians", use_container_width=True, help="Reset to training-set medians"):
            for _c in ALL_RAW_COLS:
                st.session_state.pop(f"inp_{_c}", None)
            st.rerun()

    st.sidebar.markdown("**Sensor Readings**")
    vals = {}
    for col in ALL_RAW_COLS:
        default = float(train_medians.get(col, 0.0))
        if col in ("TypeOfSteel_A300", "TypeOfSteel_A400"):
            vals[col] = st.sidebar.selectbox(col, [0, 1], index=int(default), key=f"inp_{col}")
        else:
            vals[col] = st.sidebar.number_input(col, value=default, format="%.4f", key=f"inp_{col}")
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

tab_predict, tab_dashboard, tab_chat = st.tabs(["🔍 Prediction", "📊 Analytics Dashboard", "🤖 QC Assistant"])

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

            # Save to session state for Analytics Dashboard
            st.session_state.batch_results = results_df
            st.session_state.batch_X_scaled = X_scaled_df
            st.session_state.batch_shap_computed = False
            st.session_state.batch_shap_importance = None
            st.session_state.batch_shap_values = None
            st.session_state.batch_X_sample = None

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

            # Save batch context for chatbot (includes full dashboard summary)
            st.session_state.prediction_context = {
                "batch":             True,
                "total_rows":        len(results_df),
                "avg_confidence":    results_df["Confidence"].mean(),
                "tier_summary":      tier_counts.to_dict(),
                "class_summary":     results_df["Predicted_Class"].value_counts().to_dict(),
                "dashboard_summary": build_dashboard_summary(results_df),
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
# TAB 2 — ANALYTICS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab_dashboard:
    st.subheader("📊 Batch Analytics Dashboard")

    _res = st.session_state.batch_results
    _Xs  = st.session_state.batch_X_scaled

    if _res is None:
        st.info("Upload a CSV in the **🔍 Prediction** tab to unlock the analytics dashboard.")
    else:
        total   = len(_res)
        n_high  = int((_res["Risk_Tier"] == "High").sum())
        n_med   = int((_res["Risk_Tier"] == "Medium").sum())
        n_low   = int((_res["Risk_Tier"] == "Low").sum())
        top_cls = _res["Predicted_Class"].mode()[0]
        avg_cf  = _res["Confidence"].mean()

        # ── KPI Row ──────────────────────────────────────────────────────────
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("📦 Total Plates", f"{total:,}")
        k2.metric("🔴 High Risk",    f"{n_high}",  f"{n_high/total:.0%}")
        k3.metric("🟠 Medium Risk",  f"{n_med}",   f"{n_med/total:.0%}")
        k4.metric("🟢 Low Risk",     f"{n_low}",   f"{n_low/total:.0%}")
        k5.metric("🏆 Top Defect",   top_cls,      f"Avg conf {avg_cf:.1%}")

        st.markdown("---")

        # ── Row 2: Defect distribution + Risk tiers ──────────────────────────
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### Defect Class Distribution")
            _cc = _res["Predicted_Class"].value_counts().reset_index()
            _cc.columns = ["Class", "Count"]
            fig_donut = px.pie(
                _cc, names="Class", values="Count",
                color="Class", color_discrete_map=CLASS_COLORS,
                hole=0.45,
            )
            fig_donut.update_traces(textposition="outside", textinfo="percent+label")
            fig_donut.update_layout(
                margin=dict(t=10, b=40, l=10, r=10),
                showlegend=False, height=360,
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        with col_b:
            st.markdown("#### Risk Tier Breakdown")
            _tier_df = pd.DataFrame({
                "Tier":  ["High", "Medium", "Low"],
                "Count": [n_high, n_med, n_low],
                "Pct":   [f"{n_high/total:.1%}", f"{n_med/total:.1%}", f"{n_low/total:.1%}"],
            })
            fig_risk = px.bar(
                _tier_df, x="Count", y="Tier", orientation="h",
                color="Tier",
                color_discrete_map={"High": "#e74c3c", "Medium": "#e67e22", "Low": "#27ae60"},
                text="Pct",
            )
            fig_risk.update_traces(textposition="outside")
            fig_risk.update_layout(
                showlegend=False, height=360,
                margin=dict(t=10, b=10, l=10, r=60),
                xaxis_title="Number of Plates", yaxis_title="",
            )
            st.plotly_chart(fig_risk, use_container_width=True)

        st.markdown("---")

        # ── Row 3: Confidence distribution by class ───────────────────────────
        st.markdown("#### Prediction Confidence by Defect Class")
        fig_box = px.box(
            _res, x="Predicted_Class", y="Confidence",
            color="Predicted_Class", color_discrete_map=CLASS_COLORS,
            points="outliers",
            category_orders={"Predicted_Class": list(CLASS_COLORS.keys())},
        )
        fig_box.update_layout(
            showlegend=False, height=380,
            margin=dict(t=10, b=10, l=10, r=10),
            xaxis_title="", yaxis_title="Confidence",
            yaxis_tickformat=".0%",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        fig_box.update_xaxes(showgrid=False)
        fig_box.update_yaxes(showgrid=True, gridcolor="#e0e0e0")
        st.plotly_chart(fig_box, use_container_width=True)

        st.markdown("---")

        # ── Row 4: Probability heatmap + SHAP global importance ───────────────
        col_c, col_d = st.columns(2)

        with col_c:
            st.markdown("#### Avg Class Probability per Predicted Defect")
            _prob_cols = [c for c in _res.columns if c.startswith("P(")]
            _cls_labels = [c[2:-1] for c in _prob_cols]
            _heat = _res.groupby("Predicted_Class")[_prob_cols].mean()
            _heat.columns = _cls_labels
            _heat = _heat.reindex(
                [c for c in CLASS_COLORS if c in _heat.index]
            )
            fig_heat = px.imshow(
                _heat, color_continuous_scale="Blues",
                aspect="auto", text_auto=".2f",
                zmin=0, zmax=1,
            )
            fig_heat.update_layout(
                height=380, margin=dict(t=10, b=10, l=10, r=10),
                xaxis_title="Class Probability", yaxis_title="Predicted As",
                coloraxis_colorbar=dict(title="Prob"),
            )
            st.plotly_chart(fig_heat, use_container_width=True)

        with col_d:
            st.markdown("#### SHAP Global Feature Importance (batch sample)")
            if not st.session_state.batch_shap_computed:
                with st.spinner("Computing SHAP values for batch sample…"):
                    try:
                        _n_sample = min(60, len(_Xs))
                        _X_samp = _Xs.iloc[:_n_sample]
                        _sv = explainer.shap_values(_X_samp)
                        if isinstance(_sv, np.ndarray) and _sv.ndim == 3:
                            _mean_abs = np.abs(_sv).mean(axis=(0, 2))
                        elif isinstance(_sv, list):
                            _mean_abs = np.abs(np.array(_sv)).mean(axis=(0, 1))
                        else:
                            _mean_abs = np.abs(_sv).mean(axis=0)
                        _imp_df = pd.DataFrame({
                            "Feature":    feat_names,
                            "Importance": _mean_abs,
                        }).sort_values("Importance", ascending=False)
                        st.session_state.batch_shap_importance = _imp_df
                        st.session_state.batch_shap_values = _sv
                        st.session_state.batch_X_sample = _X_samp
                        _top5 = _imp_df.head(5).apply(
                            lambda r: {"name": r["Feature"], "importance": round(r["Importance"], 4)},
                            axis=1,
                        ).tolist()
                        if "dashboard_summary" in st.session_state.prediction_context:
                            st.session_state.prediction_context["dashboard_summary"]["shap_top_features"] = _top5
                        st.session_state.batch_shap_computed = True
                    except Exception as _e:
                        st.warning(f"SHAP computation failed: {_e}")
                        st.session_state.batch_shap_computed = True

            if st.session_state.batch_shap_importance is not None:
                _plot_imp = st.session_state.batch_shap_importance.sort_values("Importance").tail(15)
                fig_imp = px.bar(
                    _plot_imp, x="Importance", y="Feature", orientation="h",
                    color="Importance", color_continuous_scale="Blues",
                )
                fig_imp.update_layout(
                    showlegend=False, height=380,
                    margin=dict(t=10, b=10, l=10, r=10),
                    xaxis_title="Mean |SHAP Value|", yaxis_title="",
                    coloraxis_showscale=False,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                fig_imp.update_xaxes(showgrid=True, gridcolor="#e0e0e0")
                st.plotly_chart(fig_imp, use_container_width=True)

        # ── SHAP Beeswarm (full-width, shown once SHAP is ready) ─────────────
        if st.session_state.batch_shap_values is not None:
            st.markdown("---")
            st.markdown("#### SHAP Beeswarm — Feature Impact Across Batch")
            st.caption(
                "Each dot = one plate. Position = SHAP impact on prediction. "
                "Color = feature value (red=high, blue=low). "
                "Width of spread shows how much a feature varies across plates."
            )
            try:
                _sv_bee  = st.session_state.batch_shap_values
                _Xb      = st.session_state.batch_X_sample

                # Convert 3-D array → list so shap handles multiclass correctly
                if isinstance(_sv_bee, np.ndarray) and _sv_bee.ndim == 3:
                    _sv_bee = [_sv_bee[:, :, k] for k in range(_sv_bee.shape[2])]

                fig_bee, _ = plt.subplots(figsize=(10, 7))
                shap.summary_plot(
                    _sv_bee, _Xb,
                    feature_names=feat_names,
                    class_names=list(rf.classes_),
                    max_display=15,
                    show=False,
                )
                plt.tight_layout()
                st.pyplot(fig_bee)
                plt.close(fig_bee)
            except Exception as _e:
                st.warning(f"Beeswarm plot failed: {_e}")

        st.markdown("---")

        # ── Row 5: Defect × Risk stacked bar ─────────────────────────────────
        st.markdown("#### Defect Class × Risk Tier")
        _cross = _res.groupby(["Predicted_Class", "Risk_Tier"]).size().reset_index(name="Count")
        fig_stack = px.bar(
            _cross, x="Predicted_Class", y="Count", color="Risk_Tier",
            color_discrete_map={"High": "#e74c3c", "Medium": "#e67e22", "Low": "#27ae60"},
            barmode="stack",
            category_orders={
                "Predicted_Class": list(CLASS_COLORS.keys()),
                "Risk_Tier": ["High", "Medium", "Low"],
            },
        )
        fig_stack.update_layout(
            height=360, margin=dict(t=10, b=10, l=10, r=10),
            xaxis_title="Defect Class", yaxis_title="Plate Count",
            legend_title="Risk Tier",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        fig_stack.update_yaxes(showgrid=True, gridcolor="#e0e0e0")
        st.plotly_chart(fig_stack, use_container_width=True)

        st.markdown("---")

        # ── Row 6: Actionable summary table ───────────────────────────────────
        st.markdown("#### Actionable Summary by Defect Class")
        _summ = _res.groupby("Predicted_Class").agg(
            Count=("Predicted_Class", "count"),
            High=("Risk_Tier", lambda x: (x == "High").sum()),
            Medium=("Risk_Tier", lambda x: (x == "Medium").sum()),
            Low=("Risk_Tier", lambda x: (x == "Low").sum()),
            Avg_Conf=("Confidence", "mean"),
        ).reset_index()
        _summ["% of Total"] = (_summ["Count"] / total * 100).round(1).astype(str) + "%"
        _summ["Avg Confidence"] = _summ["Avg_Conf"].map("{:.1%}".format)
        _summ["Action"] = _summ["Predicted_Class"].map(
            lambda c: ACTION_MAP.get(c, ("Manual review", ""))[0]
        )
        _summ = _summ.rename(columns={
            "Predicted_Class": "Defect Class",
            "High": "🔴 High",
            "Medium": "🟠 Medium",
            "Low": "🟢 Low",
        }).sort_values("Count", ascending=False).drop(columns=["Avg_Conf"])
        st.dataframe(
            _summ[["Defect Class", "Count", "% of Total", "🔴 High", "🟠 Medium", "🟢 Low", "Avg Confidence", "Action"]],
            use_container_width=True, hide_index=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — QC ASSISTANT CHATBOT
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

    pending = st.session_state.pop("pending_question", None)

    with st.form("chat_form", clear_on_submit=True):
        _ci, _cb = st.columns([5, 1])
        with _ci:
            _typed = st.text_input(
                "message",
                placeholder="Ask about the prediction, defect types, or QC procedures…",
                label_visibility="collapsed",
            )
        with _cb:
            _send = st.form_submit_button("Send ➤", use_container_width=True)

    question = pending or (_typed.strip() if _send and _typed.strip() else None)

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