"""
Steel Plates IDSS — Groq-powered Chatbot
Provides conversational explanations of predictions, SHAP values, and defect knowledge.
"""

from openai import OpenAI
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the prototype/ directory
load_dotenv(Path(__file__).parent / ".env")

# ── Groq client ───────────────────────────────────────────────────────────────
def get_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set. Add it to prototype/.env")
    return OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert Quality Control AI assistant for a steel manufacturing plant.
You help operators understand defect predictions made by the Steel Plates Defect IDSS (Intelligent Decision Support System).

You know about these 7 defect types and their factory actions:
- Pastry: Reject — send to scrap line
- Z_Scratch: Reject — deep scratch, unusable
- K_Scatch: Reject — cross-direction scratch
- Stains: Hold — chemical cleaning possible
- Dirtiness: Hold — surface cleaning + re-inspect
- Bumps: Conditional accept — check tolerance spec
- Other_Faults: Human review — ambiguous category

Risk tiers:
- High (>=70% confidence): Auto pass/reject
- Medium (40-70%): Human review recommended (60 seconds)
- Low (<40%): Escalate to senior QC

Key features that drive predictions:
- Length_of_Conveyer: process-stage indicator
- Defect_Area_Ratio: spread vs localised defect
- Log_Pixels_Areas: scale-invariant defect size
- Luminosity_Range: scratch sharpness vs stain diffuseness
- Steel_Plate_Thickness: plate grade and rolling process

Analytics Dashboard panels (visible when a batch CSV is uploaded):
1. KPI Cards — top row: total plates, High/Medium/Low risk counts with %, top defect class, avg confidence.
2. Defect Class Distribution (donut chart) — percentage breakdown of which defect types were predicted across the batch.
3. Risk Tier Breakdown (horizontal bars) — Red=High, Orange=Medium, Green=Low; shows how many plates fall in each risk tier.
4. Confidence by Defect Class (box plot) — distribution of the model's top-class probability per defect. A higher box median = the model is more certain about that defect type. Wider spread = more variable confidence.
5. Avg Class Probability Heatmap — rows = what the model predicted, columns = probability for every class. The diagonal should be bright (high self-probability). Off-diagonal brightness reveals which classes the model confuses.
6. SHAP Global Feature Importance (bar chart) — mean |SHAP value| across a sample of the batch; shows which sensor readings drive predictions most overall.
7. Defect × Risk Stacked Bar — for each defect class, how many plates are High/Medium/Low risk; reveals which defect types are most reliably or unreliably detected.
8. Actionable Summary Table — per-defect row with count, % of total, risk tier breakdown, avg confidence, and recommended factory action.

When asked about any dashboard panel, explain what the panel shows conceptually AND reference the actual numbers from the batch context provided.

Business recommendations:
1. Auto pass/reject on High-tier plates (>=0.70 confidence)
2. Monthly calibration of Length_of_Conveyer sensors
3. Safety-net routing for Other_Faults predictions
4. Monthly retraining with active learning on Medium-tier plates
5. Camera-cleaning alert on consecutive Stains/Dirtiness predictions

Model: Stacking Ensemble (Random Forest + SVM + Logistic Regression meta-learner)
Overall Macro AUC: 0.9545 | Accuracy: 76.4%
High-confidence automation coverage: 65.8% of plates at 88.3% accuracy

Keep answers concise, practical, and factory-floor friendly.
When given prediction context, reference the specific values in your explanation.
If asked something outside your domain, say so honestly."""


def build_context_message(prediction_context: dict) -> str:
    """Build a context string from the current prediction state."""
    if not prediction_context:
        return ""

    lines = ["Current prediction context:"]

    if prediction_context.get("predicted_class"):
        lines.append(f"- Predicted defect: {prediction_context['predicted_class']}")
    if prediction_context.get("confidence"):
        lines.append(f"- Confidence: {prediction_context['confidence']:.1%}")
    if prediction_context.get("risk_tier"):
        lines.append(f"- Risk tier: {prediction_context['risk_tier']}")
    if prediction_context.get("action"):
        lines.append(f"- Recommended action: {prediction_context['action']}")
    if prediction_context.get("top_features"):
        lines.append("- Top SHAP feature contributions:")
        for feat in prediction_context["top_features"]:
            direction = "up" if feat["shap"] > 0 else "down"
            lines.append(
                f"    {direction} {feat['name']}: SHAP={feat['shap']:+.4f}, value={feat['value']:.4f}"
            )
    if prediction_context.get("all_probs"):
        lines.append("- All class probabilities:")
        for cls, prob in prediction_context["all_probs"].items():
            lines.append(f"    {cls}: {prob:.3f}")

    # Full analytics dashboard context (batch mode)
    ds = prediction_context.get("dashboard_summary") if prediction_context.get("batch") else None
    if ds:
        lines.append("\n=== Analytics Dashboard — current batch data ===")
        if ds.get("kpi"):
            lines.append(f"[KPI Cards] {ds['kpi']}")
        if ds.get("defect_distribution"):
            lines.append("[Defect Donut] Distribution across batch:")
            for cls, info in ds["defect_distribution"].items():
                lines.append(f"    {cls}: {info}")
        if ds.get("risk_tier_breakdown"):
            lines.append("[Risk Tier Bars]")
            for tier, info in ds["risk_tier_breakdown"].items():
                lines.append(f"    {tier}: {info}")
        if ds.get("confidence_by_class"):
            lines.append(f"[Confidence Box Plot] (highest confidence: {ds.get('highest_conf_class')}, lowest: {ds.get('lowest_conf_class')})")
            for cls, info in ds["confidence_by_class"].items():
                lines.append(f"    {cls}: {info}")
        if ds.get("heatmap_notes"):
            lines.append("[Probability Heatmap] Avg probabilities per predicted class:")
            lines.extend(ds["heatmap_notes"])
        if ds.get("shap_top_features"):
            lines.append("[SHAP Feature Importance] Top drivers across batch:")
            for feat in ds["shap_top_features"]:
                lines.append(f"    {feat['name']}: mean |SHAP|={feat['importance']:.4f}")
        if ds.get("risk_by_class"):
            lines.append("[Defect × Risk Stacked Bar]")
            for cls, info in ds["risk_by_class"].items():
                lines.append(f"    {cls}: {info}")
        if ds.get("actionable_summary"):
            lines.append("[Actionable Summary Table]")
            for row in ds["actionable_summary"]:
                lines.append(
                    f"    {row['class']}: {row['count']} plates ({row['pct']}), "
                    f"risk={row['risk']}, avg_conf={row['avg_conf']}, action={row['action']}"
                )

    return "\n".join(lines)


def chat(
    user_message: str,
    history: list,
    prediction_context: dict = None,
    model: str = "llama-3.3-70b-versatile",
) -> tuple[str, list]:
    """
    Send a message to Groq and return the response + updated history.

    Args:
        user_message: The user's input text
        history: List of {"role": ..., "content": ...} dicts (conversation so far)
        prediction_context: Dict with current prediction info to inject as context
        model: Groq model name

    Returns:
        (assistant_reply, updated_history)
    """
    client = get_client()

    # Build messages list
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Inject prediction context as a system-level note if available
    ctx = build_context_message(prediction_context or {})
    if ctx:
        messages.append({
            "role": "system",
            "content": ctx,
        })

    # Add conversation history
    messages.extend(history)

    # Add current user message
    messages.append({"role": "user", "content": user_message})

    # Call Groq API
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=600,
        temperature=0.4,
    )

    reply = response.choices[0].message.content.strip()

    # Update history
    updated_history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": reply},
    ]

    return reply, updated_history


def get_suggested_questions(predicted_class: str = None) -> list[str]:
    """Return context-aware suggested questions for the user."""
    base = [
        "What does this defect mean for the plate?",
        "Why is this prediction uncertain?",
        "What are the most important features for this prediction?",
        "When should I escalate to senior QC?",
        "How accurate is the model for this defect type?",
    ]

    class_specific = {
        "Other_Faults": [
            "Why is Other_Faults hard to classify?",
            "What sub-types could Other_Faults represent?",
        ],
        "Bumps": [
            "When can a Bumps plate be accepted?",
            "What tolerance spec should I check for Bumps?",
        ],
        "Stains": [
            "What chemical cleaning process works for Stains?",
            "Could consecutive Stains mean a camera issue?",
        ],
        "Dirtiness": [
            "How do I re-inspect after cleaning a Dirtiness plate?",
            "Could consecutive Dirtiness mean a camera issue?",
        ],
        "Z_Scratch": ["Why is Z_Scratch always rejected?"],
        "K_Scatch": ["What causes cross-direction scratches?"],
        "Pastry": ["What happens to scrapped Pastry plates?"],
    }

    extras = class_specific.get(predicted_class, [])
    return (extras + base)[:5]