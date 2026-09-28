"""Build a standalone interactive HTML dashboard (preview of the Tableau dashboard)."""
import json
import pandas as pd
import plotly.express as px
from config import PROCESSED, REPORTS, TABLEAU

RED, GREY = "#C0392B", "#7F8C8D"
TIER_COLORS = {"High": "#922B21", "Elevated": "#E74C3C", "Moderate": "#F5B041", "Low": "#58D68D"}


def styled(fig, height=380):
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=20, r=20, t=50, b=20),
                      font=dict(family="Segoe UI, Arial", size=12), title_font=dict(size=15))
    return fig


def main():
    scored = pd.read_csv(TABLEAU / "learner_risk_scores.csv")
    profiles = pd.read_csv(TABLEAU / "segment_profiles.csv")
    importance = pd.read_csv(TABLEAU / "feature_importance.csv")
    demo = pd.read_csv(TABLEAU / "risk_by_demographic.csv")
    inactivity = pd.read_csv(TABLEAU / "inactivity_bands.csv")
    metrics = json.loads((REPORTS / "model_metrics.json").read_text())
    m = metrics["test_metrics"]

    tiers = scored["risk_tier"].value_counts().reindex(["High", "Elevated", "Moderate", "Low"]).reset_index()
    tiers.columns = ["risk_tier", "learners"]
    figs = [
        px.bar(tiers, x="risk_tier", y="learners", color="risk_tier", color_discrete_map=TIER_COLORS,
               text="learners", title="Learners by Predicted Risk Tier"),
        px.bar(profiles.sort_values("learners"), x="learners", y="segment", orientation="h", text="learners",
               color_discrete_sequence=[RED], hover_data=["recommended_intervention"],
               title="At-Risk Learners by Intervention Segment"),
        px.bar(importance.head(12).sort_values("importance"), x="importance", y="feature", orientation="h",
               color_discrete_sequence=[RED], title="Top Predictive Features (Random Forest)"),
        px.bar(inactivity, x="inactivity_band", y="at_risk_rate_pct", text="at_risk_rate_pct",
               color_discrete_sequence=[RED], title="At-Risk Rate by LMS Inactivity"),
        px.bar(demo[demo.dimension == "college"].sort_values("at_risk_rate_pct", ascending=False),
               x="group", y="at_risk_rate_pct", text="at_risk_rate_pct", color_discrete_sequence=[GREY],
               title="At-Risk Rate by College (%)"),
        px.bar(demo[demo.dimension.isin(["enrollment_status", "modality", "residency"])],
               x="group", y="at_risk_rate_pct", color="dimension", text="at_risk_rate_pct",
               title="At-Risk Rate by Enrollment, Modality and Residency (%)"),
        px.scatter(scored.sample(2500, random_state=1), x="engagement_index", y="term_gpa", color="risk_tier",
                   color_discrete_map=TIER_COLORS, opacity=0.6,
                   title="Engagement vs Term GPA (sample of 2,500 learners)"),
        px.histogram(scored, x="risk_score", color="at_risk", nbins=40, barmode="overlay",
                     color_discrete_map={0: "#58D68D", 1: RED}, title="Risk Score Distribution by Actual Outcome"),
    ]
    for f in figs:
        styled(f)
        f.update_layout(showlegend=f.layout.title.text.startswith(("Engagement", "Risk Score", "At-Risk Rate by Enr")))

    kpis = [
        ("Learners Analyzed", f"{len(scored):,}"),
        ("At-Risk Learners", f"{int(scored.at_risk.sum()):,}"),
        ("Model Accuracy", f"{m['accuracy']:.1%}"),
        ("ROC AUC", f"{m['roc_auc']:.2f}"),
        ("Recall (At Risk)", f"{m['recall']:.1%}"),
        ("Precision Lift from EDA", f"+{metrics['precision_lift_pts']:.1f} pts"),
    ]
    kpi_html = "".join(f'<div class="kpi"><div class="v">{v}</div><div class="l">{l}</div></div>' for l, v in kpis)
    charts = "".join(f'<div class="card">{f.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False)}</div>'
                     for i, f in enumerate(figs))
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Student Persistence & At-Risk Analytics</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;background:#F4F6F7;margin:0;color:#1C2833}}
header{{background:#1C2833;color:#fff;padding:22px 32px}} header h1{{margin:0;font-size:24px}}
header p{{margin:6px 0 0;color:#AEB6BF}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:14px;padding:20px 32px}}
.kpi{{background:#fff;border-left:5px solid {RED};padding:14px 16px;border-radius:6px;box-shadow:0 1px 3px #0001}}
.kpi .v{{font-size:26px;font-weight:700}} .kpi .l{{font-size:12px;color:#5D6D7E;text-transform:uppercase}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:16px;padding:0 32px 32px}}
.card{{background:#fff;border-radius:6px;padding:8px;box-shadow:0 1px 3px #0001;overflow-x:auto}}
</style></head><body>
<header><h1>Student Persistence & At-Risk Analytics</h1>
<p>Random Forest early-warning model and intervention segmentation across {len(scored):,} learner records</p></header>
<div class="kpis">{kpi_html}</div><div class="grid">{charts}</div></body></html>"""
    out = REPORTS / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    print(f"Dashboard saved to {out}")


if __name__ == "__main__":
    main()
