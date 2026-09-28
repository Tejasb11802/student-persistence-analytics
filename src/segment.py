"""
Score every learner and map at-risk learners into actionable segments.

- Out-of-fold Random Forest probabilities for all 7,032 learners (no leakage)
- Risk tiers: High / Elevated / Moderate / Low
- K-Means personas for the 1,659 at-risk learners using behavior,
  engagement and demographic factors, each paired with an intervention
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from config import RAW_CSV, PROCESSED, FIGURES, RANDOM_STATE
from features import engineer, OPTIMIZED_FEATURES
from train_model import make_model

SEGMENT_FEATURES = [
    "credit_completion_ratio", "term_gpa", "assignment_submission_rate", "late_submission_rate",
    "engagement_index", "login_trend", "days_since_last_login", "support_touchpoints",
    "financial_hold", "part_time", "work_hours_per_week", "first_generation",
]

PERSONAS = {
    "Disengaged / Fading Out": {"days_since_last_login": 1, "login_trend": -1, "engagement_index": -1},
    "Financial Barrier": {"financial_hold": 2},
    "Overextended Part-Time Workers": {"part_time": 1, "work_hours_per_week": 1},
    "Academically Struggling, Still Engaged": {"term_gpa": -1, "credit_completion_ratio": -1, "engagement_index": 1},
}

INTERVENTIONS = {
    "Disengaged / Fading Out": "Proactive outreach within 48 hours of 7+ days LMS inactivity; peer mentor check-in.",
    "Financial Barrier": "Financial aid counseling, emergency grant screening and hold resolution before registration.",
    "Overextended Part-Time Workers": "Flexible scheduling, online section options and time management coaching.",
    "Academically Struggling, Still Engaged": "Mandatory tutoring referral, early alert to instructors and academic recovery plan.",
}


def risk_tier(p: float) -> str:
    if p >= 0.70:
        return "High"
    if p >= 0.50:
        return "Elevated"
    if p >= 0.31:
        return "Moderate"
    return "Low"


def name_clusters(centroids_z: pd.DataFrame) -> dict:
    names = list(PERSONAS)
    score = np.zeros((len(centroids_z), len(names)))
    for j, name in enumerate(names):
        for feat, w in PERSONAS[name].items():
            score[:, j] += w * centroids_z[feat].values
    rows, cols = linear_sum_assignment(-score)
    return {int(r): names[c] for r, c in zip(rows, cols)}


def main():
    d = engineer(pd.read_csv(RAW_CSV))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    d["risk_score"] = cross_val_predict(make_model(), d[OPTIMIZED_FEATURES], d["at_risk"],
                                        cv=cv, method="predict_proba")[:, 1].round(4)
    d["risk_tier"] = d["risk_score"].apply(risk_tier)

    at_risk = d[d["at_risk"] == 1].copy()
    scaler = StandardScaler().fit(d[SEGMENT_FEATURES])
    Z = scaler.transform(at_risk[SEGMENT_FEATURES])
    km = KMeans(n_clusters=4, n_init=20, random_state=RANDOM_STATE).fit(Z)
    at_risk["cluster"] = km.labels_
    centroids_z = pd.DataFrame(km.cluster_centers_, columns=SEGMENT_FEATURES)
    mapping = name_clusters(centroids_z)
    at_risk["segment"] = at_risk["cluster"].map(mapping)
    at_risk["recommended_intervention"] = at_risk["segment"].map(INTERVENTIONS)

    d = d.merge(at_risk[["student_id", "segment", "recommended_intervention"]], on="student_id", how="left")
    d["segment"] = d["segment"].fillna("Persisting")
    d["recommended_intervention"] = d["recommended_intervention"].fillna("Standard advising cadence")
    d.to_csv(PROCESSED / "learners_scored.csv", index=False)

    profile_cols = SEGMENT_FEATURES + ["risk_score"]
    profiles = (d[d["at_risk"] == 1].groupby("segment")
                .agg(learners=("student_id", "count"), **{c: (c, "mean") for c in profile_cols})
                .round(3).sort_values("learners", ascending=False).reset_index())
    profiles["share_of_at_risk_pct"] = (100 * profiles["learners"] / profiles["learners"].sum()).round(1)
    profiles["recommended_intervention"] = profiles["segment"].map(INTERVENTIONS)
    profiles.to_csv(PROCESSED / "at_risk_segment_profiles.csv", index=False)

    demo_rows = []
    for col in ["gender", "residency", "enrollment_status", "modality", "college", "housing",
                "first_generation", "pell_eligible"]:
        g = d.groupby(col).agg(learners=("student_id", "count"), at_risk=("at_risk", "sum"),
                               avg_risk_score=("risk_score", "mean")).reset_index()
        g["at_risk_rate_pct"] = (100 * g["at_risk"] / g["learners"]).round(2)
        g.insert(0, "dimension", col)
        g = g.rename(columns={col: "group"})
        g["group"] = g["group"].astype(str)
        demo_rows.append(g)
    demo = pd.concat(demo_rows, ignore_index=True)
    demo["avg_risk_score"] = demo["avg_risk_score"].round(4)
    demo.to_csv(PROCESSED / "risk_by_demographic.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(profiles["segment"][::-1], profiles["learners"][::-1], color="#C0392B")
    for y, v in enumerate(profiles["learners"][::-1]):
        ax.text(v + 10, y, f"{v:,}", va="center")
    ax.set_title(f"{len(at_risk):,} At-Risk Learners Mapped to Intervention Segments", fontweight="bold")
    ax.set_xlabel("Learners")
    fig.tight_layout()
    fig.savefig(FIGURES / "at_risk_segments.png")
    plt.close(fig)

    print(f"Scored {len(d):,} learners. Risk tiers:")
    print(d["risk_tier"].value_counts().reindex(["High", "Elevated", "Moderate", "Low"]).to_string())
    print(f"\nMapped {len(at_risk):,} at-risk learners into segments:")
    print(profiles[["segment", "learners", "share_of_at_risk_pct", "term_gpa",
                    "days_since_last_login", "financial_hold", "part_time"]].to_string(index=False))


if __name__ == "__main__":
    main()
