"""Write clean, Tableau-ready extracts to the tableau/ folder."""
import shutil
import pandas as pd
from config import PROCESSED, REPORTS, TABLEAU


def main():
    scored = pd.read_csv(PROCESSED / "learners_scored.csv")
    keep = ["student_id", "age", "gender", "first_generation", "pell_eligible", "residency",
            "enrollment_status", "modality", "college", "housing", "work_hours_per_week",
            "term_gpa", "credit_completion_ratio", "assignment_submission_rate", "late_submission_rate",
            "lms_logins_wk1_4", "lms_logins_wk5_8", "login_trend", "engagement_index",
            "days_since_last_login", "advising_visits", "tutoring_sessions", "support_touchpoints",
            "financial_hold", "at_risk", "risk_score", "risk_tier", "segment", "recommended_intervention"]
    scored[keep].round(4).to_csv(TABLEAU / "learner_risk_scores.csv", index=False)

    copies = {
        PROCESSED / "at_risk_segment_profiles.csv": "segment_profiles.csv",
        PROCESSED / "risk_by_demographic.csv": "risk_by_demographic.csv",
        PROCESSED / "sql_inactivity_bands.csv": "inactivity_bands.csv",
        REPORTS / "feature_importance.csv": "feature_importance.csv",
        REPORTS / "model_comparison.csv": "model_comparison.csv",
    }
    for src, name in copies.items():
        shutil.copy(src, TABLEAU / name)
    print(f"Tableau extracts written to {TABLEAU}")


if __name__ == "__main__":
    main()
