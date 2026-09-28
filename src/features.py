"""Feature engineering and feature sets for the persistence model."""
import pandas as pd

ID_COLS = ["student_id"]
RAW_TARGET = "persisted_next_term"

# Engineered features selected through EDA (see eda.py)
OPTIMIZED_FEATURES = [
    "credit_completion_ratio",
    "term_gpa",
    "academic_momentum",
    "assignment_submission_rate",
    "late_submission_rate",
    "lms_logins_wk5_8",
    "login_trend",
    "days_since_last_login",
    "engagement_index",
    "financial_hold",
    "advising_visits",
    "tutoring_sessions",
    "support_touchpoints",
    "part_time",
    "work_hours_per_week",
    "first_generation",
    "scholarship",
    "on_campus",
]


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["at_risk"] = 1 - d[RAW_TARGET]
    d["credit_completion_ratio"] = d["credits_earned"] / d["credits_attempted"]
    d["academic_momentum"] = d["term_gpa"] - d["hs_gpa"]
    d["login_trend"] = (d["lms_logins_wk5_8"] - d["lms_logins_wk1_4"]) / (d["lms_logins_wk1_4"] + 1)
    d["engagement_index"] = (
        d["lms_logins_wk5_8"].rank(pct=True)
        + d["avg_session_minutes"].rank(pct=True)
        + d["assignment_submission_rate"].rank(pct=True)
        + d["forum_posts"].rank(pct=True)
    ) / 4
    d["support_touchpoints"] = d["advising_visits"] + d["tutoring_sessions"]
    d["inactive_14d"] = (d["days_since_last_login"] > 14).astype(int)
    d["part_time"] = (d["enrollment_status"] == "Part-Time").astype(int)
    d["on_campus"] = (d["housing"] == "On-Campus").astype(int)
    return d


def baseline_matrix(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Baseline: every raw column as exported, one-hot encoded, no engineering."""
    raw = raw_df.drop(columns=ID_COLS + [RAW_TARGET], errors="ignore")
    return pd.get_dummies(raw, drop_first=True, dtype=int)
