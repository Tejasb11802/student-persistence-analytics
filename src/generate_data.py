"""
Generate a realistic synthetic learner dataset (7,032 records).

The data mimics a university student information system joined with
LMS engagement logs. Persistence is driven by academic progress,
engagement behavior, financial holds and support usage, plus noise.
Exactly 1,659 learners do not persist to the next term (at risk).
"""
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 11
N_STUDENTS = 7032
N_AT_RISK = 1659
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "learner_records.csv"


def generate(seed: int = SEED, noise: float = 1.04) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = N_STUDENTS

    age = np.clip(np.round(rng.gamma(2.2, 2.6, n) + 17), 17, 58).astype(int)
    gender = rng.choice(["Female", "Male", "Non-binary"], n, p=[0.52, 0.46, 0.02])
    first_gen = rng.binomial(1, 0.34, n)
    pell = rng.binomial(1, np.clip(0.28 + 0.18 * first_gen, 0, 1))
    residency = rng.choice(["In-State", "Out-of-State", "International"], n, p=[0.64, 0.26, 0.10])
    enroll = np.where(rng.random(n) < np.clip(0.18 + 0.02 * (age - 18), 0.1, 0.8), "Part-Time", "Full-Time")
    modality = rng.choice(["In-Person", "Online", "Hybrid"], n, p=[0.55, 0.25, 0.20])
    college = rng.choice(
        ["Engineering", "Business", "Liberal Arts", "Health Sciences", "Sciences", "Education"],
        n, p=[0.22, 0.20, 0.18, 0.15, 0.15, 0.10])
    housing = np.where((modality == "In-Person") & (age < 22) & (rng.random(n) < 0.6), "On-Campus", "Off-Campus")
    work_hours = np.clip(rng.normal(12 + 10 * (enroll == "Part-Time") + 4 * pell, 8, n), 0, 45).round()

    hs_gpa = np.clip(rng.normal(3.3, 0.42, n), 2.0, 4.0).round(2)
    credits_attempted = np.where(enroll == "Full-Time", rng.integers(12, 19, n), rng.integers(3, 12, n))

    # hidden motivation / engagement trait
    motivation = rng.normal(0, 1, n)

    logins_early = np.clip(rng.normal(10 + 2.5 * motivation - 1.5 * (modality == "In-Person"), 3, n), 0.5, None)
    trend = np.clip(rng.normal(-0.10 + 0.15 * motivation, 0.25, n), -0.95, 0.8)
    logins_late = np.clip(logins_early * (1 + trend) + rng.normal(0, 0.6, n), 0, None)
    logins_early, logins_late = logins_early.round(1), logins_late.round(1)
    session_min = np.clip(rng.normal(34 + 7 * motivation, 11, n), 3, None).round(1)
    submission_rate = np.clip(rng.normal(0.84 + 0.09 * motivation, 0.09, n), 0.05, 1.0).round(3)
    late_rate = np.clip(rng.normal(0.15 - 0.06 * motivation + 0.002 * work_hours, 0.08, n), 0, 0.9).round(3)
    forum_posts = rng.poisson(np.clip(4 + 2 * motivation + 3 * (modality == "Online"), 0.2, None))
    days_since_login = np.clip(rng.exponential(np.exp(1.4 - 0.55 * motivation)), 0, 60).round().astype(int)
    advising = rng.poisson(np.clip(1.6 + 0.4 * motivation + 0.5 * first_gen, 0.1, None))
    tutoring = rng.poisson(np.clip(1.2 + 0.5 * motivation, 0.1, None))

    term_gpa = np.clip(
        0.45 * hs_gpa + 1.45 + 0.34 * motivation + 0.06 * tutoring - 0.008 * work_hours + rng.normal(0, 0.34, n),
        0.0, 4.0).round(2)
    completion = np.clip(0.58 + 0.12 * term_gpa + 0.05 * motivation + rng.normal(0, 0.07, n), 0.1, 1.0)
    credits_earned = np.minimum(credits_attempted, np.round(credits_attempted * completion)).astype(int)
    financial_hold = rng.binomial(1, np.clip(0.07 + 0.08 * pell + 0.02 * (residency == "International"), 0, 1))
    scholarship = rng.binomial(1, np.clip(0.12 + 0.18 * (hs_gpa > 3.6), 0, 1))

    # low-signal operational fields (kept to mirror real exports)
    distance_miles = np.clip(rng.lognormal(2.0, 0.9, n), 0.2, 400).round(1)
    zip_prefix = rng.integers(850, 866, n)
    orientation_attended = rng.binomial(1, 0.78, n)
    email_open_rate = np.clip(rng.normal(0.55, 0.2, n), 0, 1).round(3)
    parking_permit = rng.binomial(1, 0.4, n)

    ratio = credits_earned / credits_attempted
    risk = (
        -2.6 * (ratio - 0.85)
        - 0.95 * (term_gpa - 2.8)
        - 3.2 * (submission_rate - 0.84)
        - 1.9 * ((logins_late - logins_early) / (logins_early + 1))
        + 0.045 * (days_since_login - 4)
        + 1.4 * late_rate
        + 0.95 * financial_hold
        - 0.20 * np.minimum(advising, 4)
        - 0.10 * np.minimum(tutoring, 4)
        + 0.35 * (enroll == "Part-Time")
        + 0.018 * work_hours
        + 0.18 * first_gen
        - 0.25 * scholarship
        - 0.18 * (housing == "On-Campus")
        + rng.normal(0, noise, n)
    )
    at_risk = np.zeros(n, dtype=int)
    at_risk[np.argsort(-risk)[:N_AT_RISK]] = 1

    return pd.DataFrame({
        "student_id": [f"S{100000 + i}" for i in range(n)],
        "age": age, "gender": gender, "first_generation": first_gen, "pell_eligible": pell,
        "residency": residency, "enrollment_status": enroll, "modality": modality,
        "college": college, "housing": housing, "work_hours_per_week": work_hours,
        "hs_gpa": hs_gpa, "term_gpa": term_gpa,
        "credits_attempted": credits_attempted, "credits_earned": credits_earned,
        "lms_logins_wk1_4": logins_early, "lms_logins_wk5_8": logins_late, "avg_session_minutes": session_min,
        "assignment_submission_rate": submission_rate, "late_submission_rate": late_rate,
        "forum_posts": forum_posts, "days_since_last_login": days_since_login,
        "advising_visits": advising, "tutoring_sessions": tutoring,
        "financial_hold": financial_hold, "scholarship": scholarship,
        "distance_from_campus_miles": distance_miles, "zip_prefix": zip_prefix,
        "orientation_attended": orientation_attended, "email_open_rate": email_open_rate,
        "parking_permit": parking_permit,
        "persisted_next_term": 1 - at_risk,
    })


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = generate()
    data.to_csv(OUT, index=False)
    print(f"Saved {len(data):,} learner records to {OUT}")
    print(f"Learners not persisting (at risk): {(data.persisted_next_term == 0).sum():,}")


if __name__ == "__main__":
    main()
