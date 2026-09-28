"""
Score new learners with the trained model.

Usage (from project root):
    python src/predict.py path\\to\\new_learners.csv
The input CSV must use the same columns as data/raw/learner_records.csv
(persisted_next_term is optional). Output: reports/new_learner_predictions.csv
"""
import sys
import joblib
import pandas as pd
from config import MODELS, REPORTS
from features import engineer
from segment import risk_tier


def main(path: str) -> None:
    bundle = joblib.load(MODELS / "rf_persistence_model.joblib")
    df = pd.read_csv(path)
    if "persisted_next_term" not in df.columns:
        df["persisted_next_term"] = 1
    d = engineer(df)
    d["risk_score"] = bundle["model"].predict_proba(d[bundle["features"]])[:, 1].round(4)
    d["flagged_at_risk"] = (d["risk_score"] >= bundle["threshold"]).astype(int)
    d["risk_tier"] = d["risk_score"].apply(risk_tier)
    out = REPORTS / "new_learner_predictions.csv"
    d[["student_id", "risk_score", "risk_tier", "flagged_at_risk"]].to_csv(out, index=False)
    print(f"Scored {len(d):,} learners, flagged {d.flagged_at_risk.sum():,} at risk. Saved to {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python src/predict.py path\\to\\new_learners.csv")
    main(sys.argv[1])
