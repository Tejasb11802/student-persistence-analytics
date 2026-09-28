"""Shared paths and settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "learner_records.csv"
PROCESSED = ROOT / "data" / "processed"
DB_PATH = ROOT / "data" / "learners.db"
SQL_DIR = ROOT / "sql"
FIGURES = ROOT / "reports" / "figures"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
TABLEAU = ROOT / "tableau"

RANDOM_STATE = 42
TARGET = "at_risk"
TARGET_RECALL = 0.90  # early warning: catch at least 90% of at-risk learners

for folder in (PROCESSED, FIGURES, MODELS, TABLEAU):
    folder.mkdir(parents=True, exist_ok=True)
