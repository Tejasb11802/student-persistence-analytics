"""Run the full Student Persistence & At-Risk Analytics pipeline."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import generate_data, load_database, eda, train_model, segment, export_tableau, build_dashboard  # noqa: E402

STEPS = [
    ("1/7 Generating learner data", generate_data.main),
    ("2/7 Loading SQLite database and running SQL analysis", load_database.main),
    ("3/7 Exploratory data analysis", eda.main),
    ("4/7 Training Random Forest models", train_model.main),
    ("5/7 Scoring learners and mapping at-risk segments", segment.main),
    ("6/7 Exporting Tableau extracts", export_tableau.main),
    ("7/7 Building HTML dashboard", build_dashboard.main),
]

if __name__ == "__main__":
    skip_generate = "--skip-generate" in sys.argv  # use when running on your own data
    for title, step in STEPS:
        if skip_generate and step is generate_data.main:
            print(f"Skipping: {title}")
            continue
        print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
        step()
    print("\nPipeline complete. Open reports/dashboard.html in your browser.")
