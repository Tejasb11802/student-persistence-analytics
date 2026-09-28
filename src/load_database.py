"""Load learner records into SQLite and run the SQL analysis queries."""
import re
import sqlite3
import pandas as pd
from config import RAW_CSV, DB_PATH, SQL_DIR, PROCESSED


def load() -> None:
    df = pd.read_csv(RAW_CSV)
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript((SQL_DIR / "schema.sql").read_text())
        df.to_sql("learners", conn, if_exists="append", index=False)
        count = conn.execute("SELECT COUNT(*) FROM learners").fetchone()[0]
    print(f"Loaded {count:,} rows into {DB_PATH.name}")


def run_queries() -> dict:
    sql_text = (SQL_DIR / "analysis_queries.sql").read_text()
    blocks = re.split(r"--\s*name:\s*", sql_text)[1:]
    results = {}
    with sqlite3.connect(DB_PATH) as conn:
        for block in blocks:
            name, query = block.split("\n", 1)
            name = name.strip()
            result = pd.read_sql_query(query, conn)
            result.to_csv(PROCESSED / f"sql_{name}.csv", index=False)
            results[name] = result
    print(f"Ran {len(results)} SQL queries, outputs saved to data/processed/")
    print(results["overall_persistence"].to_string(index=False))
    return results


def main():
    load()
    run_queries()


if __name__ == "__main__":
    main()
