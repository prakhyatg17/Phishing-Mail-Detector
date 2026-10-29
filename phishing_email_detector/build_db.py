"""
build_db.py – Load, clean, and write emails into phishing_emails.db.
Creates the `emails` table and empty result tables for train_models.py.
"""

import sqlite3
import sys
from common import DB_PATH, ensure_dirs
from preprocess import load_sources, build_clean_dataset


def create_tables(conn: sqlite3.Connection):
    """Create all tables (drop if they exist)."""
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS emails")
    cur.execute("""
        CREATE TABLE emails (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            source      TEXT    NOT NULL,
            subject     TEXT,
            body_trunc  TEXT,
            clean_text  TEXT    NOT NULL,
            label       INTEGER NOT NULL,
            split       TEXT
        )
    """)

    cur.execute("DROP TABLE IF EXISTS model_results")
    cur.execute("""
        CREATE TABLE model_results (
            model               TEXT PRIMARY KEY,
            accuracy            REAL,
            precision           REAL,
            recall              REAL,
            f1                  REAL,
            roc_auc             REAL,
            fnr                 REAL,
            train_seconds       REAL,
            confusion_matrix_json TEXT
        )
    """)

    cur.execute("DROP TABLE IF EXISTS source_results")
    cur.execute("""
        CREATE TABLE source_results (
            model   TEXT,
            source  TEXT,
            n       INTEGER,
            accuracy REAL,
            recall  REAL,
            PRIMARY KEY (model, source)
        )
    """)

    cur.execute("DROP TABLE IF EXISTS cross_source_results")
    cur.execute("""
        CREATE TABLE cross_source_results (
            held_out_source TEXT PRIMARY KEY,
            n               INTEGER,
            accuracy        REAL,
            recall          REAL,
            f1              REAL
        )
    """)

    cur.execute("DROP TABLE IF EXISTS roc_points")
    cur.execute("""
        CREATE TABLE roc_points (
            model    TEXT PRIMARY KEY,
            fpr_json TEXT,
            tpr_json TEXT
        )
    """)

    cur.execute("DROP TABLE IF EXISTS top_words")
    cur.execute("""
        CREATE TABLE top_words (
            word        TEXT PRIMARY KEY,
            coefficient REAL
        )
    """)

    cur.execute("DROP TABLE IF EXISTS predictions")
    cur.execute("""
        CREATE TABLE predictions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at      TEXT,
            subject         TEXT,
            text_preview    TEXT,
            model_probs_json TEXT,
            ensemble_prob   REAL,
            verdict         TEXT
        )
    """)

    conn.commit()


def main():
    ensure_dirs()

    print("=" * 60)
    print("Loading source CSVs ...")
    df = load_sources()

    print("\nCleaning dataset ...")
    df = build_clean_dataset(df)

    # Prepare body_trunc (first 5000 chars of body)
    df["body_trunc"] = df["body"].astype(str).str[:5000]

    # Write to SQLite
    print(f"\nWriting {len(df):,} rows to {DB_PATH} ...")
    conn = sqlite3.connect(str(DB_PATH))
    create_tables(conn)

    rows = df[["source", "subject", "body_trunc", "clean_text", "label"]].values.tolist()
    conn.executemany(
        "INSERT INTO emails (source, subject, body_trunc, clean_text, label) VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()

    # Verify
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM emails")
    total = cur.fetchone()[0]
    print(f"  emails table: {total:,} rows")

    cur.execute("SELECT source, COUNT(*), SUM(label) FROM emails GROUP BY source")
    for row in cur.fetchall():
        src, n, n_phish = row
        print(f"    {src:20s}  {n:>6,} rows   phishing: {n_phish:>5,}")

    for tbl in ["model_results", "source_results", "cross_source_results",
                "roc_points", "top_words", "predictions"]:
        cur.execute(f"SELECT COUNT(*) FROM {tbl}")
        print(f"  {tbl}: {cur.fetchone()[0]} rows (empty, ready)")

    conn.close()
    print("\nDone.  Database created at:", DB_PATH)


if __name__ == "__main__":
    main()
