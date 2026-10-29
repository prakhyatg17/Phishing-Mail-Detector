"""
common.py – shared paths, DB helpers, and read-only connection for the
Phishing Email Detector project.
"""

import os
import sqlite3
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"
DB_PATH = PROJECT_DIR / "phishing_emails.db"

# ── CSV source files ──────────────────────────────────────────────────
SOURCE_FILES = {
    "CEAS_08":        DATA_DIR / "CEAS_08.csv",
    "Enron":          DATA_DIR / "Enron.csv",
    "Ling":           DATA_DIR / "Ling.csv",
    "Nazario":        DATA_DIR / "Nazario.csv",
    "Nigerian_Fraud": DATA_DIR / "Nigerian_Fraud.csv",
    "SpamAssasin":    DATA_DIR / "SpamAssasin.csv",
}

SANITY_CSV = DATA_DIR / "phishing_email.csv"

# ── DB helpers ────────────────────────────────────────────────────────

def get_db_connection(readonly: bool = False) -> sqlite3.Connection:
    """Return a SQLite connection.  Pass readonly=True for the GUI."""
    if readonly:
        uri = f"file:{DB_PATH}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
    else:
        conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def ensure_dirs():
    """Create data/ and models/ directories if missing."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

