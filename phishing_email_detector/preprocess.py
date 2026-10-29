"""
preprocess.py – single source of truth for text preprocessing.
Used by build_db.py, train_models.py, AND app.py (the Streamlit GUI).
"""

import html
import re
import pandas as pd
from common import SOURCE_FILES, SANITY_CSV

# ── Compiled regexes (compile once) ──────────────────────────────────
_RE_HTML_TAGS = re.compile(r"<[^>]+>")
_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_EMAIL = re.compile(r"\S+@\S+\.\S+")
_RE_DIGITS = re.compile(r"\d+")
_RE_NON_ALPHA = re.compile(r"[^a-z\s]")
_RE_WHITESPACE = re.compile(r"\s+")

MAX_RAW_LEN = 5_000


# ── load_sources ─────────────────────────────────────────────────────

def _read_csv_safe(path):
    """Read a CSV, falling back to latin-1 if UTF-8 fails."""
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def load_sources() -> pd.DataFrame:
    """
    Read the six source CSVs, add a `source` column, unify columns to
    [source, subject, body, label], and create a raw text field.
    """
    frames = []
    for name, path in SOURCE_FILES.items():
        df = _read_csv_safe(path)
        # Normalise column names to lowercase
        df.columns = [c.strip().lower() for c in df.columns]
        # Fill missing subject / body
        if "subject" not in df.columns:
            df["subject"] = ""
        if "body" not in df.columns:
            df["body"] = ""
        df["subject"] = df["subject"].fillna("")
        df["body"] = df["body"].fillna("")
        df["source"] = name
        # Keep only what we need
        df = df[["source", "subject", "body", "label"]].copy()
        frames.append(df)
        print(f"  {name:20s}  {len(df):>6,} rows   label dist: {dict(df['label'].value_counts().sort_index())}")

    merged = pd.concat(frames, ignore_index=True)
    print(f"\n  Total after merge: {len(merged):,} rows")

    # Sanity check against phishing_email.csv
    try:
        sanity = _read_csv_safe(SANITY_CSV)
        print(f"  Sanity CSV rows:   {len(sanity):,}  (expected 82,486)")
    except FileNotFoundError:
        print("  Sanity CSV not found – skipping check.")

    return merged


# ── clean_text ───────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Apply the full cleaning pipeline to a single text string.
    Steps:
      1. Truncate to 5,000 chars
      2. html.unescape + strip HTML tags
      3. Replace URLs → " urltoken ", emails → " emailtoken "
      4. Lowercase
      5. Replace digit sequences → " numtoken "
      6. Remove non-letter chars, collapse whitespace
      7. Drop tokens of length 1 or > 25
    """
    if not isinstance(text, str) or len(text) == 0:
        return ""
    # 1 – truncate
    text = text[:MAX_RAW_LEN]
    # 2 – html
    text = html.unescape(text)
    text = _RE_HTML_TAGS.sub(" ", text)
    # 3 – URLs & emails
    text = _RE_URL.sub(" urltoken ", text)
    text = _RE_EMAIL.sub(" emailtoken ", text)
    # 4 – lowercase
    text = text.lower()
    # 5 – digits
    text = _RE_DIGITS.sub(" numtoken ", text)
    # 6 – non-alpha
    text = _RE_NON_ALPHA.sub(" ", text)
    text = _RE_WHITESPACE.sub(" ", text).strip()
    # 7 – token length filter
    tokens = [t for t in text.split() if 2 <= len(t) <= 25]
    return " ".join(tokens)


# ── Dataset cleaning ────────────────────────────────────────────────

def build_clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply clean_text, drop short rows and duplicates, report stats.
    Expects columns: source, subject, body, label.
    Returns df with added 'clean_text' column.
    """
    n0 = len(df)
    # Create raw text
    df = df.copy()
    df["raw_text"] = (df["subject"].astype(str) + " " + df["body"].astype(str))
    # Clean
    print("\n  Cleaning texts ...")
    df["clean_text"] = df["raw_text"].apply(clean_text)

    # Drop rows with < 3 tokens
    df["_ntokens"] = df["clean_text"].apply(lambda t: len(t.split()))
    short_mask = df["_ntokens"] < 3
    n_short = short_mask.sum()
    df = df[~short_mask].copy()
    print(f"  Dropped {n_short:,} rows with < 3 tokens  ({n0} -> {len(df)})")

    # Drop duplicate clean_text (keep first)
    n_before_dedup = len(df)
    df = df.drop_duplicates(subset="clean_text", keep="first")
    n_dup = n_before_dedup - len(df)
    print(f"  Dropped {n_dup:,} duplicate texts         ({n_before_dedup} -> {len(df)})")

    df.drop(columns=["_ntokens", "raw_text"], inplace=True)

    # Final class balance per source
    print("\n  Final class balance per source:")
    for src in df["source"].unique():
        sub = df[df["source"] == src]
        dist = dict(sub["label"].value_counts().sort_index())
        print(f"    {src:20s}  {len(sub):>6,} rows   {dist}")
    print(f"\n  Grand total: {len(df):,} rows")
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("Loading sources ...")
    df = load_sources()
    print("\nBuilding clean dataset ...")
    df = build_clean_dataset(df)
    print("Done.")
