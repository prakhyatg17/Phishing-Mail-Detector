"""
app.py – Streamlit GUI for the Phishing Email Detector.
Run with:  streamlit run app.py
"""

import datetime
import email
import json
import re
import sqlite3
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from common import DB_PATH, MODELS_DIR
from preprocess import clean_text

# ─────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Phishing Email Detector", page_icon="🛡️", layout="wide")

# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    """Load TF-IDF vectorizer and all trained models."""
    tfidf = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
    model_files = {
        "Logistic Regression": "logistic_regression.joblib",
        "SVM (LinearSVC)": "svm_linearsvc.joblib",
        "Random Forest": "random_forest.joblib",
        "AdaBoost": "adaboost.joblib",
        "Ensemble (Soft Voting)": "ensemble_soft_voting.joblib",
    }
    models = {}
    for name, fname in model_files.items():
        p = MODELS_DIR / fname
        if p.exists():
            models[name] = joblib.load(p)
    return tfidf, models


def get_ro_connection():
    uri = f"file:{DB_PATH}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def get_rw_connection():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


# ─────────────────────────────────────────────────────────────────────
# Pre-flight checks
# ─────────────────────────────────────────────────────────────────────
def preflight():
    ok = True
    if not DB_PATH.exists():
        st.error(f"Database not found at `{DB_PATH}`.\n\nRun: `python build_db.py`")
        ok = False
    if not (MODELS_DIR / "tfidf_vectorizer.joblib").exists():
        st.error(f"Models not found in `{MODELS_DIR}`.\n\nRun: `python train_models.py`")
        ok = False
    if not ok:
        st.stop()


# ─────────────────────────────────────────────────────────────────────
# Example emails
# ─────────────────────────────────────────────────────────────────────
EXAMPLE_PHISHING = {
    "subject": "URGENT: Your Account Has Been Suspended",
    "body": (
        "Dear Valued Customer,\n\n"
        "We have detected suspicious activity on your account. Your account has been "
        "temporarily suspended. To restore access, please verify your identity immediately "
        "by clicking the link below:\n\n"
        "http://secure-banking-verify.com/login?id=839201\n\n"
        "If you do not verify within 24 hours, your account will be permanently closed "
        "and all funds will be frozen.\n\n"
        "Sincerely,\nBank Security Team"
    ),
}

EXAMPLE_LEGIT = {
    "subject": "Team meeting notes - October 28",
    "body": (
        "Hi everyone,\n\n"
        "Here are the key takeaways from today's standup:\n\n"
        "1. The new dashboard feature is on track for next sprint.\n"
        "2. QA found two minor bugs in the login flow; Jake is fixing them.\n"
        "3. We'll demo to stakeholders on Friday at 3 PM.\n\n"
        "Let me know if I missed anything.\n\n"
        "Best,\nSarah"
    ),
}


# ─────────────────────────────────────────────────────────────────────
# Red flags heuristics
# ─────────────────────────────────────────────────────────────────────
def compute_red_flags(raw_text: str) -> list[dict]:
    """Return a list of {flag, detail} dicts for heuristic red flags."""
    flags = []
    lower = raw_text.lower()

    # URLs
    urls = re.findall(r"https?://\S+|www\.\S+", raw_text)
    if urls:
        flags.append({"flag": "🔗 URLs detected", "detail": f"{len(urls)} URL(s) found"})

    # Urgency
    urgency = [w for w in ["urgent", "verify", "suspended", "immediately",
                            "expire", "locked", "confirm", "warning"]
               if w in lower]
    if urgency:
        flags.append({"flag": "⚠️ Urgency words", "detail": ", ".join(urgency)})

    # Money / prize
    money = [w for w in ["money", "prize", "winner", "million", "dollar",
                          "congratulations", "lottery", "reward", "inheritance"]
             if w in lower]
    if money:
        flags.append({"flag": "💰 Money / prize words", "detail": ", ".join(money)})

    # ALL-CAPS ratio
    alpha_chars = [c for c in raw_text if c.isalpha()]
    if alpha_chars:
        caps_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        if caps_ratio > 0.3:
            flags.append({"flag": "🔠 High CAPS ratio",
                          "detail": f"{caps_ratio:.0%} uppercase letters"})

    # Mismatch hints (displayed URL vs actual)
    mismatches = re.findall(r'<a[^>]*href=["\']([^"\']+)["\'][^>]*>([^<]+)</a>', raw_text, re.I)
    for href, text in mismatches:
        if "http" in text.lower() and href not in text:
            flags.append({"flag": "🎭 Link mismatch",
                          "detail": f"Display: {text[:50]} → Actual: {href[:50]}"})

    return flags


# ─────────────────────────────────────────────────────────────────────
# Top contributing words
# ─────────────────────────────────────────────────────────────────────
def get_top_contributing_words(text_cleaned, tfidf, lr_model, n=10):
    """Return top words pushing prediction toward phishing."""
    vec = tfidf.transform([text_cleaned])
    feature_names = tfidf.get_feature_names_out()
    coefs = lr_model.coef_[0]

    # For each non-zero feature, compute coef * tfidf_value
    indices = vec.nonzero()[1]
    contributions = []
    for idx in indices:
        score = coefs[idx] * vec[0, idx]
        contributions.append((feature_names[idx], float(score), float(coefs[idx])))

    contributions.sort(key=lambda x: x[1], reverse=True)
    return contributions[:n]


# ═════════════════════════════════════════════════════════════════════
# PAGES
# ═════════════════════════════════════════════════════════════════════

def page_detect():
    st.header("🔍 Detect Phishing Email")

    tfidf, models = load_models()

    if "det_subject" not in st.session_state:
        st.session_state["det_subject"] = ""
    if "det_body" not in st.session_state:
        st.session_state["det_body"] = ""

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📧 Example: Phishing"):
            st.session_state["det_subject"] = EXAMPLE_PHISHING["subject"]
            st.session_state["det_body"] = EXAMPLE_PHISHING["body"]
            st.rerun()
    with col2:
        if st.button("✉️ Example: Legitimate"):
            st.session_state["det_subject"] = EXAMPLE_LEGIT["subject"]
            st.session_state["det_body"] = EXAMPLE_LEGIT["body"]
            st.rerun()

    subject = st.text_input("Subject", key="det_subject")
    body = st.text_area("Body", height=200, key="det_body")

    # File upload
    uploaded = st.file_uploader("Or upload a .txt / .eml file", type=["txt", "eml"])
    if uploaded is not None:
        raw_bytes = uploaded.read()
        if uploaded.name.endswith(".eml"):
            msg = email.message_from_bytes(raw_bytes)
            st.session_state["det_subject"] = msg.get("Subject", "")
            payload = msg.get_payload(decode=True)
            if payload:
                st.session_state["det_body"] = payload.decode("utf-8", errors="replace")
            else:
                st.session_state["det_body"] = str(msg.get_payload())
            st.rerun()
        else:
            try:
                text = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                text = raw_bytes.decode("latin-1")
            st.session_state["det_body"] = text
            st.rerun()

    if st.button("🔎 Analyse", type="primary"):
        raw_text = (subject or "") + " " + (body or "")
        if len(raw_text.strip()) < 5:
            st.warning("Please enter some text to analyse.")
            return

        cleaned = clean_text(raw_text)

        # Get predictions from all models
        X = tfidf.transform([cleaned])
        probs = {}
        for name, model in models.items():
            prob = model.predict_proba(X)[0, 1]
            probs[name] = float(prob)

        ensemble_prob = probs.get("Ensemble (Soft Voting)", 0.5)
        verdict = "🚨 PHISHING" if ensemble_prob >= 0.5 else "✅ LEGITIMATE"

        # Verdict banner
        if ensemble_prob >= 0.5:
            st.error(f"## {verdict}")
        else:
            st.success(f"## {verdict}")

        st.metric("Ensemble Probability (Phishing)", f"{ensemble_prob:.1%}")
        st.progress(ensemble_prob)

        # Individual model probabilities
        st.subheader("Model Probabilities")
        ind_probs = {k: v for k, v in probs.items() if k != "Ensemble (Soft Voting)"}
        fig, ax = plt.subplots(figsize=(8, 3))
        names = list(ind_probs.keys())
        vals = list(ind_probs.values())
        colors = ["#e74c3c" if v >= 0.5 else "#2ecc71" for v in vals]
        ax.barh(names, vals, color=colors)
        ax.set_xlim(0, 1)
        ax.axvline(0.5, color="gray", linestyle="--", alpha=0.7)
        ax.set_xlabel("Phishing Probability")
        plt.tight_layout()
        st.pyplot(fig)

        # Agreement
        agree_count = sum(1 for v in ind_probs.values() if v >= 0.5)
        st.info(f"**Agreement:** {agree_count}/{len(ind_probs)} models say phishing")

        # Red flags
        st.subheader("🚩 Red Flags (Rule-Based)")
        flags = compute_red_flags(raw_text)
        if flags:
            for f in flags:
                st.warning(f"**{f['flag']}** — {f['detail']}")
        else:
            st.success("No obvious red flags detected.")

        # Top contributing words
        st.subheader("📊 Top Words Pushing Toward Phishing")
        lr_model = models.get("Logistic Regression")
        if lr_model:
            contribs = get_top_contributing_words(cleaned, tfidf, lr_model, n=10)
            if contribs:
                cdf = pd.DataFrame(contribs, columns=["Word", "Contribution", "Coefficient"])
                st.dataframe(cdf, width="stretch")
            else:
                st.info("No significant contributing words found.")

        # Log to predictions table
        try:
            conn = get_rw_connection()
            conn.execute(
                "INSERT INTO predictions (created_at, subject, text_preview, "
                "model_probs_json, ensemble_prob, verdict) VALUES (?,?,?,?,?,?)",
                (
                    datetime.datetime.now().isoformat(),
                    subject[:200] if subject else "",
                    (body or "")[:200],
                    json.dumps(probs),
                    ensemble_prob,
                    verdict,
                ),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass  # Don't crash the UI for logging failures


def page_model_comparison():
    st.header("📊 Model Comparison")

    conn = get_ro_connection()

    # Metrics table
    df = pd.read_sql("SELECT * FROM model_results", conn)
    if df.empty:
        st.warning("No model results found. Run `python train_models.py` first.")
        return

    st.subheader("Metrics Table")
    display_cols = ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "fnr", "train_seconds"]
    dfd = df[display_cols].copy()

    # Highlight best per column
    def highlight_best(s):
        if s.name in ["model", "train_seconds", "confusion_matrix_json"]:
            return [""] * len(s)
        if s.name == "fnr":
            best = s.min()
        else:
            best = s.max()
        return ["background-color: #d4edda" if v == best else "" for v in s]

    st.dataframe(dfd.style.apply(highlight_best), width="stretch")

    # Grouped bar chart
    st.subheader("Metrics Comparison")
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(df))
    w = 0.15
    for i, m in enumerate(metrics):
        ax.bar(x + i * w, df[m], w, label=m.upper())
    ax.set_xticks(x + w * 2)
    ax.set_xticklabels(df["model"], rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("Model Metrics Comparison")
    plt.tight_layout()
    st.pyplot(fig)

    # ROC curves
    st.subheader("ROC Curves")
    roc_df = pd.read_sql("SELECT * FROM roc_points", conn)
    fig, ax = plt.subplots(figsize=(8, 6))
    for _, row in roc_df.iterrows():
        fpr = json.loads(row["fpr_json"])
        tpr = json.loads(row["tpr_json"])
        ax.plot(fpr, tpr, label=row["model"])
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)

    # Confusion matrix heatmap
    st.subheader("Confusion Matrix")
    model_sel = st.selectbox("Select model", df["model"].tolist())
    row = df[df["model"] == model_sel].iloc[0]
    cm = json.loads(row["confusion_matrix_json"])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i][j]:,}", ha="center", va="center",
                    color="white" if cm[i][j] > max(cm[0][0], cm[1][1]) / 2 else "black")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Legitimate", "Phishing"])
    ax.set_yticklabels(["Legitimate", "Phishing"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix – {model_sel}")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    st.pyplot(fig)

    # Training time
    st.subheader("Training Time")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.barh(df["model"], df["train_seconds"], color="#3498db")
    ax.set_xlabel("Seconds")
    ax.set_title("Training Time per Model")
    plt.tight_layout()
    st.pyplot(fig)

    conn.close()


def page_preprocessing():
    st.header("🧹 Preprocessing Pipeline")

    st.markdown("""
    ### Cleaning Steps (applied in order)
    1. **Truncate** raw text to 5,000 characters
    2. **HTML unescape** + strip HTML tags
    3. **Replace** URLs → `urltoken`, emails → `emailtoken`
    4. **Lowercase** everything
    5. **Replace** digit sequences → `numtoken`
    6. **Remove** non-letter characters, collapse whitespace
    7. **Drop** tokens of length 1 or > 25
    8. TF-IDF `stop_words="english"` removes common stop words
    """)

    conn = get_ro_connection()

    # Example email: raw vs cleaned
    st.subheader("Raw vs Cleaned Example")
    row = conn.execute(
        "SELECT subject, body_trunc, clean_text FROM emails WHERE length(body_trunc) > 100 LIMIT 1"
    ).fetchone()
    if row:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Raw (subject + body)**")
            raw = (row["subject"] or "") + " " + (row["body_trunc"] or "")
            st.text_area("Raw text", raw[:2000], height=300, disabled=True)
        with col2:
            st.markdown("**Cleaned**")
            st.text_area("Cleaned text", row["clean_text"][:2000], height=300, disabled=True)

    # Stats
    st.subheader("Dataset Statistics")
    stats = conn.execute("""
        SELECT
            COUNT(*) as total_rows,
            AVG(length(body_trunc)) as avg_body_len,
            AVG(length(clean_text)) as avg_clean_len,
            SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END) as phishing,
            SUM(CASE WHEN label = 0 THEN 1 ELSE 0 END) as legitimate
        FROM emails
    """).fetchone()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", f"{stats['total_rows']:,}")
    col2.metric("Avg Body Length", f"{stats['avg_body_len']:,.0f} chars")
    col3.metric("Avg Clean Length", f"{stats['avg_clean_len']:,.0f} chars")
    col4.metric("Phishing / Legit", f"{stats['phishing']:,} / {stats['legitimate']:,}")
    conn.close()


def page_dataset_explorer():
    st.header("📁 Dataset Explorer")

    conn = get_ro_connection()

    # Stacked bar chart
    st.subheader("Label Distribution by Source")
    src_dist = pd.read_sql("""
        SELECT source, label, COUNT(*) as cnt
        FROM emails GROUP BY source, label
    """, conn)
    if not src_dist.empty:
        pivot = src_dist.pivot(index="source", columns="label", values="cnt").fillna(0)
        pivot.columns = ["Legitimate", "Phishing"]
        fig, ax = plt.subplots(figsize=(10, 4))
        pivot.plot(kind="bar", stacked=True, ax=ax, color=["#2ecc71", "#e74c3c"])
        ax.set_ylabel("Count")
        ax.set_title("Emails per Source")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        st.pyplot(fig)

    # Text length histogram
    st.subheader("Clean Text Length Distribution")
    lengths = pd.read_sql("SELECT length(clean_text) as tlen, label FROM emails", conn)
    fig, ax = plt.subplots(figsize=(10, 4))
    for lbl, color, name in [(0, "#2ecc71", "Legitimate"), (1, "#e74c3c", "Phishing")]:
        sub = lengths[lengths["label"] == lbl]
        ax.hist(sub["tlen"], bins=60, alpha=0.6, label=name, color=color)
    ax.set_xlabel("Clean text length (chars)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.set_title("Text Length Distribution")
    plt.tight_layout()
    st.pyplot(fig)

    # Per-source accuracy table
    st.subheader("Per-Source Accuracy & Recall")
    src_res = pd.read_sql("SELECT * FROM source_results", conn)
    if not src_res.empty:
        st.dataframe(src_res, width="stretch")

    # Leave-one-source-out
    st.subheader("Leave-One-Source-Out Evaluation")
    cross = pd.read_sql("SELECT * FROM cross_source_results", conn)
    if not cross.empty:
        st.dataframe(cross, width="stretch")
        st.info(
            "**What this tells us:** When a model is trained on all sources *except* one and "
            "tested on the held-out source, a low accuracy suggests the model learned the "
            "*style* of each source rather than generalisable phishing patterns. For phishing-only "
            "sources (Nazario, Nigerian_Fraud), only recall is meaningful—it shows whether "
            "the model can still detect phishing from an unseen distribution."
        )

    # Read-only SQL box
    st.subheader("🔍 Run SQL Query")
    query = st.text_area("Enter a SELECT query:", value="SELECT source, COUNT(*) as n FROM emails GROUP BY source")
    if st.button("Run Query"):
        q = query.strip()
        if not q.upper().startswith("SELECT"):
            st.error("Only SELECT queries are allowed.")
        else:
            try:
                result = pd.read_sql(q, conn)
                st.dataframe(result, width="stretch")
            except Exception as e:
                st.error(f"Query error: {e}")

    conn.close()


def page_prediction_history():
    st.header("📋 Prediction History")

    conn = get_rw_connection()

    # Metrics
    total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    flagged = conn.execute("SELECT COUNT(*) FROM predictions WHERE verdict LIKE '%PHISHING%'").fetchone()[0]

    col1, col2 = st.columns(2)
    col1.metric("Total Analyses", total)
    col2.metric("Flagged as Phishing", flagged)

    # Table
    df = pd.read_sql("SELECT * FROM predictions ORDER BY id DESC", conn)
    if not df.empty:
        st.dataframe(df, width="stretch")
    else:
        st.info("No predictions yet. Go to the **Detect Email** page to analyse an email.")

    # Clear
    if st.button("🗑️ Clear History"):
        conn.execute("DELETE FROM predictions")
        conn.commit()
        st.rerun()

    conn.close()


# ═════════════════════════════════════════════════════════════════════
# Navigation
# ═════════════════════════════════════════════════════════════════════

PAGES = {
    "🔍 Detect Email": page_detect,
    "📊 Model Comparison": page_model_comparison,
    "🧹 Preprocessing": page_preprocessing,
    "📁 Dataset Explorer": page_dataset_explorer,
    "📋 Prediction History": page_prediction_history,
}

def main():
    preflight()
    st.sidebar.title("🛡️ Phishing Detector")
    page = st.sidebar.radio("Navigate", list(PAGES.keys()))
    PAGES[page]()

if __name__ == "__main__":
    main()

