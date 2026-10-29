"""
train_models.py - Train five ML models + ensemble on the SQLite database.
Flags:
  --fast      Use a stratified 25,000-row sample for quick iteration
  --stacking  Also train a StackingClassifier-style ensemble
"""

import argparse
import json
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from common import DB_PATH, MODELS_DIR, ensure_dirs, get_db_connection
from ensemble import SoftVotingEnsemble

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_data(fast: bool = False):
    """Load emails from SQLite, optionally subsample."""
    conn = get_db_connection()
    df = pd.read_sql("SELECT id, source, clean_text, label FROM emails", conn)
    conn.close()
    print(f"Loaded {len(df):,} rows from DB.")
    if fast:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=25_000, random_state=42)
        idx, _ = next(sss.split(df, df["label"]))
        df = df.iloc[idx].reset_index(drop=True)
        print(f"  --fast mode: sampled {len(df):,} rows")
    return df


def split_data(df: pd.DataFrame):
    """Stratified 80/20 split; writes 'split' column back to the DB."""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(sss.split(df, df["label"]))
    df["split"] = "test"
    df.loc[train_idx, "split"] = "train"

    # Write split back
    conn = get_db_connection()
    # Reset all splits first
    conn.execute("UPDATE emails SET split = NULL")
    for _, row in df.iterrows():
        conn.execute("UPDATE emails SET split = ? WHERE id = ?", (row["split"], row["id"]))
    conn.commit()
    conn.close()

    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    print(f"Train: {len(train):,}  Test: {len(test):,}")
    print(f"  Train label dist: {dict(train['label'].value_counts().sort_index())}")
    print(f"  Test  label dist: {dict(test['label'].value_counts().sort_index())}")
    return train, test


def evaluate(name, model, X_test, y_test, train_sec):
    """Compute all metrics for a single model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    fnr = cm[1, 0] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0.0
    fpr_arr, tpr_arr, _ = roc_curve(y_test, y_prob)

    return {
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "fnr": fnr,
        "train_seconds": train_sec,
        "confusion_matrix": cm.tolist(),
        "fpr": fpr_arr.tolist(),
        "tpr": tpr_arr.tolist(),
    }


def downsample_roc(fpr, tpr, n=100):
    """Downsample ROC curve to ~n points."""
    if len(fpr) <= n:
        return fpr, tpr
    idx = np.linspace(0, len(fpr) - 1, n, dtype=int)
    return [fpr[i] for i in idx], [tpr[i] for i in idx]


# ---------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Use 25k sample")
    parser.add_argument("--stacking", action="store_true", help="Also train stacking ensemble")
    args = parser.parse_args()

    ensure_dirs()
    print("=" * 60)
    print("PHISHING EMAIL DETECTOR - Training Pipeline")
    print("=" * 60)

    # 1. Load & split
    df = load_data(fast=args.fast)
    train_df, test_df = split_data(df)

    X_train_text = train_df["clean_text"].values
    X_test_text = test_df["clean_text"].values
    y_train = train_df["label"].values
    y_test = test_df["label"].values

    # 2. TF-IDF (fit on train only)
    print("\nFitting TF-IDF vectorizer ...")
    tfidf = TfidfVectorizer(
        max_features=30_000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True,
        stop_words="english",
    )
    X_train = tfidf.fit_transform(X_train_text)
    X_test = tfidf.transform(X_test_text)
    print(f"  Vocabulary size: {len(tfidf.vocabulary_):,}")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test  shape: {X_test.shape}")

    # 3. Define models
    models = {
        "Logistic Regression": LogisticRegression(
            solver="liblinear", C=1, random_state=42, max_iter=1000
        ),
        "SVM (LinearSVC)": CalibratedClassifierCV(
            LinearSVC(C=1, random_state=42, max_iter=5000), cv=3
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_features="sqrt", n_jobs=-1, random_state=42
        ),
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2, random_state=42),
            n_estimators=200, learning_rate=0.5, random_state=42,
        ),
    }

    # 4. Train & evaluate
    results = []
    fitted_models = {}
    for name, model in models.items():
        print(f"\n{'-' * 40}")
        print(f"Training {name} ...")
        t0 = time.time()
        model.fit(X_train, y_train)
        train_sec = time.time() - t0
        fitted_models[name] = model

        res = evaluate(name, model, X_test, y_test, train_sec)
        results.append(res)
        print(f"  Accuracy: {res['accuracy']:.4f}  Precision: {res['precision']:.4f}  "
              f"Recall: {res['recall']:.4f}  F1: {res['f1']:.4f}  AUC: {res['roc_auc']:.4f}  "
              f"FNR: {res['fnr']:.4f}  Time: {train_sec:.1f}s")

    # 5. Ensemble (soft voting)
    print(f"\n{'-' * 40}")
    print("Building SoftVotingEnsemble ...")
    ensemble = SoftVotingEnsemble(
        estimators=[(n, m) for n, m in fitted_models.items()]
    )
    t0 = time.time()
    ens_sec = time.time() - t0
    res = evaluate("Ensemble (Soft Voting)", ensemble, X_test, y_test, ens_sec)
    results.append(res)
    fitted_models["Ensemble (Soft Voting)"] = ensemble
    print(f"  Accuracy: {res['accuracy']:.4f}  Precision: {res['precision']:.4f}  "
          f"Recall: {res['recall']:.4f}  F1: {res['f1']:.4f}  AUC: {res['roc_auc']:.4f}")

    # 6. Optional stacking
    if args.stacking:
        print(f"\n{'-' * 40}")
        print("Training StackingClassifier ...")
        stack_estimators = [
            ("lr", LogisticRegression(solver="liblinear", C=1, random_state=42, max_iter=1000)),
            ("svm", CalibratedClassifierCV(LinearSVC(C=1, random_state=42, max_iter=5000), cv=3)),
            ("rf", RandomForestClassifier(n_estimators=200, max_features="sqrt", n_jobs=-1, random_state=42)),
            ("ada", AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=2, random_state=42),
                n_estimators=200, learning_rate=0.5, random_state=42)),
        ]
        stacker = StackingClassifier(
            estimators=stack_estimators,
            final_estimator=LogisticRegression(solver="liblinear", random_state=42),
            cv=5, n_jobs=-1,
        )
        t0 = time.time()
        stacker.fit(X_train, y_train)
        stack_sec = time.time() - t0
        res = evaluate("Stacking Ensemble", stacker, X_test, y_test, stack_sec)
        results.append(res)
        fitted_models["Stacking Ensemble"] = stacker
        print(f"  Accuracy: {res['accuracy']:.4f}  F1: {res['f1']:.4f}  Time: {stack_sec:.1f}s")

    # -- Comparison table ----------------------------------------------
    print("\n" + "=" * 90)
    print("MODEL COMPARISON")
    print("=" * 90)
    header = f"{'Model':<28s} {'Acc':>6s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'AUC':>6s} {'FNR':>6s} {'Time':>7s}"
    print(header)
    print("-" * 90)
    for r in results:
        print(f"{r['model']:<28s} {r['accuracy']:>6.4f} {r['precision']:>6.4f} "
              f"{r['recall']:>6.4f} {r['f1']:>6.4f} {r['roc_auc']:>6.4f} "
              f"{r['fnr']:>6.4f} {r['train_seconds']:>6.1f}s")
    print("=" * 90)

    # -- Per-source evaluation -----------------------------------------
    print("\nPER-SOURCE EVALUATION")
    print("-" * 80)
    source_results_data = []
    for name, model in fitted_models.items():
        for src in test_df["source"].unique():
            mask = test_df["source"].values == src
            if mask.sum() == 0:
                continue
            X_s = X_test[mask]
            y_s = y_test[mask]
            y_pred = model.predict(X_s)
            acc = accuracy_score(y_s, y_pred)
            rec = recall_score(y_s, y_pred, zero_division=0)
            source_results_data.append({
                "model": name, "source": src, "n": int(mask.sum()),
                "accuracy": acc, "recall": rec,
            })
    src_df = pd.DataFrame(source_results_data)
    for name in fitted_models:
        sub = src_df[src_df["model"] == name]
        print(f"\n  {name}:")
        for _, row in sub.iterrows():
            print(f"    {row['source']:20s}  n={row['n']:>5d}  acc={row['accuracy']:.4f}  rec={row['recall']:.4f}")

    # -- Leave-one-source-out (LR only) -------------------------------
    print("\n" + "=" * 80)
    print("LEAVE-ONE-SOURCE-OUT EVALUATION (Logistic Regression)")
    print("=" * 80)
    cross_source_data = []
    sources_all = df["source"].unique()
    for held in sources_all:
        train_mask = df["source"] != held
        test_mask = df["source"] == held
        X_tr_text = df.loc[train_mask, "clean_text"].values
        y_tr = df.loc[train_mask, "label"].values
        X_te_text = df.loc[test_mask, "clean_text"].values
        y_te = df.loc[test_mask, "label"].values

        vec = TfidfVectorizer(
            max_features=30_000, ngram_range=(1, 2), min_df=3,
            max_df=0.9, sublinear_tf=True, stop_words="english",
        )
        X_tr = vec.fit_transform(X_tr_text)
        X_te = vec.transform(X_te_text)

        lr = LogisticRegression(solver="liblinear", C=1, random_state=42, max_iter=1000)
        lr.fit(X_tr, y_tr)
        y_pred = lr.predict(X_te)

        n_unique = len(set(y_te))
        acc = accuracy_score(y_te, y_pred) if n_unique > 1 else None
        rec = recall_score(y_te, y_pred, zero_division=0)
        f1_val = f1_score(y_te, y_pred, zero_division=0) if n_unique > 1 else None

        cross_source_data.append({
            "held_out_source": held,
            "n": len(y_te),
            "accuracy": acc,
            "recall": rec,
            "f1": f1_val,
        })
        if n_unique == 1:
            print(f"  {held:20s}  n={len(y_te):>5d}  (phishing-only) recall={rec:.4f}")
        else:
            print(f"  {held:20s}  n={len(y_te):>5d}  acc={acc:.4f}  rec={rec:.4f}  f1={f1_val:.4f}")

    # -- Save models --------------------------------------------------
    print("\nSaving models ...")
    joblib.dump(tfidf, MODELS_DIR / "tfidf_vectorizer.joblib")
    for name, model in fitted_models.items():
        fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(model, MODELS_DIR / f"{fname}.joblib")
    print("  Saved to:", MODELS_DIR)

    # -- Write results to DB ------------------------------------------
    print("\nWriting results to SQLite ...")
    conn = get_db_connection()
    cur = conn.cursor()

    # model_results
    cur.execute("DELETE FROM model_results")
    for r in results:
        cur.execute(
            "INSERT INTO model_results VALUES (?,?,?,?,?,?,?,?,?)",
            (r["model"], r["accuracy"], r["precision"], r["recall"], r["f1"],
             r["roc_auc"], r["fnr"], r["train_seconds"],
             json.dumps(r["confusion_matrix"])),
        )

    # roc_points
    cur.execute("DELETE FROM roc_points")
    for r in results:
        fpr_ds, tpr_ds = downsample_roc(r["fpr"], r["tpr"])
        cur.execute(
            "INSERT INTO roc_points VALUES (?,?,?)",
            (r["model"], json.dumps(fpr_ds), json.dumps(tpr_ds)),
        )

    # source_results
    cur.execute("DELETE FROM source_results")
    for row in source_results_data:
        cur.execute(
            "INSERT INTO source_results VALUES (?,?,?,?,?)",
            (row["model"], row["source"], row["n"], row["accuracy"], row["recall"]),
        )

    # cross_source_results
    cur.execute("DELETE FROM cross_source_results")
    for row in cross_source_data:
        cur.execute(
            "INSERT INTO cross_source_results VALUES (?,?,?,?,?)",
            (row["held_out_source"], row["n"], row["accuracy"], row["recall"], row["f1"]),
        )

    # top_words (LR coefficients)
    cur.execute("DELETE FROM top_words")
    lr_model = fitted_models["Logistic Regression"]
    feature_names = tfidf.get_feature_names_out()
    coefs = lr_model.coef_[0]
    top_idx = np.argsort(coefs)
    top_phish = top_idx[-30:][::-1]
    top_legit = top_idx[:30]
    for i in list(top_phish) + list(top_legit):
        cur.execute(
            "INSERT OR IGNORE INTO top_words VALUES (?,?)",
            (feature_names[i], float(coefs[i])),
        )

    conn.commit()
    conn.close()
    print("Done. All results written to DB.")


if __name__ == "__main__":
    main()
