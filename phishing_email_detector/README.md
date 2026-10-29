# Phishing Email Detector

A machine-learning-based phishing email detection system with five supervised models, an ensemble classifier, and an interactive Streamlit GUI.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build the SQLite database (loads, cleans, deduplicates ~82k emails)
python build_db.py

# 3. Train models (fast mode for quick iteration)
python train_models.py --fast

# 3b. Full training (recommended for production-quality results)
python train_models.py

# 4. Launch the Streamlit app
streamlit run app.py
```

## Project Files

| File | Description |
|---|---|
| `requirements.txt` | Python dependencies |
| `common.py` | Shared paths, DB helpers, read-only connection |
| `preprocess.py` | Text preprocessing pipeline (single source of truth) |
| `ensemble.py` | SoftVotingEnsemble class (picklable) |
| `build_db.py` | Loads 6 CSV sources, cleans, writes SQLite DB |
| `train_models.py` | Trains 5 models + ensemble, evaluates, writes results |
| `app.py` | Streamlit GUI with 5 pages |
| `data/` | Source CSV files |
| `models/` | Saved vectorizer and model files (.joblib) |
| `phishing_emails.db` | SQLite database with emails and results |

## Data Sources

| Source | Rows (raw) | Description |
|---|---|---|
| CEAS_08 | 39,154 | CEAS 2008 spam challenge dataset |
| Enron | 29,767 | Enron email corpus (ham + spam) |
| Ling | 2,859 | Linguistics mailing list (mostly ham) |
| Nazario | 1,565 | Phishing emails collected by Jose Nazario (all phishing) |
| Nigerian_Fraud | 3,332 | Nigerian 419/advance-fee scam emails (all phishing) |
| SpamAssasin | 5,809 | SpamAssassin public corpus |

After cleaning (removing short/duplicate texts): ~73,800 rows.

## Preprocessing Pipeline

1. Truncate to 5,000 characters
2. HTML unescape + strip tags
3. URLs -> `urltoken`, emails -> `emailtoken`
4. Lowercase
5. Digit sequences -> `numtoken`
6. Remove non-letter characters, collapse whitespace
7. Drop tokens of length 1 or > 25
8. TF-IDF with `stop_words="english"` removes common English stop words

## Models

1. **Logistic Regression** (solver=liblinear, C=1)
2. **SVM (LinearSVC)** wrapped in CalibratedClassifierCV(cv=3) for probability outputs. Uses a linear kernel because RBF is computationally infeasible on ~60k sparse text vectors with 30k features.
3. **Random Forest** (n_estimators=200, max_features=sqrt)
4. **AdaBoost** (base=DecisionTreeClassifier(max_depth=2), n_estimators=200, lr=0.5)
5. **Ensemble (Soft Voting)** - averages predict_proba of the four models above

## Database Tables

| Table | Description |
|---|---|
| `emails` | Cleaned email data with source, subject, body, clean_text, label, split |
| `model_results` | Per-model metrics: accuracy, precision, recall, F1, AUC, FNR, training time |
| `source_results` | Per-model, per-source accuracy and recall |
| `cross_source_results` | Leave-one-source-out evaluation results |
| `roc_points` | Downsampled ROC curve data for plotting |
| `top_words` | Top 30 phishing + 30 legitimate words (LR coefficients) |
| `predictions` | Logged user analyses from the GUI |

## Streamlit App Pages

1. **Detect Email** - Analyse any email text, upload .txt/.eml files, view model predictions and red flags
2. **Model Comparison** - Metrics table, grouped bar chart, ROC curves, confusion matrices, training times
3. **Preprocessing** - Pipeline description, raw vs cleaned example, dataset statistics
4. **Dataset Explorer** - Label distribution by source, text length histogram, per-source accuracy, leave-one-source-out results, SQL query box
5. **Prediction History** - Logged analyses with clear button

## Limitations

- **Source bias**: Models may learn source-specific writing style rather than generalisable phishing indicators. The leave-one-source-out evaluation reveals this.
- **Old data**: Enron emails are from 2001; Nigerian fraud emails from 1998-2007; CEAS from 2008. Modern phishing is more sophisticated.
- **English only**: All training data is English. Non-English phishing emails will not be detected well.
- **Spam != Phishing**: The dataset labels spam and phishing as the same class (label=1). True phishing (credential theft, malware delivery) is a subset of spam.
- **Not production-ready**: This is an educational project. A production system would need real-time email parsing, header analysis, URL reputation checks, and continuous retraining.
- **Linear SVM**: We use LinearSVC instead of RBF SVM because the RBF kernel has O(n^2) training complexity, making it impractical for 60k+ sparse TF-IDF vectors.

