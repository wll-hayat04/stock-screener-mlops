import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pipeline.fetch_data import load_ticker, TICKERS
from pipeline.features import compute_features

FEATURE_COLS = [
    "return_1d", "return_5d", "return_10d", "return_20d",
    "ma_ratio_10_50", "ma_ratio_10_20",
    "rsi", "macd", "macd_signal", "macd_hist",
    "bb_width", "bb_pct",
    "volatility_20d", "volume_ratio"
]

MODELS = {
    "RandomForest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ]),
    "GradientBoosting": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=100, random_state=42))
    ]),
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ]),
}

def load_all_data():
    frames = []
    for ticker in TICKERS:
        try:
            df = load_ticker(ticker)
            df = compute_features(df)
            df["Ticker"] = ticker
            frames.append(df)
        except Exception as e:
            print(f"Warning: {ticker} failed — {e}")
    return pd.concat(frames, ignore_index=True)

def train_and_evaluate(model_dir="model/saved"):
    os.makedirs(model_dir, exist_ok=True)
    print("Loading data...")
    all_df = load_all_data()

    X = all_df[FEATURE_COLS]
    y = all_df["target"].astype(int)

    tscv = TimeSeriesSplit(n_splits=5)
    results = []

    for name, model in MODELS.items():
        accs, f1s, aucs = [], [], []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            accs.append(accuracy_score(y_test, y_pred))
            f1s.append(f1_score(y_test, y_pred, zero_division=0))
            aucs.append(roc_auc_score(y_test, y_prob))

        model.fit(X, y)
        joblib.dump(model, os.path.join(model_dir, f"{name}.pkl"))

        results.append({
            "Model": name,
            "Accuracy": round(np.mean(accs), 4),
            "F1": round(np.mean(f1s), 4),
            "ROC-AUC": round(np.mean(aucs), 4),
        })
        print(f"✓ {name} — Acc: {np.mean(accs):.4f} | F1: {np.mean(f1s):.4f} | AUC: {np.mean(aucs):.4f}")

    return pd.DataFrame(results)

if __name__ == "__main__":
    df = train_and_evaluate()
    print(df)
