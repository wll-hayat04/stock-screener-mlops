import os
import sys
import joblib
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pipeline.fetch_data import load_ticker, TICKERS
from pipeline.features import compute_features
from model.train import FEATURE_COLS

def load_best_model(model_dir="model/saved"):
    for name in ["RandomForest", "GradientBoosting", "LogisticRegression"]:
        path = os.path.join(model_dir, f"{name}.pkl")
        if os.path.exists(path):
            return joblib.load(path)
    raise FileNotFoundError("No trained model found. Run model/train.py first.")

def generate_signals(model_dir="model/saved"):
    model = load_best_model(model_dir)
    rows = []

    for ticker in TICKERS:
        try:
            df = load_ticker(ticker)
            df = compute_features(df)
            latest = df.iloc[[-1]]
            X = latest[FEATURE_COLS]
            prob = model.predict_proba(X)[0, 1]
            signal = "BUY" if prob > 0.6 else ("SELL" if prob < 0.4 else "HOLD")

            rows.append({
                "Ticker":     ticker,
                "Date":       latest["Date"].values[0],
                "Close":      round(latest["Close"].values[0], 2),
                "RSI":        round(latest["rsi"].values[0], 1),
                "MACD":       round(latest["macd"].values[0], 3),
                "BB_%":       round(latest["bb_pct"].values[0] * 100, 1),
                "Volatility": round(latest["volatility_20d"].values[0] * 100, 1),
                "Signal":     signal,
                "Confidence": round(prob * 100, 1),
            })
        except Exception as e:
            print(f"Warning: {ticker} — {e}")

    return pd.DataFrame(rows).sort_values("Confidence", ascending=False).reset_index(drop=True)

if __name__ == "__main__":
    signals = generate_signals()
    print(signals)
