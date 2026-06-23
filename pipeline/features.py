import pandas as pd
import numpy as np

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_features(df):
    df = df.copy()

    # ── Returns ───────────────────────────────────────
    df["return_1d"]  = df["Close"].pct_change(1)
    df["return_5d"]  = df["Close"].pct_change(5)
    df["return_10d"] = df["Close"].pct_change(10)
    df["return_20d"] = df["Close"].pct_change(20)

    # ── Moving Averages ───────────────────────────────
    df["ma_10"] = df["Close"].rolling(10).mean()
    df["ma_20"] = df["Close"].rolling(20).mean()
    df["ma_50"] = df["Close"].rolling(50).mean()
    df["ma_ratio_10_50"] = df["ma_10"] / df["ma_50"]
    df["ma_ratio_10_20"] = df["ma_10"] / df["ma_20"]

    # ── RSI ───────────────────────────────────────────
    df["rsi"] = compute_rsi(df["Close"], period=14)

    # ── MACD ──────────────────────────────────────────
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"]        = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # ── Bollinger Bands ───────────────────────────────
    mid = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    df["bb_upper"] = mid + 2 * std
    df["bb_lower"] = mid - 2 * std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / mid
    df["bb_pct"]   = (df["Close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # ── Volatility ────────────────────────────────────
    df["volatility_20d"] = df["return_1d"].rolling(20).std() * np.sqrt(252)

    # ── Volume ────────────────────────────────────────
    df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

    # ── Target: will price be higher in 5 days? ──────
    df["target"] = (df["Close"].shift(-5) > df["Close"]).astype(int)

    return df.dropna().reset_index(drop=True)
