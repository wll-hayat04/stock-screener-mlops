import yfinance as yf
import pandas as pd
import sqlite3
import os

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "GS", "BAC", "XOM", "CVX", "JNJ", "PFE", "UNH"]

DB_PATH = "data/stocks.db"

def fetch_and_store(tickers=TICKERS, period="2y"):
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    for ticker in tickers:
        try:
            df = yf.download(ticker, period=period, auto_adjust=True)
            df.reset_index(inplace=True)
            df["Ticker"] = ticker
            df.to_sql(ticker, conn, if_exists="replace", index=False)
        except Exception as e:
            print(ticker, "failed:", e)
    conn.close()

def load_ticker(ticker):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM " + ticker, conn)
    conn.close()

    # Columns look like "('Date', '')" and "('Close', 'AAPL')"
    new_cols = []
    for col in df.columns:
        try:
            parsed = eval(col)
            field = parsed[0]
            new_cols.append(field)
        except:
            new_cols.append(col)
    df.columns = new_cols

    df["Date"] = pd.to_datetime(df["Date"])
    df["Ticker"] = ticker
    df = df.sort_values("Date").reset_index(drop=True)
    return df[["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]]