import yfinance as  yf
import pandas as pd
import sqlite3
import os

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "TSLA", "NVDA", "JPM", "GS", "BAC",
    "XOM", "CVX", "JNJ", "PFE", "UNH"
]

DB_PATH = "data/stocks.db"

def fetch_and_store(tickers=TICKERS, period="2y"):
    os.makedirs("data,exist_ok=True")
    conn=sqlite3.connect(DB_PATH)

    for ticker in tickers:
        print(f"Fetching {ticker}...")
        try:
            df=yf.download(ticker,period=period,auto_adjust=True)
            df.reset_index(inplace=True)
            df["Ticker"]=ticker
            df.to_sql(ticker,conn,if_exists="replace",index=False)
            print(f" {ticker} - {len(df)} rows stored")
        except Exception as e:
            print(f"  {ticker} failed: {e}")
        
    conn.close()
    print("\nDone. Data stored in", DB_PATH)
        
def load_ticker(ticker):
    conn=sqlite3.connect(DB_PATH)
    df=pd.read_sql(f"SELECT * FROM '{ticker}'",conn)
    conn.close()
    df["Date"]=pd.to_datetime(df["Date"])
    df.sort_values("Date",inplace=True)
    df.reset_index(drop=True,inplace=True)
    
    return df

if __name__=="__main__":
    os.makedirs("data",exist_ok=True)
    fetch_and_store()