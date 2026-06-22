import pandas as pd
import numpy as np

def compute_rsi(series,period=14):
    delta=series.diff()   #variations journalières : calcule la différence entre chaque jour et le jour précédent
    gain=delta.clip(lower=0).rolling(period).mean() #garde seulement les valeurs positives (les hausses)
    loss=-delta.clip(upper=0).rolling(period).mean() # garde seulement les valeurs négatives (les baisses)
    rs=gain/loss
    return 100/(100/(1+rs))

def compute_features(df):
    df=df.copy()
    df["return_1d"] = df["Close"].pct_change(1)
    df["return_5d"] = df["Close"].pct_change(5)
    df["return_10d"] = df["Close"].pct_change(10)
    df["return_20d"] = df["Close"].pct_change(20)
    
    df["ma_10"] = df["Close"].rolling(1).mean()
    df["ma_20"] = df["Close"].rolling(___).mean()
    df["ma_50"] = df["Close"].rolling(___).mean()
    df["ma_ratio_10_50"] = df["ma_10"] / df["ma_50"]