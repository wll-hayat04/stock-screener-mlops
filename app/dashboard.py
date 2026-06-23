import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="StockSense — MLOps", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600;9..40,700;9..40,800&display=swap');
  @import url('https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.0/font/bootstrap-icons.css');

  html, body, [class*="css"] { font-family: 'DM Sans', system-ui, sans-serif; }
  .main, .block-container { background: #ffffff !important; }
  section[data-testid="stSidebar"] { background: #f8f7ff !important; border-right: 1px solid #ede9fe; }

  .hero {
    background: linear-gradient(135deg, #5b21b6 0%, #7c3aed 55%, #a78bfa 100%);
    border-radius: 20px; padding: 2.75rem 2.5rem; margin-bottom: 2rem; color: white;
  }
  .hero-tag {
    display: inline-block; background: rgba(255,255,255,0.15); border: 1px solid rgba(255,255,255,0.25);
    border-radius: 30px; padding: 4px 14px; font-size: 0.7rem; font-weight: 500;
    letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 1rem;
  }
  .hero h1 { font-size: 2.5rem; font-weight: 700; margin: 0 0 0.5rem; letter-spacing: -0.03em; line-height: 1.15; }
  .hero p  { font-size: 0.95rem; opacity: 0.82; margin: 0; line-height: 1.65; font-weight: 400; }

  .kpi { background: #fff; border: 1px solid #ede9fe; border-radius: 14px;
         padding: 1.25rem 1.5rem; text-align: center; box-shadow: 0 2px 8px rgba(124,58,237,0.06); }
  .kpi-icon { width: 38px; height: 38px; border-radius: 10px; background: #f3f0ff;
              display: flex; align-items: center; justify-content: center;
              margin: 0 auto 0.6rem; font-size: 1rem; color: #7c3aed; }
  .kpi-val  { font-size: 1.9rem; font-weight: 700; color: #5b21b6; line-height: 1; letter-spacing: -0.02em; }
  .kpi-lbl  { font-size: 0.72rem; font-weight: 500; color: #6b7280; text-transform: uppercase;
              letter-spacing: 0.08em; margin-top: 5px; }

  .feat-card { background: #faf9ff; border: 1px solid #ede9fe; border-radius: 14px;
               padding: 1.5rem; box-shadow: 0 1px 4px rgba(124,58,237,0.04); }
  .feat-icon { width: 42px; height: 42px; border-radius: 11px; background: #7c3aed;
               display: flex; align-items: center; justify-content: center;
               margin-bottom: 0.75rem; font-size: 1rem; color: white; }
  .feat-title { font-size: 0.95rem; font-weight: 600; color: #111827; margin-bottom: 5px; letter-spacing: -0.01em; }
  .feat-desc  { font-size: 0.83rem; color: #374151; line-height: 1.6; font-weight: 400; }

  .page-card { background: #fff; border: 1px solid #ede9fe; border-left: 3px solid #7c3aed;
               border-radius: 12px; padding: 1rem 1.25rem; margin-bottom: 10px;
               box-shadow: 0 1px 4px rgba(124,58,237,0.04); }
  .page-card-title { font-size: 0.95rem; font-weight: 600; color: #5b21b6; margin: 6px 0 3px; letter-spacing: -0.01em; }
  .page-card-desc  { font-size: 0.83rem; color: #374151; line-height: 1.5; }

  .tech-badge { background: #f3f0ff; border: 1px solid #ddd6fe; border-radius: 10px;
                padding: 0.8rem 1rem; text-align: center; margin-bottom: 8px; }
  .tech-name  { font-size: 0.82rem; font-weight: 600; color: #4c1d95; margin: 5px 0 2px; letter-spacing: -0.01em; }
  .tech-role  { font-size: 0.7rem; color: #6b7280; font-weight: 400; }

  .stock-tag { display: inline-block; background: #f3f0ff; color: #5b21b6;
               border: 1px solid #ddd6fe; border-radius: 6px;
               padding: 3px 10px; margin: 3px; font-size: 0.75rem; font-weight: 500; }

  .sec-label { font-size: 0.67rem; font-weight: 600; color: #7c3aed;
               text-transform: uppercase; letter-spacing: 0.1em; margin: 2rem 0 0.75rem; }

  .upload-hint { background: #faf9ff; border: 2px dashed #c4b5fd; border-radius: 14px;
                 padding: 2rem; text-align: center; margin-bottom: 1rem; }

  .cta-box { background: linear-gradient(135deg,#5b21b6,#7c3aed); border-radius: 16px;
             padding: 2rem; text-align: center; margin-top: 2rem; color: white; }
  .cta-box h3 { margin: 0 0 6px; font-weight: 700; font-size: 1.1rem; letter-spacing: -0.01em; }
  .cta-box p  { margin: 0; opacity: 0.85; font-size: 0.88rem; }

  [data-testid="stSidebar"] .stRadio label { color: #1f2937 !important; font-size: 1rem !important; font-weight: 500 !important; }
  [data-testid="stSidebar"] .stRadio span { color: #1f2937 !important; font-size: 1rem !important; font-weight: 500 !important; }
  [data-testid="stSidebar"] p { color: #1f2937 !important; font-size: 0.9rem !important; }
  [data-testid="stMetricValue"] { color: #5b21b6 !important; font-weight: 700 !important; font-size: 1.6rem !important; }
  [data-testid="stMetricLabel"] { color: #374151 !important; font-size: 0.85rem !important; }
  [data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

  #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'model', 'saved')

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1.25rem 0 1rem; border-bottom:1px solid #ede9fe; margin-bottom:1rem;'>
      <div style='font-size:1.15rem; font-weight:800; color:#5b21b6; letter-spacing:-0.5px;'>StockSense</div>
      <div style='font-size:0.68rem; color:#9ca3af; margin-top:2px; letter-spacing:0.1em; text-transform:uppercase; font-weight:500;'>MLOps Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation", ["🏠 Home", "📊 Screener", "📂 Custom CSV", "📈 Stock Detail", "🤖 Train Models"], label_visibility="collapsed")

    st.markdown("""
    <div style='margin-top:1.5rem; padding-top:1rem; border-top:1px solid #ede9fe; font-size:0.78rem; color:#6b7280; line-height:2;'>
      <div style='font-weight:600; color:#5b21b6; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em;'>Models</div>
      Random Forest · GBM · LogReg
      <div style='font-weight:600; color:#5b21b6; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em; margin-top:10px;'>Validation</div>
      TimeSeriesSplit · 5 folds
      <div style='font-weight:600; color:#5b21b6; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em; margin-top:10px;'>Coverage</div>
      15 major US tickers
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.75rem; color:#9ca3af;'>Built by <b style='color:#5b21b6;'>Hayat</b> · <a href='https://github.com/wll-hayat04' style='color:#7c3aed; text-decoration:none;'>GitHub ↗</a></div>", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
models_exist = (os.path.exists(MODEL_DIR) and any(f.endswith(".pkl") for f in os.listdir(MODEL_DIR))) if os.path.exists(MODEL_DIR) else False

def base_layout(height=400, showlegend=True):
    return dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#faf9ff",
        font=dict(color="#1f2937", family="DM Sans", size=13),
        margin=dict(t=30, b=10, l=10, r=10),
        height=height,
        showlegend=showlegend,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=12, color="#374151")),
        xaxis=dict(gridcolor="#f3f0ff", linecolor="#ede9fe", tickfont=dict(color="#374151", size=11)),
        yaxis=dict(gridcolor="#f3f0ff", linecolor="#ede9fe", tickfont=dict(color="#374151", size=11)),
    )

@st.cache_data
def get_signals():
    from model.predict import generate_signals
    return generate_signals(MODEL_DIR)

@st.cache_data
def get_stock_data(ticker):
    from pipeline.fetch_data import load_ticker
    from pipeline.features import compute_features
    return compute_features(load_ticker(ticker))

def train_models():
    from model.train import train_and_evaluate
    return train_and_evaluate(model_dir=MODEL_DIR)

def detect_and_load_csv(f):
    df = pd.read_csv(f)
    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ["date","time","timestamp","datetime"]: col_map[c] = "Date"
        elif cl in ["open","open price","o"]:            col_map[c] = "Open"
        elif cl in ["high","high price","h"]:            col_map[c] = "High"
        elif cl in ["low","low price","l"]:              col_map[c] = "Low"
        elif cl in ["close","close price","c","adj close","adjusted close","price"]: col_map[c] = "Close"
        elif cl in ["volume","vol","v"]:                 col_map[c] = "Volume"
    df = df.rename(columns=col_map)
    missing = [c for c in ["Date","Open","High","Low","Close","Volume"] if c not in df.columns]
    if missing: return None, f"Missing: {missing}"
    df["Date"] = pd.to_datetime(df["Date"])
    df["Ticker"] = "CUSTOM"
    return df.sort_values("Date").reset_index(drop=True)[["Date","Ticker","Open","High","Low","Close","Volume"]], None

def candlestick_fig(sub):
    fig = go.Figure(data=[go.Candlestick(
        x=sub["Date"], open=sub["Open"], high=sub["High"], low=sub["Low"], close=sub["Close"],
        increasing_line_color="#7c3aed", decreasing_line_color="#e11d48",
        increasing_fillcolor="#ede9fe", decreasing_fillcolor="#fee2e2"
    )])
    if "ma_20" in sub.columns:
        fig.add_trace(go.Scatter(x=sub["Date"], y=sub["ma_20"], name="MA20", line=dict(color="#7c3aed", width=1.5, dash="dot")))
    if "ma_50" in sub.columns:
        fig.add_trace(go.Scatter(x=sub["Date"], y=sub["ma_50"], name="MA50", line=dict(color="#a78bfa", width=1.5, dash="dot")))
    layout = base_layout(400)
    layout["xaxis"]["rangeslider"] = dict(visible=False)
    fig.update_layout(**layout)
    return fig

def rsi_fig(sub):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub["Date"], y=sub["rsi"], fill="tozeroy",
                             fillcolor="rgba(124,58,237,0.07)", line=dict(color="#7c3aed", width=2), name="RSI"))
    fig.add_hline(y=70, line_dash="dash", line_color="#e11d48", line_width=1, annotation_text="70", annotation_font_color="#e11d48")
    fig.add_hline(y=30, line_dash="dash", line_color="#059669", line_width=1, annotation_text="30", annotation_font_color="#059669")
    layout = base_layout(240, showlegend=False)
    layout["yaxis"]["range"] = [0, 100]
    fig.update_layout(**layout)
    return fig

def macd_fig(sub):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub["Date"], y=sub["macd"], name="MACD", line=dict(color="#7c3aed", width=2)))
    fig.add_trace(go.Scatter(x=sub["Date"], y=sub["macd_signal"], name="Signal", line=dict(color="#a78bfa", width=1.5)))
    fig.add_trace(go.Bar(x=sub["Date"], y=sub["macd_hist"], name="Hist",
                         marker_color=["#059669" if v>=0 else "#e11d48" for v in sub["macd_hist"]], opacity=0.45))
    fig.update_layout(**base_layout(240))
    return fig

def show_indicators(sub):
    latest = sub.iloc[-1]; prev = sub.iloc[-2]
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Close",      f"${latest['Close']:.2f}", f"{((latest['Close']-prev['Close'])/prev['Close']*100):.2f}%")
    c2.metric("RSI",        f"{latest['rsi']:.1f}", "Overbought" if latest['rsi']>70 else ("Oversold" if latest['rsi']<30 else "Neutral"))
    c3.metric("MACD",       f"{latest['macd']:.3f}")
    c4.metric("BB %",       f"{latest['bb_pct']*100:.1f}%")
    c5.metric("Volatility", f"{latest['volatility_20d']*100:.1f}%")

def show_charts(sub):
    st.markdown("<div class='sec-label'>Price Chart</div>", unsafe_allow_html=True)
    st.plotly_chart(candlestick_fig(sub), use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='sec-label'>RSI (14)</div>", unsafe_allow_html=True)
        st.plotly_chart(rsi_fig(sub), use_container_width=True)
    with col2:
        st.markdown("<div class='sec-label'>MACD</div>", unsafe_allow_html=True)
        st.plotly_chart(macd_fig(sub), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("""
    <div class='hero'>
      <div class='hero-tag'>MLOps · Machine Learning · Finance</div>
      <h1>Stock Screener<br>Powered by ML</h1>
      <p>An end-to-end MLOps pipeline that analyzes 15 major US stocks using technical indicators
         and machine learning — generating BUY / HOLD / SELL signals with confidence scores in real time.</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class='sec-label'>About</div>
    <div style='background:#faf9ff; border:1px solid #ede9fe; border-radius:14px;
                padding:1.5rem 2rem; color:#1f2937; line-height:1.8; font-size:0.95rem; margin-bottom:1.5rem;'>
        <b style='color:#5b21b6;'>StockSense</b> computes <b>14 technical indicators</b> from daily OHLCV price data
        and feeds them into 3 machine learning models trained to predict whether a stock will go up in the next 5 days.
        Each stock receives a probability score converted into a
        <b style='color:#059669;'>BUY</b>, <b style='color:#d97706;'>HOLD</b>, or <b style='color:#dc2626;'>SELL</b> signal.
        You can also upload <b>any CSV</b> from Yahoo Finance to analyze it instantly.
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='sec-label'>Pipeline</div>", unsafe_allow_html=True)
    cols = st.columns(5)
    for col, icon, title, desc in zip(cols, [
        "bi-database","bi-cpu","bi-diagram-3","bi-send","bi-bar-chart"],[
        "1. Data","2. Features","3. Training","4. Signals","5. Dashboard"],[
        "Daily OHLCV for 15 tickers loaded from SQLite",
        "RSI, MACD, Bollinger Bands, returns, volatility, volume",
        "3 models with TimeSeriesSplit cross-validation",
        "Probability → BUY / HOLD / SELL per ticker",
        "Interactive charts, screener table and CSV upload"]):
        col.markdown(f"""<div class='feat-card'>
          <div class='feat-icon'><i class='{icon}'></i></div>
          <div class='feat-title'>{title}</div>
          <div class='feat-desc'>{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='sec-label'>Pages</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    for i, (icon, title, desc) in enumerate([
        ("bi-table",     "Screener",     "BUY/HOLD/SELL signals for all 15 stocks with confidence bar chart and color-coded table"),
        ("bi-upload",    "Custom CSV",   "Upload any stock CSV from Yahoo Finance and get candlestick, RSI, MACD and ML signal"),
        ("bi-graph-up",  "Stock Detail", "Candlestick with MA20/MA50, RSI with overbought zones, MACD histogram per ticker"),
        ("bi-lightning", "Train Models", "Train Random Forest, Gradient Boosting and Logistic Regression — compare accuracy and AUC"),
    ]):
        (c1 if i%2==0 else c2).markdown(f"""<div class='page-card'>
          <i class='{icon}' style='color:#7c3aed; font-size:1.1rem;'></i>
          <div class='page-card-title'>{title}</div>
          <div class='page-card-desc'>{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='sec-label'>Tech Stack</div>", unsafe_allow_html=True)
    tcols = st.columns(4)
    for i, (icon, name, role) in enumerate([
        ("bi-filetype-py","Python 3.11","Core language"),
        ("bi-layout-text-window","Streamlit","Dashboard"),
        ("bi-cpu","scikit-learn","ML models"),
        ("bi-table","pandas / numpy","Data"),
        ("bi-bar-chart-line","Plotly","Charts"),
        ("bi-hdd","SQLite","Database"),
        ("bi-graph-up-arrow","yfinance","Market data"),
        ("bi-save","joblib","Model saving"),
    ]):
        tcols[i%4].markdown(f"""<div class='tech-badge'>
          <i class='{icon}' style='font-size:1.25rem; color:#7c3aed;'></i>
          <div class='tech-name'>{name}</div>
          <div class='tech-role'>{role}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='sec-label'>Stocks Covered</div>", unsafe_allow_html=True)
    scols = st.columns(5)
    for col, (sector, tickers) in zip(scols, {
        "Technology":["AAPL","MSFT","GOOGL","AMZN","META","NVDA"],
        "EV":["TSLA"], "Finance":["JPM","GS","BAC"],
        "Energy":["XOM","CVX"], "Healthcare":["JNJ","PFE","UNH"],
    }.items()):
        col.markdown(f"""<div style='background:#faf9ff; border:1px solid #ede9fe; border-radius:12px; padding:1rem;'>
          <div style='font-size:0.7rem; font-weight:700; color:#7c3aed; text-transform:uppercase;
                      letter-spacing:0.1em; margin-bottom:8px;'>{sector}</div>
          {''.join([f"<span class='stock-tag'>{t}</span>" for t in tickers])}
        </div>""", unsafe_allow_html=True)

    st.markdown("""<div class='cta-box'>
      <h3>Ready to start?</h3>
      <p>Go to <b>Train Models</b> to build the ML pipeline, then open <b>Screener</b> to see live signals.</p>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SCREENER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Screener":
    st.markdown("""<div class='hero'>
      <div class='hero-tag'>Live Signals</div>
      <h1>Stock Screener</h1>
      <p>ML-powered BUY / HOLD / SELL signals for 15 major US stocks</p>
    </div>""", unsafe_allow_html=True)

    if not models_exist:
        st.warning("No trained model found. Go to **Train Models** first.")
        st.stop()

    with st.spinner("Generating signals..."):
        signals = get_signals()

    c1,c2,c3,c4 = st.columns(4)
    for col, icon, val, label in [
        (c1,"bi-bar-chart-fill",len(signals),"Stocks Screened"),
        (c2,"bi-arrow-up-circle-fill",len(signals[signals["Signal"]=="BUY"]),"BUY"),
        (c3,"bi-dash-circle-fill",len(signals[signals["Signal"]=="HOLD"]),"HOLD"),
        (c4,"bi-arrow-down-circle-fill",len(signals[signals["Signal"]=="SELL"]),"SELL"),
    ]:
        col.markdown(f"""<div class='kpi'>
          <div class='kpi-icon'><i class='{icon}'></i></div>
          <div class='kpi-val'>{val}</div>
          <div class='kpi-lbl'>{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='sec-label'>Confidence by Ticker</div>", unsafe_allow_html=True)
    filter_sig = st.multiselect("Filter signals", ["BUY","HOLD","SELL"], default=["BUY","HOLD","SELL"])
    filtered = signals[signals["Signal"].isin(filter_sig)]

    fig = go.Figure()
    for sig, grp in filtered.groupby("Signal"):
        c = {"BUY":"#059669","HOLD":"#d97706","SELL":"#dc2626"}.get(sig,"#7c3aed")
        fig.add_trace(go.Bar(y=grp["Ticker"], x=grp["Confidence"], orientation="h", name=sig,
                             marker_color=c, text=[f"{v:.1f}%" for v in grp["Confidence"]], textposition="inside"))
    layout = base_layout(380)
    layout["barmode"] = "group"
    layout["xaxis"]["title"] = "Confidence (%)"
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='sec-label'>Signal Table</div>", unsafe_allow_html=True)
    def color_sig(v):
        m = {"BUY":("059669","ecfdf5","a7f3d0"),"HOLD":("d97706","fffbeb","fde68a"),"SELL":("dc2626","fef2f2","fecaca")}
        if v in m:
            fg,bg,bd = m[v]
            return f"color:#{fg};background:#{bg};border:1px solid #{bd};font-weight:600;border-radius:12px;padding:2px 10px;"
        return ""
    st.dataframe(
        filtered.style.applymap(color_sig, subset=["Signal"])
                      .format({"Confidence":"{:.1f}%","Close":"${:.2f}","RSI":"{:.1f}","Volatility":"{:.1f}%"}),
        use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSV
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📂 Custom CSV":
    st.markdown("""<div class='hero'>
      <div class='hero-tag'>Upload & Analyze</div>
      <h1>Custom CSV Analysis</h1>
      <p>Upload any stock CSV and get instant technical analysis with ML signals</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class='upload-hint'>
      <i class='bi bi-cloud-upload' style='font-size:2rem; color:#7c3aed;'></i>
      <div style='font-weight:600; color:#374151; margin:8px 0 4px; font-size:0.95rem;'>Drop your CSV file here</div>
      <div style='color:#6b7280; font-size:0.83rem;'>Columns needed: Date · Open · High · Low · Close · Volume<br>
      Download from Yahoo Finance → Historical Data → Download</div>
    </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    if uploaded:
        df_raw, err = detect_and_load_csv(uploaded)
        if err:
            st.error(f"Column error: {err}")
        else:
            st.success(f"Loaded {len(df_raw):,} rows · {df_raw['Date'].min().date()} → {df_raw['Date'].max().date()}")
            from pipeline.features import compute_features
            with st.spinner("Computing indicators..."):
                df_feat = compute_features(df_raw)
            show_indicators(df_feat)
            show_charts(df_feat)

            if models_exist:
                try:
                    import joblib
                    from model.train import FEATURE_COLS
                    model = joblib.load(os.path.join(MODEL_DIR,"RandomForest.pkl"))
                    prob = model.predict_proba(df_feat.iloc[[-1]][FEATURE_COLS])[0,1]
                    sig = "BUY" if prob>0.6 else ("SELL" if prob<0.4 else "HOLD")
                    col_sig = {"BUY":"059669","HOLD":"d97706","SELL":"dc2626"}[sig]
                    st.markdown(f"""<div class='sec-label'>ML Signal</div>
                    <div style='background:#faf9ff; border:1px solid #ede9fe; border-radius:14px;
                                padding:1.25rem 1.5rem; display:inline-block; min-width:220px;'>
                      <div style='font-size:0.7rem; color:#6b7280; text-transform:uppercase; letter-spacing:0.08em; font-weight:500;'>Prediction</div>
                      <div style='font-size:2.4rem; font-weight:800; color:#{col_sig}; line-height:1.2; letter-spacing:-0.02em;'>{sig}</div>
                      <div style='font-size:0.85rem; color:#374151;'>Confidence: <b style='color:#{col_sig};'>{prob*100:.1f}%</b></div>
                    </div>""", unsafe_allow_html=True)
                except: pass

            csv_out = df_feat.to_csv(index=False).encode("utf-8")
            st.download_button("Download processed data with indicators", csv_out,
                               file_name="indicators.csv", mime="text/csv")
    else:
        st.markdown("<div class='sec-label'>Expected Format</div>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Date":["2024-01-02","2024-01-03"],"Open":[185.2,184.0],
            "High":[186.5,185.5],"Low":[183.8,183.1],"Close":[185.9,184.4],"Volume":[55000000,48000000]
        }), use_container_width=True, hide_index=True)
        st.info("Download stock data from Yahoo Finance → search any stock → Historical Data → Download CSV")

# ══════════════════════════════════════════════════════════════════════════════
# STOCK DETAIL
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Stock Detail":
    st.markdown("""<div class='hero'>
      <div class='hero-tag'>Technical Analysis</div>
      <h1>Stock Detail</h1>
      <p>Deep dive into individual stock charts and indicators</p>
    </div>""", unsafe_allow_html=True)

    from pipeline.fetch_data import TICKERS
    ticker = st.selectbox("Select ticker", TICKERS)
    with st.spinner(f"Loading {ticker}..."):
        sub = get_stock_data(ticker)
    show_indicators(sub)
    show_charts(sub)

# ══════════════════════════════════════════════════════════════════════════════
# TRAIN MODELS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Train Models":
    st.markdown("""<div class='hero'>
      <div class='hero-tag'>MLOps</div>
      <h1>Model Training</h1>
      <p>Train 3 ML models to predict 5-day stock price direction</p>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    for col, icon, title, desc in [
        (c1,"bi-diagram-3","Random Forest","Ensemble of 100 decision trees. Robust and handles non-linear patterns well."),
        (c2,"bi-lightning-fill","Gradient Boosting","Trees built sequentially — each one corrects the previous one's errors."),
        (c3,"bi-exclude","Logistic Regression","Fast linear baseline that finds a mathematical boundary between classes."),
    ]:
        col.markdown(f"""<div class='feat-card'>
          <div class='feat-icon'><i class='{icon}'></i></div>
          <div class='feat-title'>{title}</div>
          <div class='feat-desc'>{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Train All Models", type="primary", use_container_width=True):
        with st.spinner("Training... this takes 1–2 minutes"):
            try:
                results = train_models()
                st.success("All models trained and saved successfully.")
                st.dataframe(results, use_container_width=True, hide_index=True)
                get_signals.clear()
            except Exception as e:
                st.error(f"Training failed: {e}")
                st.exception(e)
    elif models_exist:
        st.success("Models already trained — go to Screener to see the signals.")