<div align="center">

<img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white"/>
<img src="https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white"/>
<img src="https://img.shields.io/badge/License-MIT-7c3aed?style=for-the-badge"/>

<br><br>

# 📈 StockSense — ML Stock Screener

**An end-to-end MLOps pipeline that screens 15 major US stocks**  
**and generates BUY / HOLD / SELL signals from technical indicators**

<br>

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Click_Here-7c3aed?style=for-the-badge)](https://stock-screener-mlops-dyncchxjizeuj47akmozn4.streamlit.app/)
[![Source Code](https://img.shields.io/badge/📂_Source_Code-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/wll-hayat04/stock-screener-mlops)
[![Portfolio](https://img.shields.io/badge/👩‍💻_Portfolio-wll--hayat04-blue?style=for-the-badge)](https://github.com/wll-hayat04)

<br>

> ⭐ If you find this project useful, please consider giving it a star!

</div>

---

## 📌 Overview

**StockSense** is a complete end-to-end MLOps project — from raw market data to a deployed interactive dashboard. It computes **14 technical indicators** from daily OHLCV price history, trains **3 machine learning models** with proper time-series cross-validation, and outputs a confidence score + signal for each stock.

You can also upload **any custom CSV** from Yahoo Finance to analyze any stock instantly.

> ⚠️ **Disclaimer:** This project is for educational purposes only. It is not financial advice. Do not make investment decisions based on these signals.

---

## ✨ Features

| Page | Description |
|------|-------------|
| 🏠 **Home** | Full pipeline explanation, tech stack overview, stocks covered |
| 📊 **Screener** | BUY / HOLD / SELL signals for all 15 stocks with confidence bar chart |
| 📂 **Custom CSV** | Upload any stock CSV — instant candlestick, RSI, MACD and ML signal |
| 📈 **Stock Detail** | Candlestick with MA20/MA50, RSI with overbought/oversold zones, MACD histogram |
| 🤖 **Train Models** | Train 3 ML models directly from the dashboard — compare Accuracy, F1 and AUC |

---

## 🧠 ML Pipeline

```
SQLite Database (15 tickers · OHLCV daily data since 2024)
              ↓
   Feature Engineering (14 technical indicators)
   RSI · MACD · Bollinger Bands · Returns (1d/5d/10d/20d)
   Moving Average Ratios · Volatility · Volume Ratio
              ↓
   TimeSeriesSplit Cross-Validation (5 folds)
   → No data leakage · Always predict future from past only
              ↓
   3 ML Models Trained & Evaluated
   Random Forest · Gradient Boosting · Logistic Regression
              ↓
   Probability Score → BUY (>60%) · HOLD (40-60%) · SELL (<40%)
```

---

## 📊 Technical Indicators (14 Features)

| Category | Indicators |
|----------|-----------|
| **Returns** | 1d · 5d · 10d · 20d price returns |
| **Moving Averages** | MA10/MA20 ratio · MA10/MA50 ratio |
| **Momentum** | RSI (14) · MACD · MACD Signal · MACD Histogram |
| **Volatility** | Bollinger Band width · BB % position · 20d annualized volatility |
| **Volume** | Volume ratio vs 20-day moving average |

---

## 🤖 Models & Performance

| Model | Accuracy | F1 Score | ROC-AUC |
|-------|----------|----------|---------|
| **Random Forest** | ~52% | ~60% | ~51% |
| **Gradient Boosting** | ~52% | ~62% | ~51% |
| **Logistic Regression** | ~55% | ~68% | ~51% |

> Performance reflects the inherent difficulty of stock prediction. Slightly above random (50%) is expected and consistent with academic literature on short-term price forecasting.

---

## 📈 Stocks Covered

| Sector | Tickers |
|--------|---------|
| 💻 Technology | `AAPL` `MSFT` `GOOGL` `AMZN` `META` `NVDA` |
| 🚗 EV | `TSLA` |
| 🏦 Finance | `JPM` `GS` `BAC` |
| ⛽ Energy | `XOM` `CVX` |
| 💊 Healthcare | `JNJ` `PFE` `UNH` |

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/wll-hayat04/stock-screener-mlops.git
cd stock-screener-mlops

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app/dashboard.py
```

Open [http://localhost:8501](http://localhost:8501) → go to **Train Models** first → then **Screener**.

---

## ☁️ Deploy on Streamlit Cloud

1. Fork this repo on GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Select repo → branch `main` → main file `app/dashboard.py`
4. Click **Deploy** → your app is live in ~2 minutes

---

## 📁 Project Structure

```
stock-screener-mlops/
│
├── app/
│   └── dashboard.py          # Streamlit dashboard (5 pages)
│
├── pipeline/
│   ├── fetch_data.py         # Load & clean stock data from SQLite
│   ├── features.py           # Compute 14 technical indicators
│   └── __init__.py
│
├── model/
│   ├── train.py              # Train 3 models with TimeSeriesSplit
│   ├── predict.py            # Generate BUY/HOLD/SELL signals
│   ├── saved/                # Trained .pkl model files (auto-generated)
│   └── __init__.py
│
├── data/
│   └── stocks.db             # SQLite database (15 tickers · OHLCV)
│
├── requirements.txt          # Python dependencies
└── README.md
```

---

## 🛠️ Tech Stack

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.11 | Core language |
| Streamlit | Latest | Interactive dashboard |
| scikit-learn | Latest | ML models + TimeSeriesSplit |
| pandas / numpy | Latest | Data manipulation |
| Plotly | Latest | Candlestick + indicator charts |
| SQLite | Built-in | Local stock database |
| yfinance | Latest | Yahoo Finance data source |
| joblib | Latest | Model serialization |

---

## 🔧 How to Use

1. **Clone** the repo and install dependencies
2. Go to **Train Models** tab → click **Train All Models** (takes ~2 min)
3. Go to **Screener** → see BUY/HOLD/SELL signals for all 15 stocks
4. Go to **Stock Detail** → pick any ticker and explore technical charts
5. Go to **Custom CSV** → upload any Yahoo Finance CSV to analyze any stock

---

## 👩‍💻 Author

<div align="center">

**Hayat** — 4th Year Engineering Student  
🌍 Morocco / Italy &nbsp;·&nbsp; 💼 Open to freelance & internships  
🔬 Interested in ML, MLOps, Data Science & Full-Stack

<br>

[![GitHub](https://img.shields.io/badge/GitHub-wll--hayat04-181717?style=for-the-badge&logo=github)](https://github.com/wll-hayat04)

</div>

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) — feel free to use, modify and distribute.

---

<div align="center">
  <sub>Built by Hayat &nbsp;·&nbsp; Star ⭐ this repo if you found it useful!</sub>
</div>