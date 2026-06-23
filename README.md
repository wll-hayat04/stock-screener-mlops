<div align="center">



<img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge\&logo=python\&logoColor=white"/>

<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge\&logo=streamlit\&logoColor=white"/>

<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge\&logo=scikit-learn\&logoColor=white"/>

<img src="https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge\&logo=plotly\&logoColor=white"/>

<img src="https://img.shields.io/badge/License-MIT-purple?style=for-the-badge"/>



<br><br>



\# 📈 StockSense — ML Stock Screener



\### An end-to-end MLOps pipeline that screens 15 major US stocks using machine learning  

\### and generates \*\*BUY / HOLD / SELL\*\* signals from technical indicators



<br>



\*\*\[🚀 Live Demo](https://stock-screener-mlops-dyncchxjizeuj47akmozn4.streamlit.app/)\*\* · \*\*\[📂 Source Code](https://github.com/wll-hayat04/stock-screener-mlops)\*\* · \*\*\[👩‍💻 Portfolio](https://github.com/wll-hayat04)\*\*



<br>



!\[StockSense Dashboard](https://via.placeholder.com/900x450/7c3aed/ffffff?text=StockSense+Dashboard+Screenshot)



</div>



\---



\## 📌 Overview



\*\*StockSense\*\* is a complete MLOps project that goes from raw stock data all the way to a deployed interactive dashboard. It computes \*\*14 technical indicators\*\* from daily OHLCV price history, trains \*\*3 machine learning models\*\* with proper time-series cross-validation, and generates a confidence score + signal for each stock.



You can also upload \*\*any custom CSV\*\* from Yahoo Finance to analyze any stock instantly — even ones outside the default 15.



\---



\## ✨ Features



| Feature | Description |

|---------|-------------|

| 📊 \*\*Screener\*\* | BUY / HOLD / SELL signals for 15 stocks with confidence scores and bar chart |

| 📂 \*\*Custom CSV\*\* | Upload any stock CSV — instant candlestick, RSI, MACD and ML signal |

| 📈 \*\*Stock Detail\*\* | Candlestick with MA20/MA50, RSI with overbought zones, MACD histogram |

| 🤖 \*\*Train Models\*\* | Train 3 ML models directly from the dashboard — compare accuracy and AUC |

| 🏠 \*\*Home Page\*\* | Full pipeline explanation, tech stack, stock coverage |



\---



\## 🧠 ML Pipeline



```

SQLite Database (15 tickers · OHLCV daily data)

&#x20;             ↓

&#x20;  Feature Engineering

&#x20;  RSI · MACD · Bollinger Bands · Returns

&#x20;  Moving Averages · Volatility · Volume Ratio

&#x20;             ↓

&#x20;  TimeSeriesSplit Cross-Validation (5 folds)

&#x20;  (no data leakage — always predict future from past)

&#x20;             ↓

&#x20;  3 ML Models Trained \& Compared

&#x20;  Random Forest · Gradient Boosting · Logistic Regression

&#x20;             ↓

&#x20;  BUY / HOLD / SELL Signal + Confidence Score per Ticker

```



\---



\## 📊 Technical Indicators (14 Features)



| Category | Indicators |

|----------|-----------|

| \*\*Returns\*\* | 1d, 5d, 10d, 20d price returns |

| \*\*Moving Averages\*\* | MA10, MA20 ratio · MA10, MA50 ratio |

| \*\*Momentum\*\* | RSI (14) · MACD · MACD Signal · MACD Histogram |

| \*\*Volatility\*\* | Bollinger Band width · BB % position · 20d volatility |

| \*\*Volume\*\* | Volume ratio vs 20d average |



\---



\## 🤖 Models



| Model | Strengths |

|-------|-----------|

| \*\*Random Forest\*\* | Robust, handles non-linear patterns, feature importance |

| \*\*Gradient Boosting\*\* | High accuracy, sequential error correction |

| \*\*Logistic Regression\*\* | Fast, interpretable linear baseline |



All models validated with \*\*TimeSeriesSplit\*\* — no look-ahead bias.



\---



\## 📈 Stocks Covered



| Sector | Tickers |

|--------|---------|

| 💻 Technology | AAPL · MSFT · GOOGL · AMZN · META · NVDA |

| 🚗 EV | TSLA |

| 🏦 Finance | JPM · GS · BAC |

| ⛽ Energy | XOM · CVX |

| 💊 Healthcare | JNJ · PFE · UNH |



\---



\## 🚀 Run Locally



```bash

\# Clone the repo

git clone https://github.com/wll-hayat04/stock-screener-mlops.git

cd stock-screener-mlops



\# Install dependencies

pip install -r requirements.txt



\# Run the dashboard

streamlit run app/dashboard.py

```



Open \[http://localhost:8501](http://localhost:8501) → go to \*\*Train Models\*\* first → then \*\*Screener\*\*.



\---



\## ☁️ Deploy on Streamlit Cloud



1\. Push this repo to GitHub

2\. Go to \[share.streamlit.io](https://share.streamlit.io)

3\. Select repo → branch `main` → main file `app/dashboard.py`

4\. Click \*\*Deploy\*\*



\---



\## 📁 Project Structure



```

stock-screener-mlops/

│

├── app/

│   └── dashboard.py          # Streamlit dashboard (5 pages)

│

├── pipeline/

│   ├── fetch\_data.py         # Load \& parse stock data from SQLite

│   ├── features.py           # Compute 14 technical indicators

│   └── \_\_init\_\_.py

│

├── model/

│   ├── train.py              # Train 3 models with TimeSeriesSplit

│   ├── predict.py            # Generate BUY/HOLD/SELL signals

│   ├── saved/                # Trained .pkl model files

│   └── \_\_init\_\_.py

│

├── data/

│   └── stocks.db             # SQLite database (15 tickers · OHLCV)

│

├── requirements.txt

└── README.md

```



\---



\## 🛠️ Tech Stack



| Tool | Purpose |

|------|---------|

| Python 3.11 | Core language |

| Streamlit | Interactive dashboard |

| scikit-learn | ML models + TimeSeriesSplit |

| pandas / numpy | Data manipulation |

| Plotly | Candlestick + indicator charts |

| SQLite | Local stock database |

| yfinance | Market data source |

| joblib | Model serialization |



\---



\## 👩‍💻 Author



<div align="center">



\*\*Hayat\*\* — 4th Year Engineering Student  

🌍 Morocco / Italy · 💼 Open to freelance \& internships



\[!\[GitHub](https://img.shields.io/badge/GitHub-wll--hayat04-181717?style=for-the-badge\&logo=github)](https://github.com/wll-hayat04)



</div>



\---



\## 📄 License



This project is licensed under the \[MIT License](LICENSE).



\---



<div align="center">

&#x20; <sub>Built by Hayat · If you find this useful, consider giving it a ⭐</sub>

</div>

