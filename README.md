# 📈 Stock Data Intelligence Dashboard

A mini financial data platform for NSE stocks — built with FastAPI, SQLite, yfinance, and Chart.js.

---

## 🏗️ Architecture

```
stock-dashboard/
├── backend/
│   ├── main.py              # FastAPI app + all REST endpoints
│   ├── data_collector.py    # yfinance data fetching
│   ├── data_processor.py    # Pandas transformations + custom metrics
│   ├── database.py          # SQLite + SQLAlchemy ORM
│   └── ml_predictor.py      # Linear Regression price prediction
├── frontend/
│   └── index.html           # Dashboard UI (Chart.js, vanilla JS)
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone & install

```bash
git clone https://github.com/<your-username>/stock-dashboard.git
cd stock-dashboard
pip install -r requirements.txt
```

### 2. Run the server

```bash
cd backend
uvicorn main:app --reload
```

- **API Docs (Swagger):** http://localhost:8000/docs
- **Dashboard:**          http://localhost:8000/static/index.html

> First run fetches 1 year of data for 10 NSE stocks (~30 seconds). Subsequent runs use the cached SQLite DB.

---

## 🐳 Docker

```bash
docker build -t stock-dashboard .
docker run -p 8000:8000 stock-dashboard
```

---

## 📡 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/companies` | GET | List all companies with latest price + sentiment |
| `/data/{symbol}?days=30` | GET | OHLCV + metrics for last N days |
| `/summary/{symbol}` | GET | 52W high/low, avg close, sentiment |
| `/compare?symbol1=X&symbol2=Y&days=60` | GET | Side-by-side comparison + correlation |
| `/movers` | GET | Top 3 gainers and losers today |
| `/predict/{symbol}?days=7` | GET | ML price predictions for next N days |
| `/refresh` | POST | Re-fetch latest data from yfinance |

---

## 📊 Metrics Explained

### Required
| Metric | Formula | Meaning |
|--------|---------|---------|
| Daily Return | `(Close - Open) / Open` | Intraday gain/loss |
| MA 7 | 7-day rolling mean of Close | Short-term trend smoother |
| 52W High/Low | Max/Min close in dataset | Range context |

### Custom Metrics (our additions)
| Metric | Formula | Meaning |
|--------|---------|---------|
| **Volatility 7D** | `std(daily_return, 7) × √252` | Annualised risk — higher = more uncertain |
| **Momentum Score** | `(Close - MA30) / MA30` | Trend direction — positive = upward trend |
| **Sentiment Index** | `0.5 × momentum + 0.3 × daily_return − 0.2 × volatility_norm` | Composite "mood" indicator — clamped to [−1, 1] |

### ML Prediction
- Model: **Linear Regression** (scikit-learn)
- Features: day index, 7-day MA, lagged daily return, momentum score
- Output: next 7 business days of predicted closing prices
- Shown as a dashed green line on the prediction chart

---

## 🎨 Dashboard Features

- **Sidebar** with live prices and daily returns for all 10 stocks
- **KPI cards**: close, return %, 52W H/L, volatility, sentiment
- **4 charts**: Close + MA7, Daily Returns (bar), Volatility, ML Prediction
- **Time filters**: 30D / 90D / 180D / 1Y
- **Top Gainers & Losers** widget
- **Comparison tool**: normalised price chart + correlation score for any two stocks

---

## 🧰 Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11 |
| Backend | FastAPI + Uvicorn |
| Database | SQLite + SQLAlchemy |
| Data | yfinance + Pandas + NumPy |
| ML | scikit-learn (LinearRegression) |
| Frontend | HTML + Vanilla JS + Chart.js |
| Container | Docker |

---

## 💡 Design Decisions

1. **Why SQLite?** Zero-config for a project this size; swap `DATABASE_URL` to a PostgreSQL URL for production.
2. **Why yfinance?** Free, no API key, NSE coverage with `.NS` suffix.
3. **Why Linear Regression for predictions?** Interpretable and lightweight. The goal is a trend line, not a trading signal. Can be upgraded to Prophet or LSTM.
4. **Why normalise in `/compare`?** TCS trades at ₹3,800, INFY at ₹1,500 — raw prices on the same chart are misleading. Base-100 normalisation shows relative performance fairly.
5. **Sentiment Index** is a custom composite metric combining momentum, daily return, and inverse-volatility — useful as a visual "mood indicator" on the dashboard.

---

## 🚀 Possible Extensions

- Add live prices via WebSocket
- Swap SQLite → PostgreSQL + Docker Compose
- Upgrade ML to Facebook Prophet or LSTM
- Add user authentication + watchlists
- Deploy on Render (free tier) or Oracle Cloud Always-Free

---

*Built for Jarnox Internship Assignment — Stock Data Intelligence Dashboard*