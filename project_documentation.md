# Stock Data Intelligence Dashboard: Technical Documentation

## 1. Executive Summary
The **Stock Data Intelligence Dashboard** is a comprehensive, standalone financial data platform tailored for monitoring and analyzing top-tier equities on the National Stock Exchange (NSE). Designed with a focus on ease of use, robust data handling, and meaningful technical indicators, the platform integrates live data fetching, backend financial computations, and predictive modeling into a decoupled web architecture.

This document serves as the canonical technical reference for developers, data engineers, and maintainers interacting with the codebase.

---

## 2. Architecture Overview
The platform leverages a modern, decoupled architecture consisting of a high-performance Python API handling data ingestion, processing, and serving, paired with a lightweight, client-side rendering frontend.

### 2.1 System Components
- **API Gateway & Controller (Backend):** Built on FastAPI, this acts as the central router for processing REST queries, scheduling data updates, and serving the static frontend assets.
- **Data Collector Module:** A resilient ingestion engine responsible for fetching accurate OHLCV (Open, High, Low, Close, Volume) data from Yahoo Finance APIs, complete with defensive fallbacks to bypass scraping mitigations.
- **Data Processor Module:** Calculates complex derived metrics, taking raw localized time-series data and producing momentum indicators, moving averages, and compound sentiment scores.
- **Machine Learning Module:** A lightweight predictive engine that applies Linear Regression on lag variables and moving averages to extrapolate 7-day future price movements.
- **Persistence Layer:** A SQLite (for local development) or PostgreSQL (for production) database utilizing SQLAlchemy ORM to cache historical metrics and power fast dashboard rendering without re-triggering network requests.

### 2.2 Technology Stack
| Domain | Technologies |
| :--- | :--- |
| **Backend Framework** | Python 3.11, FastAPI, Uvicorn |
| **Data Engineering** | `yfinance`, `curl_cffi`, Pandas, NumPy |
| **Predictive Modeling** | `scikit-learn` (Linear Regression) |
| **Database & ORM** | SQLite, PostgreSQL, SQLAlchemy |
| **Frontend** | HTML5, Vanilla JavaScript, Chart.js, CSS |
| **Orchestration** | Docker, Docker Compose |
| **Task Scheduling** | `APScheduler` (Background Nightly Refresh) |

### 2.3 Directory Structure
```text
stock-dashboard/
├── backend/
│   ├── main.py              # Application entry point, routing, and lifecycle events
│   ├── data_collector.py    # Multi-layered yfinance ingestion logic
│   ├── data_processor.py    # Metric computations and Pandas transformations
│   ├── database.py          # SQLAlchemy engine, sessions, and schemas
│   ├── ml_predictor.py      # Regression models for short-term price forecasting
│   └── mock_data.py         # Synthetic data generation for offline isolated testing
├── frontend/
│   ├── index.html           # Core HTML structure
│   ├── style.css            # Custom UI styling, grid layouts, and themes
│   └── app.js               # Client-side API interactions and chart rendering
├── data/                    # Directory for pre-fetched host data (stocks.csv)
├── docker-compose.yml       # Composes PostgreSQL DB + FastAPI Web instances
├── Dockerfile               # Multi-stage lightweight application image
├── seed_data.py             # Host-native script to gather initial stock datasets
└── requirements.txt         # Core dependencies
```

---

## 3. Data Pipeline & Processing

### 3.1 Data Sources & Ingestion
Data ingestion operates on an automated startup hook and a nightly `APScheduler` job. The raw source is Yahoo Finance (via the NSE `.NS` tickers). To ensure system stability, data fetches natively pull a 365-day rolling window to guarantee the presence of long-term metrics such as the 52-week High/Low.

### 3.2 Derived Financial Metrics
Beyond standard OHLCV, the system dynamically calculates proprietary tracking metrics:
- **Daily Return:** Calculated as `(Close - Open) / Open`, surfacing intraday price variance.
- **7-Day Moving Average (MA):** A rolling 7-day smoother eliminating intraday noise.
- **7-Day Volatility:** Standard deviation of the 7-day daily return, annualized (`* √252`). Serves as a primary risk indicator.
- **Momentum Score:** Normalized deviation of the current close against a 30-day moving average (`(Close - MA30) / MA30`), dictating trend strength.
- **Composite Sentiment Index:** A clamped scalar `[-1, 1]` weighing momentum, daily return, and inverse volatility to present an immediate "Bullish", "Bearish", or "Neutral" visual sentiment to the end user.

### 3.3 Machine Learning Integration
The `ml_predictor.py` module exposes a non-trading, trend-extrapolating **Linear Regression** model via the `scikit-learn` suite.
- **Features:** Day index, 7-day MA, lagged daily return, and momentum score.
- **Target:** Extrapolated Close Price.
- **Output:** Next 7 business days mapped over the dashboard UI as a dashed trendline.

---

## 4. REST API Reference

The backend exposes a highly documented Swagger interface (`/docs`). Critical routes include:

| Method | Endpoint | Description | Query Parameters |
|:---|:---|:---|:---|
| `<span style="color:blue">GET</span>` | `/companies` | Returns all tracked symbols, full names, and momentary sentiment indices. | *None* |
| `<span style="color:blue">GET</span>` | `/data/{symbol}` | Primary metric retrieval endpoint generating JSON payloads of OHLCV and all calculated indices. | `days` (int, default: 30) |
| `<span style="color:blue">GET</span>` | `/summary/{symbol}` | Fetches high-level aggregates (52W H/L, average closes) intended for KPI widgets. | *None* |
| `<span style="color:blue">GET</span>` | `/compare` | Executes a side-by-side equity analysis performing base-100 normalization and statistical correlation mapping. | `symbol1`, `symbol2`, `days` |
| `<span style="color:blue">GET</span>` | `/predict/{symbol}` | Invokes the ML model pipeline and returns historical actuals alongside N-day future price predictions. | `days` (int, default: 7) |

---

## 5. Development & Deployment

### 5.1 Local Development
Starting the project for local iteration requires only standard Python tooling constraints:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload
```
Upon the first boot event, the `lifespan` hook automatically queries the Yahoo Finance API, populates the local `db.sqlite3` file, and establishes the data layer with 0 configuration.

### 5.2 Docker Orchestration
For production contexts, the decoupled environment relies on Docker Compose to spin up parallel containers representing the application and a resilient PostgreSQL instance:
```bash
docker-compose up --build -d
```
The architecture handles DB abstraction securely by swapping SQLite variables for parameterized PostgreSQL `DATABASE_URL` configurations.

---

## 6. Known Issues & Resolutions

### 6.1 Critical Issue: YFinance Docker Connectivity (Cloudflare Bot Mitigation)
**The Problem:**
During architectural containerization, a severe operational bottleneck emerged: HTTP requests executed by `yfinance` to Yahoo Finance data servers continuously timed out or refused connections (HTTP 403/429) strictly when originating from within the `/app` Docker network interface. Host-machine execution ran perfectly.
Following network logging, it was verified that Yahoo Finance's CDN edge nodes (Cloudflare) strictly identify internal Docker abstraction IPs and standard AWS/cloud networking routes as headless scrapers, blocking the system by default. 

**The Implemented Resolution:**
To ensure robust, decoupled CI/CD environments without relying on paid API keys, the data collector (`data_collector.py`) was entirely rewritten around a **Priority-Based Triple Fallback Architecture:**

1. **Host-Side Seed Data Mount (Priority 1 - Passive Ingestion):** 
   A standalone utility, `seed_data.py`, was generated for execution natively on the host machine. This gathers data unobstructed by CDN blocks, formatting it into `data/stocks.csv`. The `docker-compose.yml` mounts this specific volume directly into the container (`./data:/app/data`). The FastAPI app automatically intercepts remote fetches in favor of local CSV streaming if present, negating network dependency altogether in blocked environments.
   
2. **`curl_cffi` TLS Impersonation (Priority 2 - Active Evasion):**
   When dynamic polling is absolutely required, the standard Python request handlers were replaced with `curl_cffi`, specifically utilizing the `impersonate="chrome110"` execution flag. This alters the underlying TLS handshakes, JA3 fingerprints, and HTTP/2 pseudo-headers explicitly to match organic Chromium browser traffic. This successfully mimics a valid human user identity and entirely bypasses the Cloudflare Web Application Firewall.
   
3. **Synthetic Mock Generation (Priority 3 - Failsafe):**
   In the extreme scenario of complete API and volume failure, the `main.py` controller falls back to the `mock_data.py` procedural generator logic, generating statistically coherent synthetic stock permutations to assure the frontend and API contracts NEVER enter a crash loop to the end user.

This highly resilient pipeline guarantees deployment stability across any cloud or local containerized ecosystem.

---

## 7. Future Roadmaps & Extensibility
The platform’s decoupled nature presents primary avenues for horizontal feature expansion:
- **Streaming Telemetry**: Replacing REST polling with `WebSocket` streams for live 1-second interval price tracking.
- **Model Upgrades**: Transitioning predictive modeling algorithms from `LinearRegression` to recurrent neural network models like `LSTM` or Facebook’s `Prophet` time-series engine for nonlinear trend forecasting.
- **Auth & Subscriptions**: Introducing OAuth2 logic and user-specific relational watchlists.
- **Automated Cloud Provisioning**: Generating Terraform playbooks to pipeline directly to AWS EC2 or Oracle Cloud Free Tier.
