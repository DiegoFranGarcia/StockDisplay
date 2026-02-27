# Stock Dashboard

A live stock tracking dashboard with an ML-powered next-day price direction predictor. Search any ticker, view interactive charts with adjustable time intervals, and get a machine learning prediction for tomorrow's movement when you click on a stock card.

**Author:** Diego Garcia

---

## Features

- **Live stock data** — fetches real-time price, change, and volume via yfinance with in-memory caching (10-minute TTL)
- **Interactive charts** — Chart.js price history with selectable periods (1D, 5D, 1M, 3M, 1Y)
- **ML predictions** — Random Forest model predicts next-day direction (UP/DOWN) with a confidence score
- **Historical storage** — PostgreSQL database stores OHLCV history and past predictions
- **Up to 10 symbols** — comma-separated search (e.g. `AAPL, NVDA, MSFT`)

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML, CSS, JavaScript, Chart.js |
| Backend | FastAPI, Python |
| Data | yfinance |
| ML | scikit-learn (Random Forest), joblib |
| Database | PostgreSQL, SQLAlchemy |

---

## Project Structure

```
StockDisplay/
├── backend/
│   ├── main.py           # FastAPI app and all API endpoints
│   ├── database.py       # SQLAlchemy models and DB helpers
│   ├── ml_predictor.py   # Feature engineering and prediction logic
│   ├── stock_predictor.pkl  # Trained Random Forest model
│   ├── requirements.txt
│   └── .env              # DATABASE_URL (not committed)
└── frontend/
    ├── index.html
    ├── script.js
    └── style.css
```

---

## Setup

### Prerequisites

- Python 3.10+
- PostgreSQL

### 1. Configure environment

Create `backend/.env`:

```
DATABASE_URL=postgresql://user:password@localhost:5432/stockdb
```

### 2. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 3. Run the backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

### 4. Open the frontend

Open `frontend/index.html` in your browser, or serve it with any static file server.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/stocks?symbols=AAPL,MSFT` | Fetch live data for up to 10 symbols |
| GET | `/stock/{symbol}/history` | Price history from yfinance (period/interval params) |
| GET | `/stock/{symbol}/backfill` | Fetch and store up to 5 years of OHLCV history in DB |
| GET | `/db/stock/{symbol}/history` | Retrieve stored history from DB |
| POST | `/predict/{symbol}` | Run ML model and return tomorrow's prediction |
| GET | `/health` | Database health check |

---

## ML Model

The predictor uses a Random Forest classifier trained on rolling-window features:

- **Close ratio** — current close vs. rolling average over 2, 5, 60, 250, and 1000 days
- **Trend** — sum of past `Target` values (1 = up day, 0 = down day) over the same horizons
- **Threshold** — model predicts UP only when confidence ≥ 60%

A minimum of 1,000 days of stored history is required to generate predictions for a symbol. Backfill first:

```
GET /stock/AAPL/backfill
```

Then request a prediction:

```
POST /predict/AAPL
```
