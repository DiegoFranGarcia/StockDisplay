"""
Integrated Stock API with ML Predictions and Database Storage
Combines real-time data, historical storage, and ML predictions
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import yfinance as yf
from typing import List, Optional
import time
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, func
from sqlalchemy.orm import sessionmaker, Session, declarative_base
import os
import joblib
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

print(f"Connecting to database: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else 'localhost'}")

# Create database engine
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# DATABASE MODELS 
class StockHistory(Base):
    """Store historical price data for ML training"""
    __tablename__ = "stock_history"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), index=True, nullable=False)
    date = Column(String(20), index=True, nullable=False)
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    tomorrow = Column(Float)
    target = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)


class MLPrediction(Base):
    """Store ML model predictions"""
    __tablename__ = "ml_predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), index=True, nullable=False)
    prediction_date = Column(DateTime, nullable=False)
    predicted_direction = Column(String(10), nullable=False)
    confidence = Column(Float, nullable=False)
    predicted_price = Column(Float)
    model_version = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)


# ===== DATABASE HELPERS =====
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables"""
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created successfully")
        
        # Test connection
        db = SessionLocal()
        try:
            from sqlalchemy import text
            db.execute(text("SELECT 1"))
            print("✅ Database connection successful")
        finally:
            db.close()
    except Exception as e:
        print(f"❌ Database initialization error: {e}")
        raise


# ===== LIFESPAN MANAGEMENT =====
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()
    yield
    # Shutdown
    pass


# ===== FASTAPI APP =====
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool for yfinance
executor = ThreadPoolExecutor(max_workers=3)

# Caching
CACHE_DURATION = 600
stock_cache = {}


# ===== CACHING FUNCTIONS =====
def get_cached_stock(symbol: str) -> Optional[dict]:
    if symbol in stock_cache:
        cached = stock_cache[symbol]
        age = time.time() - cached["timestamp"]
        if age < CACHE_DURATION:
            return cached["data"]
    return None


def cache_stock(symbol: str, data: dict):
    stock_cache[symbol] = {
        "data": data,
        "timestamp": time.time()
    }


# ===== YFINANCE FUNCTIONS =====
def fetch_stock_sync(symbol: str):
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="5d")
        
        if hist.empty or len(hist) == 0:
            return None
        
        current_price = float(hist['Close'].iloc[-1])
        previous_close = float(hist['Close'].iloc[-2]) if len(hist) >= 2 else current_price
        
        change = current_price - previous_close
        change_percent = (change / previous_close * 100) if previous_close else 0
        volume = int(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns else 0
        
        return {
            "symbol": symbol.upper(),
            "price": round(current_price, 2),
            "change": round(change, 2),
            "change_percent": round(change_percent, 2),
            "volume": volume
        }
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None


async def fetch_stock_data(symbol: str):
    cached = get_cached_stock(symbol)
    if cached:
        return cached
    
    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(executor, fetch_stock_sync, symbol)
    
    if data:
        cache_stock(symbol, data)
    
    return data


# ===== ML PREDICTION FUNCTIONS =====
def calculate_features(df):
    """Calculate rolling average ratios and trends"""
    horizons = [2, 5, 60, 250, 1000]
    
    for horizon in horizons:
        rolling_averages = df.rolling(horizon).mean()
        ratio_column = f'Close_Ratio_{horizon}'
        df[ratio_column] = df['Close'] / rolling_averages['Close']
        
        trend_column = f'Trend_{horizon}'
        df[trend_column] = df.shift(1).rolling(horizon).sum()["Target"]
    
    return df


def prepare_data_for_prediction(db: Session, symbol: str):
    """Fetch and prepare data for ML prediction"""
    # Get historical data from database
    records = db.query(StockHistory).filter(
        StockHistory.symbol == symbol.upper()
    ).order_by(StockHistory.date.desc()).limit(1200).all()
    
    if len(records) < 1000:
        raise ValueError(
            f"Need at least 1000 days of history for {symbol}, "
            f"got {len(records)}. Run /stock/{symbol}/backfill first."
        )
    
    # Convert to DataFrame
    df = pd.DataFrame([{
        'Date': r.date,
        'Open': r.open_price,
        'High': r.high_price,
        'Low': r.low_price,
        'Close': r.close_price,
        'Volume': r.volume,
        'Target': r.target if r.target is not None else 0
    } for r in reversed(records)])
    
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Calculate features
    df = calculate_features(df)
    df = df.dropna()
    
    return df


# ===== API ENDPOINTS =====
@app.get("/")
def root():
    return {
        "message": "Stock API with ML",
        "status": "ok",
        "database": "connected",
        "endpoints": {
            "stocks": "/stocks?symbols=AAPL,MSFT",
            "history": "/stock/{symbol}/history",
            "backfill": "/stock/{symbol}/backfill",
            "predict": "/predict/{symbol}",
            "db_history": "/db/stock/{symbol}/history"
        }
    }


@app.get("/health")
def health_check(db: Session = Depends(get_db)):
    """Check database health"""
    try:
        from sqlalchemy import text
        db.execute(text("SELECT 1"))
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/stocks")
async def get_multiple_stocks(symbols: str):
    symbol_list = [s.strip().upper() for s in symbols.split(',') if s.strip()]
    
    if not symbol_list:
        raise HTTPException(status_code=400, detail="No symbols provided")
    
    if len(symbol_list) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 symbols")
    
    results = []
    for symbol in symbol_list:
        data = await fetch_stock_data(symbol)
        if data:
            results.append(data)
    
    return results


@app.get("/stock/{symbol}/backfill")
async def backfill_stock_data(symbol: str, years: int = 5, db: Session = Depends(get_db)):
    """Fetch and store historical data"""
    symbol = symbol.upper()
    
    try:
        # Check latest date in DB
        latest = db.query(func.max(StockHistory.date)).filter(
            StockHistory.symbol == symbol
        ).scalar()
        
        # Fetch data
        ticker = yf.Ticker(symbol)
        if latest:
            start_date = datetime.strptime(latest, '%Y-%m-%d')
            hist = ticker.history(start=start_date)
            action = "update"
        else:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years*365)
            hist = ticker.history(start=start_date, end=end_date)
            action = "backfill"
        
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")
        
        # Save to database
        saved = 0
        for i in range(len(hist)):
            date_str = hist.index[i].strftime('%Y-%m-%d')
            
            # Check if exists
            exists = db.query(StockHistory).filter(
                StockHistory.symbol == symbol,
                StockHistory.date == date_str
            ).first()
            
            if not exists:
                tomorrow = float(hist['Close'].iloc[i + 1]) if i < len(hist) - 1 else None
                target = 1 if (tomorrow and tomorrow > hist['Close'].iloc[i]) else 0 if tomorrow else None
                
                record = StockHistory(
                    symbol=symbol,
                    date=date_str,
                    open_price=float(hist['Open'].iloc[i]),
                    high_price=float(hist['High'].iloc[i]),
                    low_price=float(hist['Low'].iloc[i]),
                    close_price=float(hist['Close'].iloc[i]),
                    volume=int(hist['Volume'].iloc[i]),
                    tomorrow=tomorrow,
                    target=target
                )
                db.add(record)
                saved += 1
        
        db.commit()
        
        return {
            "status": "success",
            "symbol": symbol,
            "action": action,
            "records_fetched": len(hist),
            "records_saved": saved,
            "date_range": {
                "start": hist.index[0].strftime('%Y-%m-%d'),
                "end": hist.index[-1].strftime('%Y-%m-%d')
            }
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/db/stock/{symbol}/history")
def get_db_history(symbol: str, limit: int = 1000, db: Session = Depends(get_db)):
    """Get historical data from database"""
    records = db.query(StockHistory).filter(
        StockHistory.symbol == symbol.upper()
    ).order_by(StockHistory.date.desc()).limit(limit).all()
    
    if not records:
        raise HTTPException(
            status_code=404,
            detail=f"No data for {symbol}. Run /stock/{symbol}/backfill first."
        )
    
    return [{
        "date": r.date,
        "open": r.open_price,
        "high": r.high_price,
        "low": r.low_price,
        "close": r.close_price,
        "volume": r.volume,
        "target": r.target
    } for r in records]


@app.post("/predict/{symbol}")
async def make_prediction(symbol: str, db: Session = Depends(get_db)):
    """Make ML prediction for a stock"""
    try:
        # Load model
        try:
            model = joblib.load('stock_predictor.pkl')
        except FileNotFoundError:
            raise HTTPException(
                status_code=500,
                detail="ML model not found. Train the model first."
            )
        
        # Prepare data
        df = prepare_data_for_prediction(db, symbol)
        latest = df.iloc[-1:]
        
        # Feature columns
        feature_cols = [
            'Close_Ratio_2', 'Trend_2',
            'Close_Ratio_5', 'Trend_5',
            'Close_Ratio_60', 'Trend_60',
            'Close_Ratio_250', 'Trend_250',
            'Close_Ratio_1000', 'Trend_1000'
        ]
        
        # Get prediction
        prob = model.predict_proba(latest[feature_cols])[0]
        confidence = prob[1]
        predicted_direction = "UP" if confidence >= 0.6 else "DOWN"
        current_close = latest['Close'].values[0]
        
        # Store prediction
        prediction = MLPrediction(
            symbol=symbol.upper(),
            prediction_date=datetime.utcnow() + timedelta(days=1),
            predicted_direction=predicted_direction,
            confidence=float(confidence),
            model_version="v1.0"
        )
        db.add(prediction)
        db.commit()
        
        return {
            "symbol": symbol.upper(),
            "current_price": round(current_close, 2),
            "prediction": predicted_direction,
            "confidence": round(confidence * 100, 2),
            "prediction_date": (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d"),
            "model_version": "v1.0",
            "threshold": 0.6
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/stock/{symbol}/history")
async def get_stock_history(symbol: str, period: str = "1mo", interval: str = "1d"):
    """Get stock price history from yfinance"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, interval=interval)
        
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")
        
        history = []
        for i in range(len(hist)):
            date = hist.index[i]
            if interval in ['1m', '5m', '15m', '30m', '1h']:
                date_str = date.strftime('%H:%M')
            else:
                date_str = date.strftime('%Y-%m-%d')
            
            history.append({
                "date": date_str,
                "price": round(float(hist['Close'].iloc[i]), 2)
            })
        
        return {
            "symbol": symbol.upper(),
            "period": period,
            "interval": interval,
            "history": history
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)