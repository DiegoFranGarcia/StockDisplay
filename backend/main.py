from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
from typing import List, Optional
import time
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool for running yfinance in background
executor = ThreadPoolExecutor(max_workers=3)

# ===== CACHING SYSTEM =====
CACHE_DURATION = 600  # 10 minutes in seconds (reduced API calls)
stock_cache = {}
history_cache = {}

def get_cached_stock(symbol: str) -> Optional[dict]:
    """Get stock from cache if not expired"""
    if symbol in stock_cache:
        cached = stock_cache[symbol]
        age = time.time() - cached["timestamp"]
        if age < CACHE_DURATION:
            print(f"🔵 CACHE HIT for {symbol} (age: {int(age)}s)")
            return cached["data"]
        else:
            print(f"🟡 CACHE EXPIRED for {symbol} (age: {int(age)}s)")
    return None

def cache_stock(symbol: str, data: dict):
    """Store stock in cache"""
    stock_cache[symbol] = {
        "data": data,
        "timestamp": time.time()
    }
    print(f"💾 CACHED {symbol} for {CACHE_DURATION}s")

def get_cached_history(symbol: str, period: str, interval: str) -> Optional[list]:
    """Get history from cache if not expired"""
    cache_key = (symbol, period, interval)
    if cache_key in history_cache:
        cached = history_cache[cache_key]
        age = time.time() - cached["timestamp"]
        if age < CACHE_DURATION:
            print(f"🔵 CACHE HIT for {symbol} history (age: {int(age)}s)")
            return cached["data"]
    return None

def cache_history(symbol: str, period: str, interval: str, data: list):
    """Store history in cache"""
    cache_key = (symbol, period, interval)
    history_cache[cache_key] = {
        "data": data,
        "timestamp": time.time()
    }
    print(f"💾 CACHED {symbol} history for {CACHE_DURATION}s")

# ===== RATE LIMITING =====
last_api_call = 0
MIN_DELAY_BETWEEN_CALLS = 1.0  # 1 second between calls for safety

async def rate_limit():
    """Enforce minimum delay between API calls"""
    global last_api_call
    current_time = time.time()
    time_since_last = current_time - last_api_call
    
    if time_since_last < MIN_DELAY_BETWEEN_CALLS:
        wait_time = MIN_DELAY_BETWEEN_CALLS - time_since_last
        await asyncio.sleep(wait_time)
    
    last_api_call = time.time()

# ===== YFINANCE FUNCTIONS =====
def fetch_stock_sync(symbol: str):
    """Synchronous fetch for yfinance"""
    try:
        ticker = yf.Ticker(symbol)
        
        # Get recent history
        hist = ticker.history(period="5d")
        
        if hist.empty or len(hist) == 0:
            print(f"❌ {symbol}: No data returned (trying 1mo period...)")
            hist = ticker.history(period="1mo")
            
            if hist.empty or len(hist) == 0:
                print(f"❌ {symbol}: Still no data")
                return None
        
        current_price = float(hist['Close'].iloc[-1])
        
        if len(hist) >= 2:
            previous_close = float(hist['Close'].iloc[-2])
        else:
            previous_close = current_price
        
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
        print(f"❌ Error fetching {symbol}: {type(e).__name__}: {str(e)}")
        return None


async def fetch_stock_data(symbol: str):
    """Async wrapper for yfinance with caching"""
    # Check cache first
    cached = get_cached_stock(symbol)
    if cached:
        return cached
    
    # Enforce rate limiting
    await rate_limit()
    
    # Fetch from API
    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(executor, fetch_stock_sync, symbol)
    
    # Cache the result
    if data:
        cache_stock(symbol, data)
    
    return data


def fetch_history_sync(symbol: str, period: str, interval: str):
    """Synchronous history fetch"""
    try:
        ticker = yf.Ticker(symbol)
        
        # Validate interval for period
        valid_intervals = {
            "5d": ["5m", "15m", "30m", "1h"],
            "1mo": ["1d", "1h"],
            "3mo": ["1d"],
            "1y": ["1d", "1wk"],
        }
        
        if period in valid_intervals and interval not in valid_intervals[period]:
            if period == "5d":
                interval = "15m"
            else:
                interval = "1d"
            print(f"⚠️ Adjusted interval to {interval} for period {period}")
        
        print(f"📊 Fetching {symbol} history: period={period}, interval={interval}")
        hist = ticker.history(period=period, interval=interval)
        
        if hist.empty:
            print(f"❌ {symbol}: Empty history returned")
            return None
        
        history = []
        for date, row in hist.iterrows():
            if interval in ["5m", "15m", "30m", "1h"]:
                date_str = date.strftime("%Y-%m-%d %H:%M")
            else:
                date_str = date.strftime("%Y-%m-%d")
            
            history.append({
                "date": date_str,
                "price": round(float(row['Close']), 2),
                "volume": int(row['Volume']) if 'Volume' in row else 0
            })
        
        print(f"✅ {symbol}: Got {len(history)} data points")
        return history
        
    except Exception as e:
        print(f"❌ Error fetching history for {symbol}: {type(e).__name__}: {str(e)}")
        return None


async def fetch_stock_history(symbol: str, period: str, interval: str):
    """Async wrapper for history with caching"""
    # Check cache first
    cached = get_cached_history(symbol, period, interval)
    if cached:
        return cached
    
    # Enforce rate limiting
    await rate_limit()
    
    # Fetch from API
    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(executor, fetch_history_sync, symbol, period, interval)
    
    # Cache the result
    if data:
        cache_history(symbol, period, interval, data)
    
    return data


@app.get("/")
def root():
    cache_info = {
        "cached_stocks": len(stock_cache),
        "cached_histories": len(history_cache)
    }
    return {
        "message": "Stock API with yfinance (UPDATED VERSION - CACHED)",
        "status": "ok",
        "cache_duration": f"{CACHE_DURATION}s (5 minutes)",
        "rate_limit": f"{MIN_DELAY_BETWEEN_CALLS}s between API calls",
        "max_stocks": "10 recommended",
        "cache_info": cache_info,
        "tip": "Make sure yahoo.com is whitelisted on Pi-hole!",
        "yfinance_note": "Updated version without session override"
    }


@app.get("/stock/{symbol}")
async def get_stock(symbol: str):
    """Get stock data"""
    data = await fetch_stock_data(symbol)
    
    if data:
        return data
    else:
        raise HTTPException(
            status_code=503, 
            detail=f"Unable to fetch data for {symbol}. Check if yahoo.com is whitelisted on Pi-hole."
        )


@app.get("/stocks")
async def get_multiple_stocks(symbols: str):
    """Get multiple stocks - LIMITED TO 10"""
    symbol_list = [s.strip().upper() for s in symbols.split(',') if s.strip()]
    
    if not symbol_list:
        raise HTTPException(status_code=400, detail="No symbols provided")
    
    if len(symbol_list) > 10:
        raise HTTPException(
            status_code=400,
            detail=f"Too many symbols ({len(symbol_list)}). Maximum is 10."
        )
    
    results = []
    
    print(f"\n{'='*60}")
    print(f"Fetching {len(symbol_list)} stocks from yfinance (updated version)...")
    print(f"{'='*60}")
    
    for i, symbol in enumerate(symbol_list):
        print(f"\n[{i+1}/{len(symbol_list)}] Fetching {symbol}...")
        
        data = await fetch_stock_data(symbol)
        
        if data:
            results.append(data)
            print(f"✅ {symbol}: ${data['price']} ({data['change_percent']:+.2f}%)")
        else:
            print(f"❌ {symbol}: Failed to fetch")
    
    print(f"\n{'='*60}")
    print(f"✅ Successfully fetched {len(results)}/{len(symbol_list)} stocks")
    print(f"💾 Cache stats: {len(stock_cache)} stocks cached")
    print(f"{'='*60}\n")
    
    return results


@app.get("/stock/{symbol}/history")
async def get_stock_history(symbol: str, period: str = "1d", interval: str = "5m"):
    """Get historical data"""
    try:
        yf_period_map = {
            "1d": "5d",
            "5d": "5d",
            "1mo": "1mo",
            "3mo": "3mo",
            "1y": "1y"
        }
        
        yf_period = yf_period_map.get(period, "5d")
        
        print(f"\n📊 Fetching history for {symbol}")
        print(f"   Period: {period} → yfinance: {yf_period}")
        print(f"   Interval: {interval}")
        
        history = await fetch_stock_history(symbol, yf_period, interval)
        
        if history is None or len(history) == 0:
            raise HTTPException(
                status_code=503,
                detail=f"No historical data available for {symbol}"
            )
        
        return {
            "symbol": symbol.upper(),
            "history": history,
            "period": period,
            "interval": interval
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Unexpected error: {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/test")
async def test():
    """Test yfinance connection"""
    print("\n" + "="*60)
    print("🧪 Testing yfinance Connection (updated version)")
    print("="*60)
    
    test_symbol = "AAPL"
    print(f"\nTesting with {test_symbol}...")
    
    test_data = await fetch_stock_data(test_symbol)
    
    if test_data:
        print(f"✅ Current data works: ${test_data['price']}")
        
        test_history = await fetch_stock_history(test_symbol, "5d", "15m")
        
        if test_history:
            print(f"✅ Historical data works: {len(test_history)} data points")
            return {
                "status": "✅ working",
                "current_data": test_data,
                "history_points": len(test_history),
                "cache_info": {
                    "stocks_cached": len(stock_cache),
                    "histories_cached": len(history_cache),
                    "cache_duration": f"{CACHE_DURATION}s"
                },
                "yfinance_version": "Updated - no session override"
            }
        else:
            return {
                "status": "⚠️ partial",
                "current_data": test_data,
                "history_error": "Failed to fetch history"
            }
    else:
        print("❌ yfinance connection failed")
        return {
            "status": "❌ error",
            "message": "yfinance connection failed - check Pi-hole whitelist",
            "suggestion": "Make sure yahoo.com is whitelisted on your Pi-hole"
        }


@app.get("/cache/clear")
async def clear_cache():
    """Clear all caches"""
    global stock_cache, history_cache
    stocks_cleared = len(stock_cache)
    histories_cleared = len(history_cache)
    
    stock_cache = {}
    history_cache = {}
    
    return {
        "message": "Cache cleared",
        "stocks_cleared": stocks_cleared,
        "histories_cleared": histories_cleared
    }


@app.get("/cache/stats")
async def cache_stats():
    """View cache statistics"""
    stock_ages = {}
    for symbol, cached in stock_cache.items():
        age = time.time() - cached["timestamp"]
        stock_ages[symbol] = f"{int(age)}s ago"
    
    return {
        "cache_duration": f"{CACHE_DURATION}s",
        "stocks_cached": len(stock_cache),
        "histories_cached": len(history_cache),
        "stock_ages": stock_ages,
        "rate_limit": f"{MIN_DELAY_BETWEEN_CALLS}s between API calls"
    }