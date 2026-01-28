import pandas as pd
import joblib
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from database import StockHistory, MLPrediction

# Load the trained model
model = joblib.load('stock_predictor.pkl')

# Feature engineering functions (matching your training notebook)
def calculate_features(df):
    """Calculate rolling average ratios and trends"""
    horizons = [2, 5, 60, 250, 1000]
    
    for horizon in horizons:
        rolling_averages = df.rolling(horizon).mean()
        
        # Ratio of current close to rolling average
        ratio_column = f'Close_Ratio_{horizon}'
        df[ratio_column] = df['Close'] / rolling_averages['Close']
        
        # Trend: sum of previous target values
        trend_column = f'Trend_{horizon}'
        df[trend_column] = df.shift(1).rolling(horizon).sum()["Target"]
    
    return df

def prepare_data_for_prediction(db: Session, symbol: str, days_back: int = 1200):
    """
    Fetch historical data and prepare features for prediction
    
    Args:
        db: Database session
        symbol: Stock symbol
        days_back: How many days of history to fetch (needs 1000+ for features)
    
    Returns:
        DataFrame ready for prediction
    """
    # Fetch historical data
    history = db.query(StockHistory).filter(
        StockHistory.symbol == symbol.upper()
    ).order_by(StockHistory.date.desc()).limit(days_back).all()
    
    if not history or len(history) < 1000:
        raise ValueError(f"Need at least 1000 days of history for {symbol}, got {len(history)}")
    
    # Convert to DataFrame
    df = pd.DataFrame([{
        'Date': h.date,
        'Open': h.open_price,
        'High': h.high_price,
        'Low': h.low_price,
        'Close': h.close_price,
        'Volume': h.volume,
        'Target': h.target if h.target is not None else 0
    } for h in reversed(history)])  # Reverse to chronological order
    
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Calculate features
    df = calculate_features(df)
    
    # Drop NaN values (first 1000 rows will have NaNs)
    df = df.dropna()
    
    return df

def make_prediction(db: Session, symbol: str, model_version: str = "v1.0"):
    """
    Make a prediction for tomorrow's stock movement
    
    Args:
        db: Database session
        symbol: Stock symbol
        model_version: Version identifier for the model
    
    Returns:
        Prediction dictionary with direction, confidence, and predicted price
    """
    try:
        # Prepare data
        df = prepare_data_for_prediction(db, symbol)
        
        # Get the most recent row for prediction
        latest = df.iloc[-1:]
        
        # Feature columns (must match training)
        feature_cols = [
            'Close_Ratio_2', 'Trend_2',
            'Close_Ratio_5', 'Trend_5',
            'Close_Ratio_60', 'Trend_60',
            'Close_Ratio_250', 'Trend_250',
            'Close_Ratio_1000', 'Trend_1000'
        ]
        
        # Get prediction probability
        prob = model.predict_proba(latest[feature_cols])[0]
        confidence = prob[1]  # Probability of going UP
        
        # Use threshold of 0.6 (matching your training)
        predicted_direction = "UP" if confidence >= 0.6 else "DOWN"
        
        # Get current close price for reference
        current_close = latest['Close'].values[0]
        
        # Store prediction in database
        prediction = MLPrediction(
            symbol=symbol.upper(),
            prediction_date=datetime.utcnow() + timedelta(days=1),
            predicted_direction=predicted_direction,
            confidence=float(confidence),
            predicted_price=None,  # Could add price prediction later
            model_version=model_version
        )
        db.add(prediction)
        db.commit()
        db.refresh(prediction)
        
        return {
            "symbol": symbol.upper(),
            "current_price": round(current_close, 2),
            "prediction": predicted_direction,
            "confidence": round(confidence * 100, 2),
            "prediction_date": (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d"),
            "model_version": model_version,
            "threshold": 0.6
        }
        
    except Exception as e:
        print(f"Error making prediction for {symbol}: {e}")
        raise

def evaluate_past_predictions(db: Session, days_back: int = 30):
    """
    Evaluate accuracy of predictions made in the past
    
    Args:
        db: Database session
        days_back: How many days back to evaluate
    
    Returns:
        Evaluation metrics
    """
    cutoff_date = datetime.utcnow() - timedelta(days=days_back)
    
    # Get predictions that should have results by now
    predictions = db.query(MLPrediction).filter(
        MLPrediction.prediction_date < datetime.utcnow(),
        MLPrediction.prediction_date > cutoff_date,
        MLPrediction.actual_price.is_(None)  # Not yet evaluated
    ).all()
    
    evaluated = 0
    for pred in predictions:
        # Get actual price for prediction date
        pred_date_str = pred.prediction_date.strftime("%Y-%m-%d")
        
        actual_record = db.query(StockHistory).filter(
            StockHistory.symbol == pred.symbol,
            StockHistory.date == pred_date_str
        ).first()
        
        if actual_record:
            pred.actual_price = actual_record.close_price
            
            # Determine if prediction was correct
            # Get previous day's close
            prev_date = pred.prediction_date - timedelta(days=1)
            prev_date_str = prev_date.strftime("%Y-%m-%d")
            
            prev_record = db.query(StockHistory).filter(
                StockHistory.symbol == pred.symbol,
                StockHistory.date == prev_date_str
            ).first()
            
            if prev_record:
                actual_direction = "UP" if actual_record.close_price > prev_record.close_price else "DOWN"
                pred.was_correct = (pred.predicted_direction == actual_direction)
                evaluated += 1
    
    db.commit()
    
    # Calculate accuracy
    total_evaluated = db.query(MLPrediction).filter(
        MLPrediction.was_correct.isnot(None)
    ).count()
    
    correct_predictions = db.query(MLPrediction).filter(
        MLPrediction.was_correct == True
    ).count()
    
    accuracy = (correct_predictions / total_evaluated * 100) if total_evaluated > 0 else 0
    
    return {
        "evaluated_this_run": evaluated,
        "total_evaluated": total_evaluated,
        "correct_predictions": correct_predictions,
        "accuracy_percentage": round(accuracy, 2)
    }