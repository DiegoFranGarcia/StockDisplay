from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
import dotenv

dotenv.load_dotenv()

# Database configuration

DATABASE_URL = os.getenv("DATABASE_URL")

# Create engine
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# Models
class StockData(Base):
    """Store current stock data snapshots"""
    __tablename__ = "stock_data"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True, nullable=False)
    price = Column(Float, nullable=False)
    change = Column(Float, nullable=False)
    change_percent = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    def to_dict(self):
        return {
            "id": self.id,
            "symbol": self.symbol,
            "price": self.price,
            "change": self.change,
            "change_percent": self.change_percent,
            "volume": self.volume,
            "timestamp": self.timestamp.isoformat()
        }


class StockHistory(Base):
    """Store historical price data for ML training"""
    __tablename__ = "stock_history"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True, nullable=False)
    date = Column(String, index=True, nullable=False)
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    tomorrow = Column(Float)  # Next day's close price
    target = Column(Integer)  # 1 if tomorrow > today, 0 otherwise
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            "id": self.id,
            "symbol": self.symbol,
            "date": self.date,
            "open": self.open_price,
            "high": self.high_price,
            "low": self.low_price,
            "close": self.close_price,
            "volume": self.volume,
            "tomorrow": self.tomorrow,
            "target": self.target,
            "timestamp": self.timestamp.isoformat()
        }


class MLPrediction(Base):
    """Store ML model predictions"""
    __tablename__ = "ml_predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True, nullable=False)
    prediction_date = Column(DateTime, nullable=False)
    predicted_direction = Column(String, nullable=False)  # "UP" or "DOWN"
    confidence = Column(Float, nullable=False)  # 0.0 to 1.0
    predicted_price = Column(Float)
    actual_price = Column(Float)
    was_correct = Column(Boolean)
    model_version = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            "id": self.id,
            "symbol": self.symbol,
            "prediction_date": self.prediction_date.isoformat(),
            "predicted_direction": self.predicted_direction,
            "confidence": self.confidence,
            "predicted_price": self.predicted_price,
            "actual_price": self.actual_price,
            "was_correct": self.was_correct,
            "model_version": self.model_version,
            "created_at": self.created_at.isoformat()
        }


# Database helper functions
def get_db():
    """Dependency for getting database sessions"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created successfully")


def drop_all_tables():
    """Drop all tables (use with caution!)"""
    Base.metadata.drop_all(bind=engine)
    print("🗑️  All tables dropped")