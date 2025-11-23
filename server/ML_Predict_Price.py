#!/usr/bin/env python3
"""
ML_Predict_Price.py

Machine Learning multi-horizon price prediction module for swing trading.
Uses Random Forest Regression to predict closing prices across multiple timeframes
and returns the most confident prediction.

Designed for production swing trading:
- Tests multiple prediction horizons (5, 10, 15, 20 days)
- Returns only the most confident prediction with its timeframe
- No external API dependencies (no yfinance)
- Returns None if data is insufficient or confidence is low

Usage:
    from ML_Predict_Price import predict_best_horizon
    
    # df must have columns: Open, High, Low, Close, Volume (case-insensitive)
    result = predict_best_horizon(df)
    # Returns dict: {'price': float, 'days': int, 'confidence': float, 'return_pct': float, 'current_price': float} or None
"""

import pandas as pd
import numpy as np
import warnings
from typing import Optional, Tuple, List

# Suppress warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
except ImportError:
    RandomForestRegressor = None
    StandardScaler = None


# Minimum data requirements for different training strategies
MIN_DATA_POINTS = 60  # Absolute minimum for multi-horizon prediction
OPTIMAL_DATA_POINTS = 120  # Optimal for good predictions
FEATURE_HORIZONS_SHORT = [2, 5, 10]  # For limited data (60-119 days)
FEATURE_HORIZONS_LONG = [2, 5, 10, 20, 60]  # For optimal data (120+ days)

# Prediction horizons for swing trading (in days)
PREDICTION_HORIZONS = [5, 10, 15, 20]  # 1 week, 2 weeks, 3 weeks, 4 weeks
MIN_CONFIDENCE_THRESHOLD = 0.40  # Minimum R² score (balanced for practical use)


def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate that the dataframe has required OHLCV columns and sufficient data.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if RandomForestRegressor is None or StandardScaler is None:
        return False, "scikit-learn not installed"
    
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    if df is None or df.empty:
        return False, "Empty dataframe provided"
    
    # Check for required columns (case-insensitive)
    df_cols_lower = [col.lower() for col in df.columns]
    missing_cols = [col for col in required_cols if col.lower() not in df_cols_lower]
    
    if missing_cols:
        return False, f"Missing required columns: {missing_cols}"
    
    # Check minimum data points
    if len(df) < MIN_DATA_POINTS:
        return False, f"Insufficient data: {len(df)} rows, minimum {MIN_DATA_POINTS} required"
    
    # Check for null values in recent data (last 20%)
    recent_rows = max(5, len(df) // 5)
    
    # Normalize column names for checking
    df_normalized = df.copy()
    df_normalized.columns = [col.title() for col in df_normalized.columns]
    recent_data = df_normalized[required_cols].tail(recent_rows)
    
    if recent_data.isnull().any().any():
        return False, "Null values detected in recent data"
    
    # Check for zero or negative prices in recent data
    price_cols = ['Open', 'High', 'Low', 'Close']
    if (recent_data[price_cols] <= 0).any().any():
        return False, "Invalid price values (<=0) detected in recent data"
    
    return True, ""


def create_features(df: pd.DataFrame, horizons: list) -> pd.DataFrame:
    """
    Create technical features based on rolling windows.
    
    Features created:
    - Close_Ratio_N: Current close / N-day average close
    - Price_Range_N: (High - Low) / Close averaged over N days
    - Volume_Ratio_N: Current volume / N-day average volume
    - Returns_N: N-day percentage return
    """
    data = df.copy()
    
    for horizon in horizons:
        if horizon >= len(data):
            continue
            
        # Price ratios
        rolling_close = data['Close'].rolling(window=horizon, min_periods=1).mean()
        data[f'Close_Ratio_{horizon}'] = data['Close'] / rolling_close
        
        # Volatility measure
        price_range = (data['High'] - data['Low']) / data['Close']
        data[f'Price_Range_{horizon}'] = price_range.rolling(window=horizon, min_periods=1).mean()
        
        # Volume ratios
        rolling_volume = data['Volume'].rolling(window=horizon, min_periods=1).mean()
        # Avoid division by zero
        rolling_volume = rolling_volume.replace(0, 1)
        data[f'Volume_Ratio_{horizon}'] = data['Volume'] / rolling_volume
        
        # Returns
        data[f'Returns_{horizon}'] = data['Close'].pct_change(periods=horizon)
    
    # Add basic features
    data['Daily_Return'] = data['Close'].pct_change()
    data['High_Low_Ratio'] = data['High'] / data['Low']
    data['Close_Open_Ratio'] = data['Close'] / data['Open']
    
    return data


def predict_price_at_horizon(df: pd.DataFrame, horizon_days: int, 
                            confidence_threshold: float = 0.65) -> Optional[Tuple[float, float]]:
    """
    Predict closing price at a specific future horizon using Random Forest Regression.
    
    Args:
        df: DataFrame with OHLCV data (must have columns: Open, High, Low, Close, Volume)
        horizon_days: Number of days ahead to predict (e.g., 5, 10, 15, 20)
        confidence_threshold: Minimum R² score required for returning prediction
    
    Returns:
        Tuple[float, float]: (predicted_price, confidence_score) or None if:
            - Insufficient data
            - Invalid data format
            - Model confidence is too low
            - Any errors during prediction
    """
    try:
        # Validate input
        is_valid, error_msg = validate_dataframe(df)
        if not is_valid:
            print(f"Validation failed: {error_msg}")
            return None
        
        # Normalize column names to title case
        df = df.copy()
        df.columns = [col.title() for col in df.columns]
        
        # Determine feature horizons based on data length
        data_length = len(df)
        if data_length < OPTIMAL_DATA_POINTS:
            horizons = FEATURE_HORIZONS_SHORT
            n_estimators = 50
            min_samples_split = 10
        else:
            horizons = FEATURE_HORIZONS_LONG
            n_estimators = 100
            min_samples_split = 20
        
        # Create features
        data = create_features(df, horizons)
        
        # Remove rows with NaN values (from rolling calculations)
        data = data.dropna()
        
        # Check if we still have enough data after feature creation
        if len(data) < MIN_DATA_POINTS + horizon_days:
            print(f"Insufficient data after feature engineering: {len(data)} rows")
            return None
        
        # Create target variable (close price N days ahead)
        data['Target'] = data['Close'].shift(-horizon_days)
        
        # Remove rows without targets and any remaining NaN
        data = data[:-horizon_days].dropna()
        
        if len(data) < MIN_DATA_POINTS:
            print(f"Insufficient data after target creation for {horizon_days}-day horizon")
            return None
        
        # Prepare features for training
        base_predictors = ['Open', 'High', 'Low', 'Close', 'Volume',
                          'Daily_Return', 'High_Low_Ratio', 'Close_Open_Ratio']
        
        # Add generated features
        generated_features = [col for col in data.columns 
                             if any(col.startswith(prefix) for prefix in 
                                   ['Close_Ratio_', 'Price_Range_', 'Volume_Ratio_', 'Returns_'])]
        
        predictors = base_predictors + generated_features
        
        # Ensure all predictors exist in data
        predictors = [p for p in predictors if p in data.columns]
        
        X = data[predictors]
        y = data['Target']
        
        # Split into train and test (use last 20% for validation, min 5 samples)
        test_size = max(5, len(data) // 5)
        train_size = len(data) - test_size
        
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest Regressor with improved parameters
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            max_depth=15,  # Increased from 10
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model on test set
        test_score = model.score(X_test_scaled, y_test)
        
        # Check if model meets confidence threshold
        if test_score < confidence_threshold:
            return None
        
        # Prepare the most recent data point for prediction
        latest_features = X.iloc[-1:].values
        latest_features_scaled = scaler.transform(latest_features)
        
        # Make prediction
        predicted_price = model.predict(latest_features_scaled)[0]
        
        # Sanity check: predicted price should be within reasonable bounds for swing trading
        current_close = df['Close'].iloc[-1]
        # Allow 10% per week for swing trading (stocks can move significantly)
        max_change_per_week = 0.10
        weeks = horizon_days / 7.0
        max_change = max_change_per_week * weeks
        max_change = min(max_change, 0.50)  # Cap at 50% total for longer horizons
        
        if abs(predicted_price - current_close) / current_close > max_change:
            return None
        
        # Ensure predicted price is positive
        if predicted_price <= 0:
            return None
        
        return (float(predicted_price), float(test_score))
    
    except Exception as e:
        return None


def predict_best_horizon(df: pd.DataFrame, 
                        horizons: List[int] = None,
                        min_confidence: float = MIN_CONFIDENCE_THRESHOLD) -> Optional[dict]:
    """
    Predict closing price across multiple horizons and return the most confident prediction.
    
    This is the main function for production use. It tests multiple prediction timeframes
    and returns only the best one based on model confidence.
    
    Args:
        df: DataFrame with OHLCV data (must have columns: Open, High, Low, Close, Volume)
        horizons: List of horizons to test in days (default: [5, 10, 15, 20])
        min_confidence: Minimum R² score to accept any prediction (default: 0.65)
    
    Returns:
        dict: {
            'price': float,        # Predicted closing price
            'days': int,           # Number of days ahead for this prediction
            'confidence': float,   # R² score (0-1, higher is better)
            'return_pct': float,   # Expected return percentage ((target - current) / current * 100)
            'current_price': float # Current closing price for reference
        }
        or None if no horizon meets confidence threshold
    """
    if horizons is None:
        horizons = PREDICTION_HORIZONS
    
    # Validate input
    is_valid, error_msg = validate_dataframe(df)
    if not is_valid:
        return None
    
    # Normalize column names
    df = df.copy()
    df.columns = [col.title() for col in df.columns]
    
    current_price = df['Close'].iloc[-1]
    
    # Test each horizon and collect results
    results = []
    for horizon in horizons:
        prediction = predict_price_at_horizon(df, horizon, min_confidence)
        if prediction is not None:
            predicted_price, confidence = prediction
            return_pct = ((predicted_price - current_price) / current_price) * 100.0
            results.append({
                'price': predicted_price,
                'days': horizon,
                'confidence': confidence,
                'return_pct': return_pct,
                'current_price': current_price
            })
    
    # Return the prediction with highest confidence
    if not results:
        return None
    
    best_prediction = max(results, key=lambda x: x['confidence'])
    return best_prediction


# Legacy function for backward compatibility
def predict_next_close(df: pd.DataFrame, confidence_threshold: float = 0.65) -> Optional[float]:
    """
    Legacy function: Predict next day's closing price.
    For production use, prefer predict_best_horizon() instead.
    
    Returns:
        float: Predicted closing price or None
    """
    result = predict_best_horizon(df, horizons=[1], min_confidence=confidence_threshold)
    return result['price'] if result else None


if __name__ == "__main__":
    # Example usage
    print("ML_Predict_Price.py - Multi-Horizon Price Prediction")
    print("=" * 70)
    
    # Create sample OHLCV data for testing
    dates = pd.date_range(end=pd.Timestamp.now(), periods=150, freq='D')
    np.random.seed(42)
    
    # Generate realistic price movement with random walk
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, 150)  # 0.1% daily return, 2% volatility
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Create OHLCV data
    opens = prices * np.random.uniform(0.99, 1.01, 150)
    highs = np.maximum(opens, prices) * np.random.uniform(1.0, 1.02, 150)
    lows = np.minimum(opens, prices) * np.random.uniform(0.98, 1.0, 150)
    closes = prices
    volumes = np.random.randint(1000000, 10000000, 150)
    
    sample_df = pd.DataFrame({
        'Date': dates,
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Volume': volumes
    })
    
    print(f"\nSample data shape: {sample_df.shape}")
    print(f"Date range: {sample_df['Date'].min().date()} to {sample_df['Date'].max().date()}")
    print(f"Current close: ${sample_df['Close'].iloc[-1]:.2f}")
    
    # Test multi-horizon prediction
    print("\n" + "-" * 70)
    print("Testing multi-horizon prediction...")
    print("-" * 70)
    
    result = predict_best_horizon(sample_df, min_confidence=0.5)
    
    if result is not None:
        current = result['current_price']
        predicted = result['price']
        days = result['days']
        confidence = result['confidence']
        change = predicted - current
        change_pct = (change / current) * 100
        
        print(f"✓ Best Prediction Found!")
        print(f"  Current price: ${current:.2f}")
        print(f"  Predicted price: ${predicted:.2f} (in {days} days)")
        print(f"  Expected change: ${change:+.2f} ({change_pct:+.2f}%)")
        print(f"  Confidence (R²): {confidence:.3f}")
        print(f"  Target date: ~{(pd.Timestamp.now() + pd.Timedelta(days=days)).date()}")
    else:
        print("✗ No confident prediction available")
        print("  (All horizons below confidence threshold)")

 