#!/usr/bin/env python3
"""
Debug with verbose output
"""

import pandas as pd
from sqlalchemy import text
from app_imports import getDbConnection
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Get one stock
with getDbConnection() as con:
    q = text("""
        SELECT date AS Date, open AS Open, high AS High,
               low AS Low, close AS Close, volume AS Volume
        FROM finnhub_stock_prices
        WHERE symbol = 'AAPL'
        ORDER BY date ASC
    """)
    df = pd.read_sql(q, con=con)

print(f"Testing AAPL with {len(df)} days of data")
print(f"Current price: ${df['Close'].iloc[-1]:.2f}\n")

# Normalize columns
df.columns = [col.title() for col in df.columns]

# Simple feature engineering
from ML_Predict_Price import create_features, FEATURE_HORIZONS_LONG

data = create_features(df, FEATURE_HORIZONS_LONG)
data = data.dropna()

print(f"After feature engineering: {len(data)} rows\n")

# Test 10-day prediction
horizon_days = 10
data['Target'] = data['Close'].shift(-horizon_days)
data = data[:-horizon_days].dropna()

print(f"After creating {horizon_days}-day target: {len(data)} rows\n")

# Prepare features
base_predictors = ['Open', 'High', 'Low', 'Close', 'Volume',
                  'Daily_Return', 'High_Low_Ratio', 'Close_Open_Ratio']
generated_features = [col for col in data.columns 
                     if any(col.startswith(prefix) for prefix in 
                           ['Close_Ratio_', 'Price_Range_', 'Volume_Ratio_', 'Returns_'])]
predictors = base_predictors + generated_features
predictors = [p for p in predictors if p in data.columns]

print(f"Using {len(predictors)} features")

X = data[predictors]
y = data['Target']

# Train/test split
test_size = max(5, len(data) // 5)
train_size = len(data) - test_size

X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}\n")

# Scale and train
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(
    n_estimators=100,
    min_samples_split=20,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

# Evaluate
test_score = model.score(X_test_scaled, y_test)
print(f"Model R² score on test set: {test_score:.4f}")

# Make prediction
latest_features = X.iloc[-1:].values
latest_features_scaled = scaler.transform(latest_features)
predicted_price = model.predict(latest_features_scaled)[0]

current_price = df['Close'].iloc[-1]
change = predicted_price - current_price
change_pct = (change / current_price) * 100

print(f"\nPrediction:")
print(f"  Current price: ${current_price:.2f}")
print(f"  Predicted in {horizon_days} days: ${predicted_price:.2f}")
print(f"  Expected change: {change_pct:+.2f}%")

# Check sanity limits
max_change_per_day = 0.05 / 7
max_change = max_change_per_day * horizon_days
max_change = min(max_change, 0.30)
actual_change = abs(predicted_price - current_price) / current_price

print(f"\nSanity check:")
print(f"  Max allowed change: {max_change*100:.2f}%")
print(f"  Actual change: {actual_change*100:.2f}%")
print(f"  Pass: {actual_change <= max_change}")
