# SignalClassifier.py
# This module provides a rule-based and ML-based simulation of the LuxAlgo AI classifier.
# It uses historical daily closing price data to compute technical indicators
# via pandas_ta and assigns a signal integer from -4 to 4 (excluding 0).
# Positive values indicate bullish trends: 1-2 for potential reversals/emerging trends,
# 3-4 for strong established trends. Negative values indicate bearish equivalents.
# The rule-based version uses predefined rules on indicators.
# The ML-based version incorporates machine learning (e.g., clustering and classification)
# to simulate signal classification [[1]][[6]][[10]].

import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Union
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class SignalClassifier:
    """
    A class to simulate LuxAlgo AI classifier signals in rule-based and ML-based ways using technical indicators.
    
    Attributes:
        None (stateless, all computation in methods)
    
    Methods:
        get_rule_signal_int(df: pd.DataFrame) -> int:
            Computes the signal integer using rule-based logic.
        get_ml_signal_int(df: pd.DataFrame) -> int:
            Computes the signal integer using machine learning simulation.
    """
    
    def __init__(self):
        """
        Initializes the SignalClassifier instance.
        No parameters needed as it's stateless.
        """
        pass
    
    def _preprocess_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses the input DataFrame: sorts by date, ensures required columns, handles missing values.
        
        Args:
            df (pd.DataFrame): Input DataFrame with at least 'Date' and 'Close' columns.
        
        Returns:
            pd.DataFrame: Preprocessed DataFrame.
        
        Raises:
            ValueError: If DataFrame lacks required columns or has insufficient data.
        """
        if not all(col in df.columns for col in ['Date', 'Close']):
            raise ValueError("DataFrame must contain 'Date' and 'Close' columns.")
        
        # Convert 'Date' to datetime if not already
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Sort by date ascending
        df = df.sort_values(by='Date').reset_index(drop=True)
        
        # Handle missing values in 'Close' by forward-filling
        df['Close'] = df['Close'].ffill()
        
        # Ensure at least 200 periods for reliable indicators (e.g., SMA200)
        if len(df) < 200:
            raise ValueError("DataFrame must have at least 200 rows for reliable indicator computation.")
        
        return df
    
    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes technical indicators using pandas_ta and adds them to the DataFrame.
        
        Indicators include:
        - SMA50 and SMA200 for trend detection
        - RSI14 for momentum and overbought/oversold conditions
        - MACD (12,26,9) for convergence/divergence
        - Bollinger Bands (20,2) for volatility and potential reversals
        
        Args:
            df (pd.DataFrame): Preprocessed DataFrame with 'Close'.
        
        Returns:
            pd.DataFrame: DataFrame with added indicator columns.
        """
        # SMA for trend
        df['SMA50'] = ta.sma(df['Close'], length=50)
        df['SMA200'] = ta.sma(df['Close'], length=200)
        
        # RSI for momentum
        df['RSI14'] = ta.rsi(df['Close'], length=14)
        
        # MACD
        macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_Signal'] = macd['MACDs_12_26_9']
        df['MACD_Hist'] = macd['MACDh_12_26_9']
        
        # Bollinger Bands
        bb = ta.bbands(df['Close'], length=20, std=2)
        df['BB_Lower'] = bb['BBL_20_2.0']
        df['BB_Upper'] = bb['BBU_20_2.0']
        df['BB_Mid'] = bb['BBM_20_2.0']
        
        # Drop NaN rows introduced by indicators
        df = df.dropna().reset_index(drop=True)
        
        return df
    
    def _determine_trend_direction(self, row: pd.Series) -> int:
        """
        Determines the trend direction based on the latest row's indicators.
        
        - Bullish: +1 if SMA50 > SMA200
        - Bearish: -1 if SMA50 < SMA200
        - Neutral: 0 (though we avoid neutral in final signal)
        
        Args:
            row (pd.Series): Latest row with indicators.
        
        Returns:
            int: +1 (bullish), -1 (bearish), 0 (neutral).
        """
        if row['SMA50'] > row['SMA200']:
            return 1
        elif row['SMA50'] < row['SMA200']:
            return -1
        return 0
    
    def _compute_strength_score(self, row: pd.Series, direction: int) -> int:
        """
        Computes the strength score (1-4) based on multiple indicators.
        
        Scoring:
        - RSI: Oversold/Overbought for reversal (weak), extreme for strong
        - MACD: Crossover and histogram for momentum strength
        - Bollinger: Position relative to bands for volatility signals
        
        Aggregates votes to determine 1-2 (emerging/reversal) or 3-4 (established/strong).
        
        Args:
            row (pd.Series): Latest row with indicators.
            direction (int): Trend direction (+1 or -1).
        
        Returns:
            int: Strength level (1-4).
        """
        score = 0
        
        # RSI contribution
        if direction > 0:  # Bullish
            if row['RSI14'] < 30: score += 1  # Oversold, potential reversal (weak)
            elif 30 <= row['RSI14'] < 50: score += 2  # Emerging uptrend
            elif 50 <= row['RSI14'] < 70: score += 3  # Established
            elif row['RSI14'] >= 70: score += 4  # Strong momentum
        else:  # Bearish
            if row['RSI14'] > 70: score += 1  # Overbought, potential reversal (weak)
            elif 50 < row['RSI14'] <= 70: score += 2  # Emerging downtrend
            elif 30 < row['RSI14'] <= 50: score += 3  # Established
            elif row['RSI14'] <= 30: score += 4  # Strong momentum
        
        # MACD contribution
        if (direction > 0 and row['MACD'] > row['MACD_Signal'] and row['MACD_Hist'] > 0) or \
           (direction < 0 and row['MACD'] < row['MACD_Signal'] and row['MACD_Hist'] < 0):
            score += 3  # Strong confirmation
        elif (direction > 0 and row['MACD'] > row['MACD_Signal']) or \
             (direction < 0 and row['MACD'] < row['MACD_Signal']):
            score += 2  # Moderate
        else:
            score += 1  # Weak or diverging
        
        # Bollinger Bands contribution
        if direction > 0:  # Bullish
            if row['Close'] < row['BB_Lower']: score += 1  # Potential reversal (weak)
            elif row['BB_Lower'] <= row['Close'] < row['BB_Mid']: score += 2
            elif row['BB_Mid'] <= row['Close'] < row['BB_Upper']: score += 3
            elif row['Close'] >= row['BB_Upper']: score += 4  # Breakout strong
        else:  # Bearish
            if row['Close'] > row['BB_Upper']: score += 1  # Potential reversal (weak)
            elif row['BB_Mid'] < row['Close'] <= row['BB_Upper']: score += 2
            elif row['BB_Lower'] < row['Close'] <= row['BB_Mid']: score += 3
            elif row['Close'] <= row['BB_Lower']: score += 4  # Breakdown strong
        
        # Normalize score to 1-4
        max_possible = 4 + 3 + 4  # 11
        normalized = max(1, min(4, int(np.ceil((score / max_possible) * 4))))
        
        return normalized
    
    def get_rule_signal_int(self, df: pd.DataFrame) -> int:
        """
        Main function to compute the rule-based signal integer.
        
        Processes the DataFrame, computes indicators, determines direction and strength
        based on the latest available data point.
        
        Args:
            df (pd.DataFrame): DataFrame with 'Date' and 'Close' columns.
        
        Returns:
            int: Signal from -4 to -1 (bearish) or 1 to 4 (bullish).
        
        Raises:
            ValueError: For invalid input DataFrame.
        """
        df = self._preprocess_df(df)
        df = self._compute_indicators(df)
        
        # Get the latest row
        latest_row = df.iloc[-1]
        
        # Determine direction
        direction = self._determine_trend_direction(latest_row)
        if direction == 0:
            # Fallback: Use MACD for direction if MA is neutral
            direction = 1 if latest_row['MACD'] > 0 else -1
        
        # Compute strength
        strength = self._compute_strength_score(latest_row, direction)
        
        # Combine
        signal = direction * strength
        
        return signal

    def _generate_pseudo_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Generates pseudo-labels for ML training based on future returns.
        Labels are integers from -4 to 4 (excluding 0) simulating signal strength and direction.
        - Positive if future return >0, negative if <0.
        - Magnitude based on return percentile (1-4).
        
        Args:
            df (pd.DataFrame): DataFrame with 'Close'.
        
        Returns:
            pd.Series: Pseudo-labels for each row (shifted to align with features).
        """
        # Compute forward 1-day returns
        df['Return'] = df['Close'].pct_change().shift(-1)
        
        # Drop last row (no future return)
        df = df.iloc[:-1]
        
        # Direction: +1 if return >0, -1 if <0, skip 0
        direction = np.sign(df['Return'])
        direction = direction.replace(0, np.nan).ffill()  # Rare zeros, fill
        
        # Strength: Quantile-based (1-4) on absolute returns
        abs_returns = df['Return'].abs()
        quantiles = abs_returns.quantile([0.25, 0.5, 0.75])
        strength = pd.cut(abs_returns, bins=[0, quantiles[0.25], quantiles[0.5], quantiles[0.75], np.inf], labels=[1,2,3,4], include_lowest=True)
        strength = strength.astype(int)
        
        # Combine
        labels = (direction * strength).astype(int)
        
        return labels
    
    def get_ml_signal_int(self, df: pd.DataFrame) -> int:
        """
        Computes the signal integer using machine learning to simulate LuxAlgo AI classifier.
        Incorporates feature engineering, pseudo-label generation from returns, 
        KMeans clustering for initial grouping [[6]], and RandomForestClassifier for prediction [[1]][[3]][[10]].
        Trains on historical data and predicts for the latest point.
        
        Args:
            df (pd.DataFrame): DataFrame with 'Date' and 'Close' columns.
        
        Returns:
            int: Signal from -4 to -1 (bearish) or 1 to 4 (bullish).
        
        Raises:
            ValueError: For invalid input DataFrame or insufficient data for ML.
        """
        df = self._preprocess_df(df)
        df = self._compute_indicators(df)
        
        if len(df) < 250:  # Need more data for ML training
            raise ValueError("DataFrame must have at least 250 rows for ML computation.")
        
        # Features: Select relevant columns
        features = ['SMA50', 'SMA200', 'RSI14', 'MACD', 'MACD_Signal', 'MACD_Hist', 'BB_Lower', 'BB_Upper', 'BB_Mid']
        X = df[features].copy()
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply KMeans clustering to group similar indicator states (simulating ML clustering in LuxAlgo) [[6]]
        kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)  # 8 clusters for 8 possible signals (-4 to 4 excl 0)
        df['Cluster'] = kmeans.fit_predict(X_scaled)
        
        # Generate pseudo-labels based on future returns for supervised learning [[3]][[9]]
        labels = self._generate_pseudo_labels(df.copy())
        
        # Align: Labels for rows 0 to n-2, features 0 to n-2 for training, predict on n-1
        X_train = X_scaled[:-1]
        # Use to_numpy for compatibility and clearer intent
        y_train = labels.to_numpy()
        
        # Train-test split for validation (though we use full for final model)
        X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        
        # Train RandomForestClassifier [[3]]
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_split, y_train_split)
        
        # Optional: Check accuracy
        accuracy = accuracy_score(y_test_split, model.predict(X_test_split))
        # print(f"Model accuracy: {accuracy:.2f}")  # For debugging
        
        # Retrain on full historical data
        model.fit(X_train, y_train)
        
        # Predict on latest (unseen) row
        latest_features = X_scaled[-1].reshape(1, -1)
        predicted_signal = model.predict(latest_features)[0]
        
        # Ensure non-zero
        if predicted_signal == 0:
            predicted_signal = 1 if np.random.rand() > 0.5 else -1  # Rare fallback
        
        return predicted_signal
