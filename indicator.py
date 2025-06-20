#!/usr/bin/env python

import numpy as np
from eval import loadPrices

# loading prices into matrix
pricesFile="prices.txt"
prcAll = loadPrices(pricesFile)

inst1 = prcAll[:, 0]
print(inst1)

### INDICATOR FORMAT ###
# Will take in price as a vector, and output indicator values as vectors
# Need separate code to intake prices and pass into the indicators

# ------------------------------------------------------------------------------------------------------ #

### Trend Indicators ###
## These help identify the direction and strength of a market trend.

# Moving Averages (MA)

# Simple Moving Average (SMA)

# Exponential Moving Average (EMA)

# Moving Average Convergence Divergence (MACD)

# Average Directional Index (ADX)

# Parabolic SAR (Stop and Reverse)

# Ichimoku Cloud

# ------------------------------------------------------------------------------------------------------ #

### Momentum Indicators ###
## These show the speed of price movements and are useful for identifying overbought or oversold conditions.

# Relative Strength Index (RSI)

# Stochastic Oscillator

# Rate of Change (ROC)

# Commodity Channel Index (CCI)

# Williams %R

# ------------------------------------------------------------------------------------------------------ #

### Volatility Indicators ###
## These measure how much the price is changing and indicate market uncertainty or risk.

# Bollinger Bands

# Average True Range (ATR) (Close-to-Close Volatility approximates this)
def close_to_close_volatility(close_prices, period=14):
    close_prices = np.asarray(close_prices)
    abs_differences = np.abs(np.diff(close_prices))
    vol = np.full_like(close_prices, fill_value=np.nan)
    for i in range(period, len(close_prices)):
        vol[i] = np.mean(abs_differences[i - period:i])
    return vol

# Donchian Channels

# Keltner Channels

# ------------------------------------------------------------------------------------------------------ #

### Others / Composite Tools ###
## More complex or composite tools that combine several data sources.

# Fibonacci Retracement Levels

# Pivot Points

# Elliott Wave Theory

# Market Profile

# VWAP (Volume Weighted Average Price)

# ------------------------------------------------------------------------------------------------------ #
