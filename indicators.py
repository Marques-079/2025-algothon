#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
def rsi_exponential(close, period=14):
    if close.size <= 14:
        return np.full_like(close, fill_value=np.nan)

    close = np.asarray(close, dtype=float)
    delta = np.diff(close)

    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = np.empty_like(close)
    avg_loss = np.empty_like(close)
    avg_gain[:period] = np.nan
    avg_loss[:period] = np.nan

    # First average: simple average
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])

    # Subsequent: exponential smoothing (Wilder's method)
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i - 1]) / period

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = np.nan  # RSI is undefined for the first `period` points

    return rsi

def rsi_ma(close_prices, period=14):
    close_prices = np.asarray(close_prices, dtype=float)
    delta = np.diff(close_prices)  # price changes

    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    rsi = np.full(len(close_prices), np.nan)  # final RSI output (same length as close)

    for i in range(period, len(close_prices) - 1):
        avg_gain = np.mean(gain[i - period:i])
        avg_loss = np.mean(loss[i - period:i])

        if avg_loss == 0:
            rsi[i + 1] = 100  # prevent division by zero — means only gains
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100 - (100 / (1 + rs))

    return rsi

# Stochastic Oscillator

# Rate of Change (ROC)

# Commodity Channel Index (CCI)

# Williams %R

# ------------------------------------------------------------------------------------------------------ #

### Volatility Indicators ###
## These measure how much the price is changing and indicate market uncertainty or risk.

# Bollinger Bands

# Average True Range (ATR) (Close-to-Close Volatility approximates this)
def atr_close_to_close(close_prices, period=14):
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

### IMPORTING AND CHECKING ###
def loadPrices(fn):
    global nt, nInst
    df=pd.read_csv(fn, sep=r'\s+', header=None, index_col=None)
    (nt,nInst) = df.shape
    return (df.values).T

def show_graph():

    # loading prices into matrix
    pricesFile="prices.txt"
    prcAll = loadPrices(pricesFile)

    draw_rsi(prcAll)

def draw_rsi(prcAll, inst):
    # inst_i = prcAll[instrument_i, NumDays]
    closes = prcAll[inst, :]
    period = 14
    rsi_wilder = rsi_exponential(closes)
    rsi_simple = rsi_ma(closes)
    rsi_diff = rsi_wilder - rsi_simple


    # --- Plotting with subplots ---
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # --- Subplot 1: Price ---
    axs[0].plot(closes, color='black', label='Close Price')
    axs[0].set_title("Closing Price")
    axs[0].grid(True)
    axs[0].legend()

    # --- Subplot 2: RSI Comparison ---
    axs[1].plot(rsi_wilder, label='Wilder RSI', color='blue')
    axs[1].plot(rsi_simple, label='Simple RSI', color='orange', linestyle='--')
    axs[1].axhline(70, color='gray', linestyle='--', linewidth=0.5)
    axs[1].axhline(30, color='gray', linestyle='--', linewidth=0.5)
    axs[1].set_title(f"RSI Comparison (Period = {period})")
    axs[1].set_ylabel("RSI")
    axs[1].grid(True)
    axs[1].legend()

    # --- Subplot 3: RSI Difference ---
    axs[2].plot(rsi_diff, label='Wilder RSI - Simple RSI', color='green')
    axs[2].axhline(0, color='gray', linestyle='--', linewidth=0.5)
    axs[2].set_title("RSI Difference")
    axs[2].set_xlabel("Day")
    axs[2].set_ylabel("Difference")
    axs[2].grid(True)
    axs[2].legend()

    plt.tight_layout()
    plt.show()

show_graph()