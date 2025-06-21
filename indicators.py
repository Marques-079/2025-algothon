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
def ma(close, period=14, method='sma'):
    """
    Returns a moving average (SMA or EMA) over the closing prices.

    Parameters:
        close  : np.ndarray of close prices
        period : lookback window
        method : 'sma' or 'ema'

    Returns:
        ma     : np.ndarray of same length, with np.nan where not defined
    """
    close = np.asarray(close, dtype=float)
    ma = np.full_like(close, np.nan)

    if method == 'sma':
        for i in range(period - 1, len(close)):
            ma[i] = np.mean(close[i - period + 1:i + 1])

    elif method == 'ema':
        alpha = 2 / (period + 1)
        for i in range(len(close)):
            if i == period - 1:
                ma[i] = np.mean(close[:period])  # init with SMA
            elif i >= period:
                ma[i] = alpha * close[i] + (1 - alpha) * ma[i - 1]
    
    else:
        raise ValueError("method must be 'sma' or 'ema'")

    return ma

# Exponential Moving Average (EMA)

# Moving Average Convergence Divergence (MACD)

# Average Directional Index (ADX)
def adx_from_closes(close, period=14, smooth_n=None):
    close = np.asarray(close, dtype=float)
    delta = np.diff(close)

    plus_dm = np.where(delta > 0, delta, 0)
    minus_dm = np.where(delta < 0, -delta, 0)

    plus_di = np.full(len(close), np.nan)
    minus_di = np.full(len(close), np.nan)

    for i in range(period, len(delta)):
        plus_di[i] = np.mean(plus_dm[i - period:i])
        minus_di[i] = np.mean(minus_dm[i - period:i])

    # Compute pseudo ADX (just normalized DI difference)
    di_diff = np.abs(plus_di - minus_di)
    di_sum = plus_di + minus_di
    dx = 100 * (di_diff / di_sum)

    adx = np.full(len(close), np.nan)
    for i in range(period * 2, len(dx)):
        adx[i] = np.mean(dx[i - period:i])

    # --- Optional secondary smoothing ---
    if smooth_n is not None and smooth_n > 1:
        smoothed_adx = np.full_like(adx, np.nan)
        for i in range(smooth_n, len(adx)):
            smoothed_adx[i] = np.mean(adx[i - smooth_n:i])
        return smoothed_adx
    else:
        return adx

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

### Classifying Market Condition Indicators & Signals ###

def linear_reg(close_prices, look_back):
    x = np.arange(look_back)
    if len(close_prices) < look_back:
        return None
    close_prices = close_prices[-look_back:]
    # Linear regression to get trend slope
    slope, _ = np.polyfit(x, close_prices, deg=1)
    slope_pct = slope / np.mean(close_prices)

    return slope_pct

def price_ma_diff(close_prices, smooth):
    sma = ma(close_prices, smooth)
    return close_prices[-1] - sma[-1]

def ma_ma_diff(close_prices, fast, slow):
    slow_ma = ma(close_prices, slow)
    fast_ma = ma(close_prices, fast)
    return fast_ma - slow_ma

def market_condition(prcAll):
    # can use

    # linear reg

    # rsi

    # over ma (50!) price_ma_diff

    # ma over ma, 50 & 200 ma trend (or 25, 100 more reactive)

    # 

    return 0

def market_condition_test(prices: np.ndarray) -> str:
    """
    Classify market condition for a single instrument given a 1D array of closing prices.
    Returns 'bullish', 'bearish', or 'stagnant'.
    """
    prices = prices[~np.isnan(prices)]  # Remove NaNs


    tl = 100
    recent_prices = prices[-tl:]
    if len(prices) < tl:
        return "unknown"
    slope_pct = linear_reg(prices, tl)

    # Long-term average
    ma_long = np.mean(recent_prices)
    current_price = recent_prices[-1]

    # Classification logic
    # if linear regression pointing up + price above ma then bullish!
    cut_off = 0
    if slope_pct > cut_off and current_price > ma_long:
        return "bullish"
    elif slope_pct < -cut_off and current_price < ma_long:
        return "bearish"
    else:
        return "stagnant"

### IMPORTING AND CHECKING ###
def loadPrices(fn):
    global nt, nInst
    df=pd.read_csv(fn, sep=r'\s+', header=None, index_col=None)
    (nt,nInst) = df.shape
    return (df.values).T


# Master graphing function #
def show_graph():

    # loading prices into matrix
    pricesFile="prices.txt"
    prcAll = loadPrices(pricesFile)
    inst = 40

    # draw_ma(prcAll, inst, 50, 200)
    draw_market_condition(prcAll, inst)

# Graphing functions #

def draw_market_condition(prcAll, inst):
    # Evaluate regime at each time step
    closes = prcAll[inst, :]
    T = len(closes)
    regimes = []
    for t in range(T):
        regime = market_condition(closes[:t+1])
        regimes.append(regime)

    # Map regimes to colors
    regime_colors = {'bullish': 'green', 'bearish': 'red', 'stagnant': 'gray', 'unknown': 'white'}

    # Plotting
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(closes, label='Closing Price', color='black')

    for t in range(100, T):
        color = regime_colors.get(regimes[t], 'white')
        ax.axvspan(t - 1, t, color=color, alpha=0.2)

    ax.set_title("Market Regime Classification")
    ax.set_xlabel("Time (Days)")
    ax.set_ylabel("Price")
    ax.legend()
    plt.tight_layout()
    plt.show()

def draw_ma(prcAll, inst, slow, fast):
    closes = prcAll[inst, :]
    period = 14
    # --- Compute moving averages ---
    ma_50 = ma(closes, period=slow, method='sma')
    ma_200 = ma(closes, period=fast, method='sma')

    # --- Plotting ---
    plt.figure(figsize=(12, 6))
    plt.plot(closes, label='Close Price', color='black')
    plt.plot(ma_50, label=f'{fast}-Day MA', color='blue')
    plt.plot(ma_200, label=f'{slow}-Day MA', color='red')
    plt.title(f"Close Price with {fast} and {slow}-Day Moving Averages")
    plt.xlabel("Day")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

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

def draw_adx(prcAll, inst):
    # --- Calculate ADX ---
    closes = prcAll[inst, :]
    period = 14
    smoothing = 14
    adx = adx_from_closes(closes, period, smoothing)

    # --- Plot ---
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Subplot 1: Close price
    axs[0].plot(closes, label='Close Price', color='black')
    axs[0].set_title("Close Price")
    axs[0].legend()
    axs[0].grid(True)

    # Subplot 2: Pseudo-ADX
    axs[1].plot(adx, label='Pseudo ADX (close-only)', color='blue')
    axs[1].axhline(25, color='gray', linestyle='--', linewidth=0.5, label='Trend Threshold (25)')
    axs[1].set_title("Pseudo ADX from Closing Prices Only")
    axs[1].set_xlabel("Day")
    axs[1].set_ylabel("Value")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

show_graph()