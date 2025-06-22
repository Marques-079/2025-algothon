#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.stats import percentileofscore

### INDICATOR FORMAT ###
# Will take in price as a vector, and output indicator values as vectors
# Need separate code to intake prices and pass into the indicators

# ------------------------------------------------------------------------------------------------------ #

### Trend Indicators ###
## These help identify the direction and strength of a market trend.

# Moving Averages (MA)

# Simple Moving Average (SMA)
# Exponential Moving Average (EMA)

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

# Elliott Wave Theory

# Market Profile

# VWAP (Volume Weighted Average Price)'

# Pivots Points
def get_latest_pivot_high(prices: np.ndarray, left: int = 5, right: int = 5):
    """
    Finds the most recent pivot high in the price series.

    Parameters:
    - prices (np.ndarray): 1D array of prices
    - left (int): number of candles before the pivot that must be lower
    - right (int): number of candles after the pivot that must be lower

    Returns:
    - (pivot_index, pivot_price) or (None, None) if no pivot is found
    """
    n = len(prices)
    for i in range(n - right - 1, left - 1, -1):
        is_pivot = True
        for j in range(1, left + 1):
            if prices[i] <= prices[i - j]:
                is_pivot = False
                break
        if is_pivot:
            for j in range(1, right + 1):
                if prices[i] <= prices[i + j]:
                    is_pivot = False
                    break
        if is_pivot:
            return i, prices[i]
    return None, None

def get_latest_pivot_low(prices: np.ndarray, left: int = 5, right: int = 5):
    """
    Finds the most recent pivot low in the price series.

    Parameters:
    - prices (np.ndarray): 1D array of prices
    - left (int): number of candles before the pivot that must be higher
    - right (int): number of candles after the pivot that must be higher

    Returns:
    - (pivot_index, pivot_price) or (None, None) if no pivot is found
    """
    n = len(prices)
    for i in range(n - right - 1, left - 1, -1):
        is_pivot = True
        for j in range(1, left + 1):
            if prices[i] >= prices[i - j]:
                is_pivot = False
                break
        if is_pivot:
            for j in range(1, right + 1):
                if prices[i] >= prices[i + j]:
                    is_pivot = False
                    break
        if is_pivot:
            return i, prices[i]
    return None, None

# ------------------------------------------------------------------------------------------------------ #

### Classifying Market Condition Indicators & Signals ###

def linear_reg(close_prices, look_back):
    x = np.arange(look_back)
    if len(close_prices) < look_back:
        return np.nan
    close_prices = close_prices[-look_back:]
    # Linear regression to get trend slope
    slope, _ = np.polyfit(x, close_prices, deg=1)
    slope_pct = slope / np.mean(close_prices)

    return slope_pct

def price_ma_diff(close_prices, smooth):
    sma = ma(close_prices, smooth)
    return close_prices[-1] - sma[-1]

def ma_ma_diff(close_prices, fast, slow):
    if len(close_prices) < slow:
        return np.nan
        
    slow_ma = ma(close_prices[-slow:], slow)
    fast_ma = ma(close_prices[-fast:], fast)
    return fast_ma[-1] - slow_ma[-1]

def pivot_breaking(prcAll, look_length, atr_breakpoint):
    prev_high = get_latest_pivot_high(prcAll, look_length, look_length)[-1]
    prev_low = get_latest_pivot_low(prcAll, look_length, look_length)[-1]

    # exit if undefined
    if (prev_high == None) or (prev_low == None) or (len(prcAll) < 1):
        return None
    last_price = prcAll[-1]

    # change check with 1 atr
    curr_atr = atr_close_to_close(prcAll)[-1]

    if last_price > prev_high + atr_breakpoint*curr_atr:
        return 1
    if last_price < prev_low - atr_breakpoint*curr_atr:
        return -1
    return 0

def get_last_privot_position(prcAll, pivot_signals):
    end = len(prcAll)-1

    for i in range(end, -1, -1):
        if pivot_signals[i] != 0:
            return pivot_signals[i]
    return None

    # while True:
    #     if end <= 0:
    #         return None
    #     pos = pivot_breaking(prcAll[:end], look_length)
    #     if pos == None:
    #         return None
    #     if pos == 0:
    #         end -= 1
    #     else:
    #         return pos

def get_len_of_trend(day):
    # returns the length * dir of trend, otherwise None
    if day > 1 and market_signals[day-1] not in [0, np.nan]:
        count = 0
        trend = market_signals[day-1]
        while day > 1 and market_signals[day-1] == trend:
            count += 1
            day -= 1
        return count * trend
    return np.nan

# NEED TO CHANGE!
pivot_signals = np.zeros(1000)
long_pivot_signals = np.zeros(1000)
market_signals = np.zeros(1000)
def market_condition_logic(trend_cont, pivot_signal, last_pivot_signal, long_pivot_signal, last_long_pivot_signal, pivot_signal_changed):
    if pivot_signal != 0:
        return pivot_signal
    # if pivot_signal != 0 and not pivot_signal_changed:
        # if pivot_signal == None:
        #     return None
        # elif pivot_signal == 1:
        #     if long_pivot_signal == 1:
        #         return 1
        #     else:
        #         return 0

        # elif pivot_signal == -1:
        #     if long_pivot_signal == -1:
        #         return -1
        #     else:
        #         return 0

    else:
        if not smooth:
            return 0
        # if stagnant signal
        if last_pivot_signal == 1:
            if trend_cont > 0:
                return 1 
            else:
                return 0
        elif last_pivot_signal == -1:
            if trend_cont < 0:
                return -1 
            else:
                return 0
        else:
            return last_pivot_signal
    return None 

def market_condition_setup(prcAll, smooth=True):
    # market_condition returns + if up trend, - if down, 0 if stagnant

    ## General vars
    global pivot_signals
    global market_signals
    day = len(prcAll)-1

    ## Trading variables
    look_length = 10 # for pivots!
    long_look_length = 75 # for pivots!
    atr_breakpoint = 1 # for pivots!
    atr_factor = 0.03
    max_atr_factor = 3
    lin_reg_lookback = 50
    lin_factor = 0.50

    ## Altering based on length of trend!
    len_trend = get_len_of_trend(day)
    if not np.isnan(len_trend):
        lin_reg_lookback += math.floor(abs(len_trend)*lin_factor)
        atr_breakpoint += min(abs(len_trend)*atr_factor, max_atr_factor)


    # can use
        # linear reg
        # rsi
        # over ma (50!) price_ma_diff
        # ma over ma, 50 & 200 ma trend (or 25, 100 more reactive)
        # pivot highs & lows, pivot breaking with ll = 10
    
    ## get pivot information
    pivot_signal = pivot_breaking(prcAll, look_length, atr_breakpoint)
    # long_pivot_signal = pivot_breaking(prcAll, long_look_length, atr_breakpoint)
    pivot_signals[day] = pivot_signal
    # long_pivot_signals[day] = long_pivot_signal
    last_pivot_signal = get_last_privot_position(prcAll, pivot_signals)
    # last_long_pivot_signal = get_last_privot_position(prcAll, long_pivot_signals)
    # pivot_signal_changed = pivot_signal != pivot_signals[day-1] if day > 1 else False
    last_long_pivot_signal = 0
    pivot_signal_changed = 0
    long_pivot_signal = 0

    ## get overall trend information
    # trend_cont = ma_ma_diff(prcAll, 25, 100)
    trend_cont = linear_reg(prcAll, lin_reg_lookback)

    ## pass data into logic system & save
    market_signal = market_condition_logic(trend_cont, pivot_signal, last_pivot_signal, long_pivot_signal, last_long_pivot_signal, pivot_signal_changed )
    market_signals[day] = market_signal
    return market_signal

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

def show_multi_panel_graph(prcAll, inst_list, smooth=True):
    """
    Plot multiple instruments' market conditions in a grid of subplots.

    Parameters:
    - prcAll: price matrix of shape (nInst, T)
    - inst_list: list of instrument indices to plot
    - smooth: whether to apply smoothing in market condition logic
    """
    nInst = len(inst_list)
    ncols = 5  # You can change this depending on your screen layout
    nrows = math.ceil(nInst / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 3.5*nrows), squeeze=False)
    regime_colors = {'bullish': 'green', 'bearish': 'red', 'stagnant': 'gray', 'unknown': 'white'}

    for idx, inst in enumerate(inst_list):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        closes = prcAll[inst, :]
        T = len(closes)
        regimes = []

        for t in range(T):
            regime = market_condition_setup(closes[:t+1], smooth)
            regimes.append(regime)

        regimes_str = [market_name(r) for r in regimes]
        ax.plot(closes, color='black', linewidth=1.2)

        for t in range(100, T):
            color = regime_colors.get(regimes_str[t], 'white')
            ax.axvspan(t - 1, t, color=color, alpha=0.2)

        ax.set_title(f"Instrument {inst}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price")

    # Hide unused subplots
    for i in range(nInst, nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row][col].axis('off')

    plt.tight_layout()
    plt.show()

# Graphing functions #

def market_name(num):
    if num == None:
        return 'unknown'
    elif num == 0:
        return 'stagnant'
    elif num > 0:
        return 'bullish'
    return 'bearish'

def draw_market_condition(prcAll, inst):
    # Evaluate regime at each time step
    closes = prcAll[inst, :]
    T = len(closes)
    regimes = []
    for t in range(T):
        # market_condition returns + if up trend, - if down, 0 if stagnant
        regime = market_condition(closes[:t+1], smooth)
        regimes.append(regime)

    # Map regimes to colors
    regimes_str = [market_name(element) for element in regimes]
    regime_colors = {'bullish': 'green', 'bearish': 'red', 'stagnant': 'gray', 'unknown': 'white'}

    # Plotting
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(closes, label='Closing Price', color='black')

    for t in range(100, T):
        color = regime_colors.get(regimes_str[t], 'white')
        ax.axvspan(t - 1, t, color=color, alpha=0.2)

    ax.set_title(f"Market Identification - Inst: {inst}")
    ax.set_xlabel("Time (Days)")
    ax.set_ylabel("Price")

    # draw_ma(prcAll, inst, 25, 100, ax)

    ax.legend()

    plt.tight_layout()
    plt.show()

def draw_ma(prcAll, inst, slow, fast, ax=None):
    closes = prcAll[inst, :]
    period = 14
    # --- Compute moving averages ---
    ma_50 = ma(closes, period=slow, method='sma')
    ma_200 = ma(closes, period=fast, method='sma')

    # --- Plotting ---
    if ax != None:
        ax.plot(ma_50, label=f'{fast}-Day MA', color='blue')
        ax.plot(ma_200, label=f'{slow}-Day MA', color='red')
    else:
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

def main(inst_list, smooth):
    pricesFile="prices.txt"
    prcAll = loadPrices(pricesFile)
    show_multi_panel_graph(prcAll, inst_list, smooth)

## Vars to change for viewing
# Smoothing between pivot breaking
smooth = True

inst_range = True 
(lower_bound, upper_bound) = (10, 30)

if inst_range:
    inst_list = list(range(lower_bound, upper_bound))
else:
    inst_list = [5]

main(inst_list, smooth)