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

#

### Trend Indicators ###
## These help identify the direction and strength of a market trend.


# -----------------------------------------New indicators 24/06 (Generated my yours truly Chat-GoatPT)--------------------------------------- #
import numpy as np
from scipy.stats import percentileofscore

def rolling_simple_return(close: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling average simple return over `window` days:
      (close_t / close_{t-window}) - 1, then averaged over the window
    """
    n = len(close)
    result = np.full(n, np.nan, dtype=np.float64)
    for t in range(window, n):
        # slice out the last `window` returns
        rets = close[t-window+1 : t+1] / close[t-window : t] - 1
        result[t] = np.nanmean(rets)
    return result

def rolling_std(close: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling standard deviation over the given window.
    """
    result = np.full_like(close, np.nan, dtype=np.float64)
    for i in range(window - 1, len(close)):
        result[i] = np.std(close[i - window + 1 : i + 1], ddof=0)
    return result

def sma(close: np.ndarray, window: int) -> np.ndarray:
    """
    Simple moving average over the given window.
    """
    result = np.full_like(close, np.nan, dtype=np.float64)
    for i in range(window - 1, len(close)):
        result[i] = np.mean(close[i - window + 1 : i + 1])
    return result

def bollinger_band_width(close: np.ndarray, window: int) -> np.ndarray:
    """
    Bollinger Band Width: measures total band width (upper - lower) = 4 * rolling std.
    """
    s = rolling_std(close, window)
    return 4 * s

def bollinger_percent_b(close: np.ndarray, window: int) -> np.ndarray:
    """
    Bollinger %B: position of price within the Bollinger Bands on a 0–1 scale.
    """
    m = sma(close, window)
    s = rolling_std(close, window)
    lower = m - 2 * s
    width = 4 * s
    result = np.full_like(close, np.nan, dtype=np.float64)
    mask = width != 0
    result[mask] = (close[mask] - lower[mask]) / width[mask]
    return result

def log_returns(close: np.ndarray, period: int = 1) -> np.ndarray:
    """
    Log-returns: ln(close_t / close_{t-period}).
    """
    result = np.full_like(close, np.nan, dtype=np.float64)
    for i in range(period, len(close)):
        result[i] = np.log(close[i] / close[i - period])
    return result

def rolling_mean_log_returns(close: np.ndarray, window: int, period: int = 1) -> np.ndarray:
    """
    Rolling mean of log-returns over the given window.
    """
    lr = log_returns(close, period)
    result = np.full_like(close, np.nan, dtype=np.float64)
    start = window - 1 + period
    for i in range(start, len(close)):
        result[i] = np.nanmean(lr[i - window + 1 : i + 1])
    return result

def rolling_std_log_returns(close: np.ndarray, window: int, period: int = 1) -> np.ndarray:
    """
    Rolling std of log-returns over the given window.
    """
    lr = log_returns(close, period)
    result = np.full_like(close, np.nan, dtype=np.float64)
    start = window - 1 + period
    for i in range(start, len(close)):
        result[i] = np.nanstd(lr[i - window + 1 : i + 1])
    return result

def volatility_ratio(close: np.ndarray, short_window: int, long_window: int) -> np.ndarray:
    """
    Volatility ratio: rolling_std(short_window) / rolling_std(long_window).
    """
    vol_s = rolling_std(close, short_window)
    vol_l = rolling_std(close, long_window)
    result = np.full_like(close, np.nan, dtype=np.float64)
    mask = vol_l != 0
    result[mask] = vol_s[mask] / vol_l[mask]
    return result

def kama(close: np.ndarray, window: int, fast_period: int = 2, slow_period: int = 30) -> np.ndarray:
    """
    Kaufman’s Adaptive Moving Average (KAMA).
    """
    n = len(close)
    result = np.full(n, np.nan, dtype=np.float64)
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)

    if n >= window:
        result[window - 1] = np.mean(close[:window])

    for t in range(window, n):
        direction = abs(close[t] - close[t - window])
        volatility = np.sum(np.abs(close[t - window + 1 : t + 1] - close[t - window : t]))
        er = direction / volatility if volatility != 0 else 0
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        result[t] = result[t - 1] + sc * (close[t] - result[t - 1])

    return result

def first_derivative(close: np.ndarray) -> np.ndarray:
    """
    First derivative (velocity) of closing prices:
    Δprice_t = price_t − price_{t−1}
    """
    n = len(close)
    result = np.full(n, np.nan, dtype=np.float64)
    for i in range(1, n):
        result[i] = close[i] - close[i-1]
    return result

def second_derivative(close: np.ndarray) -> np.ndarray:
    """
    Second derivative (acceleration) of closing prices:
    Δ²price_t = price_t − 2·price_{t−1} + price_{t−2}
    """
    n = len(close)
    result = np.full(n, np.nan, dtype=np.float64)
    for i in range(2, n):
        result[i] = close[i] - 2*close[i-1] + close[i-2]
    return result



# ------------------------------------------------------------------------------------------------------ #

# the rank of price compared to previous days (0-1 float)
def rolling_percentile(close: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling Percentile Rank
    Measures how today's price ranks in the last 'window' days.
    Captures price extremity in context of recent range.
    """
    result = np.full_like(close, np.nan, dtype=np.float64)
    for i in range(window - 1, len(close)):
        window_vals = close[i - window + 1:i + 1]
        result[i] = percentileofscore(window_vals, window_vals[-1]) / 100
    return result

# Trending Streak integer
def up_down_streak(close: np.ndarray, direction: str = 'up') -> np.ndarray:
    """
    Up or Down Streak Counter
    Tracks consecutive gains or losses.
    Helpful for spotting overextended trends or consolidations.
    """
    streak = np.zeros_like(close)
    comp = np.greater if direction == 'up' else np.less
    for i in range(1, len(close)):
        streak[i] = streak[i-1] + 1 if comp(close[i], close[i-1]) else 0
    return streak

# Moving Averages (MA)

# Simple Moving Average (SMA)
def sma(close: np.ndarray, window: int) -> np.ndarray:
    """
    Simple Moving Average (SMA)
    Smooths price data by averaging over a window.
    Useful for identifying trend direction.
    """
    return pd.Series(close).rolling(window).mean().to_numpy()

# Exponential Moving Average (EMA)
def ema(close: np.ndarray, span: int) -> np.ndarray:
    """
    Exponential Moving Average (EMA)
    Like SMA but more responsive to recent changes.
    Captures faster shifts in trend.
    """
    return pd.Series(close).ewm(span=span, adjust=False).mean().to_numpy()

# Moving Average Convergence Divergence (MACD)
def macd(close: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Moving Average Convergence Divergence (MACD)
    Difference between short- and long-term EMAs (12 vs 26).
    Combined with signal line to detect trend shifts.
    """
    ema_12 = ema(close, 12)
    ema_26 = ema(close, 26)
    macd_line = ema_12 - ema_26
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False).mean().to_numpy()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


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
def parabolic_sar(close: np.ndarray,
                  initial_af: float = 0.02,
                  max_af: float = 0.2) -> np.ndarray:
    """
    Parabolic SAR approximation using closing prices only.
    Normally uses high/low data, but this version simulates trend following behavior
    from close prices alone. Useful as a trend direction tracker.

    Parameters:
    - close: np.ndarray of closing prices
    - initial_af: starting acceleration factor (default 0.02)
    - max_af: maximum acceleration factor (default 0.2)

    Returns:
    - np.ndarray of SAR values (same length as close), with np.nan before trend start
    """

    sar = np.full_like(close, np.nan)
    trend = 1 if close[1] > close[0] else -1  # Start with basic trend
    af = initial_af
    ep = close[1]  # extreme point (highest close in uptrend, lowest in downtrend)
    sar_val = close[0]

    for i in range(2, len(close)):
        prev_sar = sar_val
        if trend == 1:
            ep = max(ep, close[i])
            sar_val = sar_val + af * (ep - sar_val)
            if close[i] < sar_val:
                # Reverse to downtrend
                trend = -1
                sar_val = ep  # start SAR at last EP
                ep = close[i]
                af = initial_af
            else:
                if close[i] > ep:
                    ep = close[i]
                    af = min(af + initial_af, max_af)

        else:
            ep = min(ep, close[i])
            sar_val = sar_val + af * (ep - sar_val)
            if close[i] > sar_val:
                # Reverse to uptrend
                trend = 1
                sar_val = ep
                ep = close[i]
                af = initial_af
            else:
                if close[i] < ep:
                    ep = close[i]
                    af = min(af + initial_af, max_af)

        sar[i] = sar_val

    return sar

# Ichimoku Cloud

# ------------------------------------------------------------------------------------------------------ #

### Momentum Indicators ###
## These show the speed of price movements and are useful for identifying overbought or oversold conditions.

# Relative Strength Index (RSI)
def rsi(close: np.ndarray, window: int) -> np.ndarray:
    """
    Relative Strength Index (RSI)
    Oscillator between 0 and 100 reflecting price strength.
    >70 may indicate overbought, <30 oversold.
    """
    delta = np.diff(close, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_series = pd.Series(gain).rolling(window).mean()
    loss_series = pd.Series(loss).rolling(window).mean()
    
    rs = gain_series / (loss_series + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.to_numpy()

# Stochastic Oscillator - computes where % of the price is relative to max and min over last window
def stochastic_oscillator(close: np.ndarray, window: int = 14) -> np.ndarray:
    """
    Computes the Stochastic Oscillator %K using only closing prices.
    %K = (Close - Min) / (Max - Min), scaled between 0 and 1.

    Parameters:
    - close: array of closing prices
    - window: lookback period

    Returns:
    - %K array (NumPy), values from 0 to 1
    """
    close = pd.Series(close)
    lowest = close.rolling(window).min()
    highest = close.rolling(window).max()
    percent_k = (close - lowest) / (highest - lowest)
    return percent_k.to_numpy()

# Rate of Change (ROC)
def roc(close: np.ndarray, period: int) -> np.ndarray:
    """
    Rate of Change (ROC)
    Measures percentage change from 'period' days ago.
    Indicates strength and direction of momentum.
    """
    return (close / np.roll(close, period)) - 1 if period < len(close) else np.full_like(close, np.nan)

# Commodity Channel Index (CCI) - measures price deviation from its average over a period
def commodity_channel_index(close: np.ndarray, window: int = 20) -> np.ndarray:
    """
    Approximates the Commodity Channel Index (CCI) using closing prices only.
    CCI = (Close - SMA) / (0.015 * Mean Absolute Deviation)

    Parameters:
    - close: array of closing prices
    - window: lookback period

    Returns:
    - CCI values (NumPy), centered around 0
    """
    close = pd.Series(close)
    sma = close.rolling(window).mean()
    mad = close.rolling(window).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (close - sma) / (0.015 * mad)
    return cci.to_numpy()

# ------------------------------------------------------------------------------------------------------ #

### Volatility Indicators ###
## These measure how much the price is changing and indicate market uncertainty or risk.

# Bollinger Bands

# Average True Range (ATR) (Close-to-Close Volatility approximates this)
def atr_close_to_close(close: np.ndarray, window: int = 14) -> np.ndarray:
    """
    Approximate Average True Range (ATR)
    Normally uses high/low/close; here approximated with abs diff of close.
    Measures volatility for detecting quiet vs explosive periods.
    """
    diff = np.abs(np.diff(close, prepend=np.nan))
    return pd.Series(diff).rolling(window).mean().to_numpy()

# Donchian Channels
def donchian_channel_percentile(close: np.ndarray, window: int = 20) -> np.ndarray:
    """
    Computes the percentile of the current close within the Donchian Channel range.

    Formula:
    Percentile = (Close - Min) / (Max - Min), scaled from 0 to 1

    Parameters:
    - close: array of closing prices
    - window: lookback period for high/low band

    Returns:
    - Percentile location of close in band (NumPy), values from 0 to 1
    """
    close = pd.Series(close)
    lowest = close.rolling(window).min()
    highest = close.rolling(window).max()
    percentile = (close - lowest) / (highest - lowest)
    return percentile.to_numpy()

# Keltner Channels

# ------------------------------------------------------------------------------------------------------ #

### Others / Composite Tools ###
## More complex or composite tools that combine several data sources.

# Rolling ST DEV - useful for stagnant
def rolling_std(close: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling Standard Deviation
    Measures local price volatility.
    Useful for detecting stagnation or explosive moves.
    """
    return pd.Series(close).rolling(window).std().to_numpy()

# zscore - useful for outbreaks + trend reversals
def zscore(close: np.ndarray, window: int) -> np.ndarray:
    """
    Z-Score
    Normalized deviation from the rolling mean.
    Highlights relative overbought/oversold conditions.
    """
    series = pd.Series(close)
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return ((series - mean) / std).to_numpy()

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

def draw_sar(prcAll, inst, initial_af=0.02, max_af=0.2):
    """
    Draws the closing price and corresponding Parabolic SAR approximation.

    Parameters:
    - prcAll: 2D NumPy array [instruments × time]
    - inst: index of instrument to plot
    - initial_af: initial acceleration factor for SAR (default 0.02)
    - max_af: maximum acceleration factor for SAR (default 0.2)
    """
    closes = prcAll[inst, :]
    sar_vals = parabolic_sar(closes, initial_af=initial_af, max_af=max_af)

    # Determine SAR above/below for styling
    uptrend = closes > sar_vals
    downtrend = ~uptrend

    # --- Plotting ---
    plt.figure(figsize=(12, 6))
    plt.plot(closes, label='Close Price', color='black')

    # SAR dots for uptrends and downtrends
    plt.plot(np.where(uptrend, sar_vals, np.nan), marker='.', linestyle='None', color='green', label='SAR (Uptrend)')
    plt.plot(np.where(downtrend, sar_vals, np.nan), marker='.', linestyle='None', color='red', label='SAR (Downtrend)')

    plt.title("Closing Price with Parabolic SAR (Close-only approximation)")
    plt.xlabel("Day")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

### IMPORTING AND CHECKING ###
def loadPrices(fn):
    global nt, nInst
    df=pd.read_csv(fn, sep=r'\s+', header=None, index_col=None)
    (nt,nInst) = df.shape
    return (df.values).T

def main(inst_list, smooth):
    pricesFile="prices.txt"
    prcAll = loadPrices(pricesFile)
    show_multi_panel_graph(prcAll, inst_list, smooth)
    # draw_sar(prcAll, 0, 0.01, 0.05)

## Vars to change for viewing
# Smoothing between pivot breaking

if __name__ == "__main__":
    smooth = True

    inst_range = False
    (lower_bound, upper_bound) = (10, 30)

    if inst_range:
        inst_list = list(range(lower_bound, upper_bound))
    else:
        inst_list = [39, 27]


    main(inst_list, smooth)