import numpy as np
import pandas as pd
from indicators import (
    rolling_percentile, up_down_streak, sma, ema, macd, adx_from_closes,
    parabolic_sar, rsi, stochastic_oscillator, roc,
    commodity_channel_index, atr_close_to_close,
    donchian_channel_percentile, rolling_std, zscore,
    linear_reg, pivot_breaking, get_len_of_trend
)

# Load data
price_df = pd.read_csv("prices.txt", sep=r"\s+", header=None)
price_df = price_df.iloc[:500]

# pivot parameters
look_lengths   = [6, 10, 50, 100]
atr_factor     = 0.03
max_atr_factor = 3

element_feats = []

for inst in price_df.columns:
    prices = price_df[inst].values
    n = len(prices)

    # --- build dynamic pivot_break arrays ---
    pivot_dynamic = {}
    for ll in look_lengths:
        signals = []
        for t in range(n):
            window = prices[: t+1]

            # 1) compute how long the current trend has run
            len_trend = get_len_of_trend(t)

            # 2) base breakpoint is 1, bump by trend*factor up to max
            if not np.isnan(len_trend):
                atr_bp = 1 + min(abs(len_trend) * atr_factor, max_atr_factor)
            else:
                atr_bp = 1

            # 3) call your pivot_breaking with that breakpoint
            pv = pivot_breaking(window, look_length=ll, atr_breakpoint=atr_bp)
            signals.append(0 if pv is None else pv)

        pivot_dynamic[f"pivot_break_{ll}"] = np.array(signals)

    # --- now build all your other features as before ---
    feat_arrays = {
                # — Relative price position
        'roll_pct_10':   rolling_percentile(prices, window=10),
        'roll_pct_20':   rolling_percentile(prices, window=20),
        'roll_pct_50':   rolling_percentile(prices, window=50),

        # — Streaks
        'streak_up':     up_down_streak(prices, direction="up"),
        'streak_down':   up_down_streak(prices, direction="down"),

        # — Simple Moving Averages
        'sma_10':        sma(prices, window=10),
        'sma_20':        sma(prices, window=20),
        'sma_50':        sma(prices, window=50),
        'sma_100':       sma(prices, window=100),

        # — Exponential Moving Averages
        'ema_12':        ema(prices, span=12),
        'ema_26':        ema(prices, span=26),
        'ema_50':        ema(prices, span=50),
        'ema_100':       ema(prices, span=100),

        # — Trend strength
        'adx_14':        adx_from_closes(prices, period=14),

        # — Oscillators
        'rsi_6':         rsi(prices, window=6),
        'rsi_9':         rsi(prices, window=9),
        'rsi_14':        rsi(prices, window=14),
        'rsi_21':        rsi(prices, window=21),

        'sto_k_6':       stochastic_oscillator(prices, window=6),
        'sto_k_14':      stochastic_oscillator(prices, window=14),
        'sto_k_21':      stochastic_oscillator(prices, window=21),

        # — Volatility
        'atr_14':        atr_close_to_close(prices, window=14),
        'atr_21':        atr_close_to_close(prices, window=21),
        'std_10':        rolling_std(prices, window=10),
        'std_20':        rolling_std(prices, window=20),
        'std_50':        rolling_std(prices, window=50),

        # — Momentum
        'roc_5':         roc(prices, period=5),
        'roc_10':        roc(prices, period=10),
        'roc_20':        roc(prices, period=20),
        'roc_50':        roc(prices, period=50),

        # — Commodity Channel Index
        'cci_20':        commodity_channel_index(prices, window=20),

        # — Donchian Channel percentile
        'donch_pct_20':  donchian_channel_percentile(prices, window=20),
        'donch_pct_50':  donchian_channel_percentile(prices, window=50),

        # — Z-Score
        'z_20':          zscore(prices, window=20),
        'z_50':          zscore(prices, window=50),

        # — MACD (default 12,26,9)
        'macd_line':     macd(prices)[0],
        'macd_signal':   macd(prices)[1],

        # — Parabolic SAR variants
        'psar_02_2':     parabolic_sar(prices, initial_af=0.02, max_af=0.2),
        'psar_01_05':    parabolic_sar(prices, initial_af=0.01, max_af=0.05),
        'psar_005_025':  parabolic_sar(prices, initial_af=0.005, max_af=0.025),

        # — Linear regression slopes
        'slope_10':      np.full_like(prices, linear_reg(prices, look_back=10)),
        'slope_50':      np.full_like(prices, linear_reg(prices, look_back=50)),
        'slope_75':      np.full_like(prices, linear_reg(prices, look_back=75)),
        'slope_100':     np.full_like(prices, linear_reg(prices, look_back=100)),

        # — Price minus SMA
        'price_minus_sma_12':  prices - sma(prices, window=12),
        'price_minus_sma_26':  prices - sma(prices, window=26),
        'price_minus_sma_50':  prices - sma(prices, window=50),
        'price_minus_sma_100': prices - sma(prices, window=100),

        # — SMA minus SMA
        'sma_25_100_diff':     sma(prices, window=25)  - sma(prices, window=100),
        'sma_12_26_diff':      sma(prices, window=12)  - sma(prices, window=26),

        # — Pivot‐break fla
    }
    feat_arrays.update(pivot_dynamic)

    # assemble DataFrame for this instrument
    inst_df = pd.DataFrame(feat_arrays, index=price_df.index)
    inst_df['inst'] = inst
    inst_df['time'] = inst_df.index
    element_feats.append(inst_df)

# Concatenate all instruments into one tabular DataFrame
tab_df = pd.concat(element_feats, axis=0)
tab_df.set_index(['inst', 'time'], inplace=True)
tab_df = tab_df.dropna()

print(tab_df.shape)
#tab_df.to_csv("features_heavy1.csv")



