import numpy as np
import pandas as pd
from indicators_local import (
    rolling_percentile, rolling_simple_return, up_down_streak, sma, ema, macd, adx_from_closes,
    parabolic_sar, rsi, stochastic_oscillator, roc,
    commodity_channel_index, atr_close_to_close,
    donchian_channel_percentile, rolling_std, zscore,
    linear_reg, rolling_simple_return,
    rolling_std, sma, bollinger_band_width, bollinger_percent_b,log_returns,
    rolling_mean_log_returns, rolling_std_log_returns, volatility_ratio,
    kama, first_derivative, second_derivative,
)

def pipeline_regiem():

    print('Pipeline Online...')
    price_df = pd.read_csv("prices.txt", sep=r"\s+", header=None)
    price_df = price_df.iloc[:]

    element_feats = []

    for inst in price_df.columns:
        prices = price_df[inst].values
        n = len(prices)

        # --- now build all your other features as before ---
        feat_arrays = {
        
            'close':            prices,
            'log_price':        np.log(prices),

            'roll_pct_20':   rolling_percentile(prices, window=20),
            'roll_pct_50':   rolling_percentile(prices, window=50),

            # — Rolling simple returns
            'avg_ret_30': rolling_simple_return(prices, window=30),
            'avg_ret_50': rolling_simple_return(prices, window=50),
            'avg_ret_100': rolling_simple_return(prices, window=100),

            # — Streaks
            'streak_up':     up_down_streak(prices, direction="up"),
            'streak_down':   up_down_streak(prices, direction="down"),

            # — Simple Moving Averages
            'sma_20':        sma(prices, window=20),
            'sma_50':        sma(prices, window=50),
            'sma_100':       sma(prices, window=100),

            # — Exponential Moving Averages
            'ema_26':        ema(prices, span=26),
            'ema_50':        ema(prices, span=50),
            'ema_100':       ema(prices, span=100),

            # — Trend strength
            'adx_14':        adx_from_closes(prices, period=14),

            # — Oscillators
            'rsi_14':        rsi(prices, window=14),
            'rsi_21':        rsi(prices, window=21),

            'sto_k_14':      stochastic_oscillator(prices, window=14),
            'sto_k_21':      stochastic_oscillator(prices, window=21),

            # — Volatility
            'atr_21':        atr_close_to_close(prices, window=21),
            'std_10':        rolling_std(prices, window=10),
            'std_20':        rolling_std(prices, window=20),
            'std_50':        rolling_std(prices, window=50),

            # — Momentum
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
            'price_minus_sma_26':  prices - sma(prices, window=26),
            'price_minus_sma_50':  prices - sma(prices, window=50),
            'price_minus_sma_100': prices - sma(prices, window=100),

            # — SMA minus SMA
            'sma_25_100_diff':     sma(prices, window=25)  - sma(prices, window=100),
            'sma_12_26_diff':      sma(prices, window=12)  - sma(prices, window=26),

            'bb_width_30':    bollinger_band_width(prices, window=30),
            'percent_b_30':   bollinger_percent_b(prices, window=30),
            'bb_width_100':   bollinger_band_width(prices, window=100),
            'percent_b_100':  bollinger_percent_b(prices, window=100),


            # — Log‐returns & rolling stats (windows aligned with regimes)
            'log_ret_1':        log_returns(prices, period=1),
            'lr_mean_30':       rolling_mean_log_returns(prices, window=30),
            'lr_std_30':        rolling_std_log_returns(prices, window=30),
            'lr_mean_100':      rolling_mean_log_returns(prices, window=100),
            'lr_std_100':       rolling_std_log_returns(prices, window=100),


            # — Volatility ratio (short vs. long vol)
            'vol_ratio_30_100': volatility_ratio(prices, short_window=30, long_window=100),


            # — Kaufman’s Adaptive MA
            'kama_30':         kama(prices, window=30),
            'kama_100':        kama(prices, window=100),

            # — Derivatives
            'velocity':        first_derivative(prices),
            'acceleration':    second_derivative(prices),

        }

        # assemble DataFrame for this instrument
        inst_df = pd.DataFrame(feat_arrays, index=price_df.index)
        inst_df['inst'] = inst
        inst_df['time'] = inst_df.index
        element_feats.append(inst_df)

    # Concatenate all instruments into one tabular DataFrame
    tab_df = pd.concat(element_feats, axis=0)
    tab_df.set_index(['inst', 'time'], inplace=True)
    tab_df = tab_df.dropna()


    return tab_df


#print(tab_df.shape)
#tab_df.to_csv("features_heavy3.csv")




