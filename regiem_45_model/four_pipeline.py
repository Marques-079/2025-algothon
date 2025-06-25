############################################################
#  pre_proc_pipeline.py                                    #
# -------------------------------------------------------- #
# This file exposes one public function                    #
#                                                          #
#     pipeline_regiem(df_raw: pd.DataFrame) -> pd.DataFrame#
#                                                          #
# * `df_raw` must have a MultiIndex (inst, time) and a     #
#   single column named "close" with raw close prices.     #
# * The function returns a MultiIndex DataFrame of shape   #
#   (n_inst * n_timesteps, 59) with all engineered         #
#   features, indexed on the same (inst, time).            #
############################################################

from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators_local import (
    rolling_percentile, rolling_simple_return, up_down_streak, sma, ema, macd,
    adx_from_closes, parabolic_sar, rsi, stochastic_oscillator, roc,
    commodity_channel_index, atr_close_to_close, donchian_channel_percentile,
    rolling_std, zscore, linear_reg, bollinger_band_width, bollinger_percent_b,
    log_returns, rolling_mean_log_returns, rolling_std_log_returns,
    volatility_ratio, kama, first_derivative, second_derivative,
)

# --------------------------------------------------------------------------- #
# Helper: build all features for one 1‑D price array ------------------------- #
# --------------------------------------------------------------------------- #

def _build_feature_dict(prices: np.ndarray) -> dict[str, np.ndarray]:
    """Return a dict of name -> 1‑D np.ndarray (same length as *prices*)."""

    # basic lengths
    n = len(prices)

    return {
        # raw & log price ----------------------------------------------------
        "close":             prices,
        "log_price":         np.log(prices + 1e-9),

        # rolling percentiles ----------------------------------------------
        "roll_pct_20":       rolling_percentile(prices, 20),
        "roll_pct_50":       rolling_percentile(prices, 50),

        # returns -----------------------------------------------------------
        "avg_ret_30":        rolling_simple_return(prices, 30),
        "avg_ret_50":        rolling_simple_return(prices, 50),
        "avg_ret_100":       rolling_simple_return(prices, 100),

        # streaks -----------------------------------------------------------
        "streak_up":         up_down_streak(prices, "up"),
        "streak_down":       up_down_streak(prices, "down"),

        # smas --------------------------------------------------------------
        "sma_20":            sma(prices, 20),
        "sma_50":            sma(prices, 50),
        "sma_100":           sma(prices, 100),

        # emas --------------------------------------------------------------
        "ema_26":            ema(prices, 26),
        "ema_50":            ema(prices, 50),
        "ema_100":           ema(prices, 100),

        # trend strength ----------------------------------------------------
        "adx_14":            adx_from_closes(prices, 14),

        # oscillators -------------------------------------------------------
        "rsi_14":            rsi(prices, 14),
        "rsi_21":            rsi(prices, 21),
        "sto_k_14":          stochastic_oscillator(prices, 14),
        "sto_k_21":          stochastic_oscillator(prices, 21),

        # volatility --------------------------------------------------------
        "atr_21":            atr_close_to_close(prices, 21),
        "std_10":            rolling_std(prices, 10),
        "std_20":            rolling_std(prices, 20),
        "std_50":            rolling_std(prices, 50),

        # momentum ----------------------------------------------------------
        "roc_20":            roc(prices, 20),
        "roc_50":            roc(prices, 50),

        # cci ---------------------------------------------------------------
        "cci_20":            commodity_channel_index(prices, 20),

        # donchian ----------------------------------------------------------
        "donch_pct_20":      donchian_channel_percentile(prices, 20),
        "donch_pct_50":      donchian_channel_percentile(prices, 50),

        # z‑scores ----------------------------------------------------------
        "z_20":              zscore(prices, 20),
        "z_50":              zscore(prices, 50),

        # macd --------------------------------------------------------------
        "macd_line":         macd(prices)[0],
        "macd_signal":       macd(prices)[1],

        # psar --------------------------------------------------------------
        "psar_02_2":         parabolic_sar(prices, 0.02, 0.2),
        "psar_01_05":        parabolic_sar(prices, 0.01, 0.05),
        "psar_005_025":      parabolic_sar(prices, 0.005, 0.025),

        # slopes ------------------------------------------------------------
        "slope_10":          linear_reg(prices, 10),
        "slope_50":          linear_reg(prices, 50),
        "slope_75":          linear_reg(prices, 75),
        "slope_100":         linear_reg(prices, 100),

        # price minus sma ---------------------------------------------------
        "price_minus_sma_26":  prices - sma(prices, 26),
        "price_minus_sma_50":  prices - sma(prices, 50),
        "price_minus_sma_100": prices - sma(prices, 100),

        # sma minus sma -----------------------------------------------------
        "sma_25_100_diff":     sma(prices, 25) - sma(prices, 100),
        "sma_12_26_diff":      sma(prices, 12) - sma(prices, 26),

        # bollinger ---------------------------------------------------------
        "bb_width_30":         bollinger_band_width(prices, 30),
        "percent_b_30":        bollinger_percent_b(prices, 30),
        "bb_width_100":        bollinger_band_width(prices, 100),
        "percent_b_100":       bollinger_percent_b(prices, 100),

        # log-return stats --------------------------------------------------
        "log_ret_1":           log_returns(prices, 1),
        "lr_mean_30":          rolling_mean_log_returns(prices, 30),
        "lr_std_30":           rolling_std_log_returns(prices, 30),
        "lr_mean_100":         rolling_mean_log_returns(prices, 100),
        "lr_std_100":          rolling_std_log_returns(prices, 100),

        # volatility ratio --------------------------------------------------
        "vol_ratio_30_100":    volatility_ratio(prices, 30, 100),

        # kama --------------------------------------------------------------
        "kama_30":             kama(prices, 30),
        "kama_100":            kama(prices, 100),

        # derivatives -------------------------------------------------------
        "velocity":            first_derivative(prices),
        "acceleration":        second_derivative(prices),
    }

# --------------------------------------------------------------------------- #

def pipeline_regiem(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Compute heavy feature set for every time‑step of every instrument.

    Parameters
    ----------
    df_raw : DataFrame
        MultiIndex (inst, time) with exactly one column ``'close'``.

    Returns
    -------
    DataFrame
        MultiIndex (inst, time) × 59 feature columns. Rows containing
        NA in *any* column are dropped.
    """
    if not isinstance(df_raw.index, pd.MultiIndex):
        raise ValueError("df_raw must have a MultiIndex (inst, time)")
    if list(df_raw.columns) != ["close"]:
        raise ValueError("df_raw must have exactly one column named 'close'")

    feats_list: list[pd.DataFrame] = []

    for inst, grp in df_raw.groupby(level="inst", sort=False):
        prices = grp["close"].values
        feature_dict = _build_feature_dict(prices)
        inst_df = pd.DataFrame(feature_dict, index=grp.index.get_level_values("time"))
        inst_df["inst"] = inst
        inst_df["time"] = inst_df.index
        feats_list.append(inst_df)

    out = pd.concat(feats_list, axis=0)
    out.set_index(["inst", "time"], inplace=True)
    out = out.dropna()
    return out
