import numpy as np
import pandas as pd

def compute_slope_vol(prices: pd.Series, slope_win: int, vol_win: int) -> pd.DataFrame:
    """
    Computes rolling slope & rolling volatility over full series.
    """
    logp = np.log(prices)
    t = np.arange(slope_win)
    X = np.vstack([t, np.ones_like(t)]).T

    def slope_of_window(y):
        m, _ = np.linalg.lstsq(X, y, rcond=None)[0]
        return m

    slope = (
        logp.rolling(window=slope_win, min_periods=slope_win)
            .apply(slope_of_window, raw=True)
    )
    rtn = logp.diff()
    vol = rtn.rolling(window=vol_win, min_periods=vol_win).std()

    return pd.DataFrame({"slope": slope, "vol": vol})


import numpy as np
import pandas as pd
from precision_labeller import plot_all_regimes_long  # adjust if needed

def build_feature_matrix_from_array(prices_array: np.ndarray,
                                    drop_last: int = 10) -> pd.DataFrame:
    """
    prices_array: 2D NumPy array of shape (n_inst, T),
                  each row is one instrument’s price history.
    drop_last:     number of final timesteps to trim (as in original pipeline).
    Returns a pandas DataFrame whose rows correspond to each
    (instrument, time) pair up to T - drop_last - 1, with columns:
      ['ma_reg', 'ema_reg', 'slope_vol_reg', 'macd_reg',
       'kalman_reg', 'fib_reg', 'psar_reg', 'zscore_reg', 'wret_reg',
       'true_regime', 'inst', 'time']
    exactly as in your original CSV output.
    """
    # Convert to the same shape as your original df_raw: (T, n_inst)
    df_raw = pd.DataFrame(prices_array.T)
    T, n_inst = df_raw.shape

    all_rows = []

    for inst in range(n_inst):
        close = df_raw.iloc[:, inst]
        high  = close.copy()
        low   = close.copy()

        # get ground-truth regimes for this instrument
        true_regs = plot_all_regimes_long(end_point=T, plot_graph=False, inst=inst)
        true_regs = pd.Series(true_regs[: T - drop_last],
                              name="true_regime",
                              index=np.arange(T - drop_last))

        features = pd.DataFrame(index=true_regs.index)

        # — MA-based regimes —
        logp = np.log(close)
        ma_s = logp.rolling(window=5,  min_periods=1).mean()
        ma_l = logp.rolling(window=70, min_periods=1).mean()
        features["ma_reg"] = np.where(ma_l > ma_s, 0, 2)[: T - drop_last]

        # — EMA-based regimes —
        ema_s = logp.ewm(span=5,  adjust=False).mean()
        ema_l = logp.ewm(span=50, adjust=False).mean()
        features["ema_reg"] = np.where(ema_s > ema_l, 2, 0)[: T - drop_last]

        # — Slope/Vol regimes —
        sv_df = compute_slope_vol(close, slope_win=30, vol_win=100).dropna()
        idx   = sv_df.index[sv_df.index < T - drop_last]
        slope = sv_df.loc[idx, "slope"]
        vol   = sv_df.loc[idx, "vol"]
        median_vol = vol.median()
        features.loc[idx, "slope_vol_reg"] = np.where(
            (slope > 0) & (vol < median_vol), 2, 0
        )

        # — MACD regimes —
        ema_s2 = logp.ewm(span=50, adjust=False).mean()
        ema_l2 = logp.ewm(span=90, adjust=False).mean()
        macd   = ema_s2 - ema_l2
        signal = macd.ewm(span=40, adjust=False).mean()
        features["macd_reg"] = np.where(
            macd[: T - drop_last] > signal[: T - drop_last], 2, 0
        )

        # — Kalman regimes —
        x_est = np.zeros(T)
        P     = np.zeros(T)
        x_est[0], P[0] = logp.iloc[0], 1.0
        for t in range(1, T):
            x_pred = x_est[t-1]
            P_pred = P[t-1] + 0.01
            K      = P_pred / (P_pred + 10.0)
            x_est[t] = x_pred + K * (logp.iloc[t] - x_pred)
            P[t]     = (1 - K) * P_pred
        features["kalman_reg"] = np.where(
            logp[: T - drop_last] > x_est[: T - drop_last], 2, 0
        )

        # — Fibonacci regimes —
        high_win   = close.rolling(window=50, min_periods=50).max()
        low_win    = close.rolling(window=50, min_periods=50).min()
        fib_range  = high_win - low_win
        lower, upper = (
            low_win + 0.618 * fib_range,
            low_win + 0.786 * fib_range,
        )
        features["fib_reg"] = np.where(
            close[: T - drop_last] > upper[: T - drop_last], 2,
            np.where(close[: T - drop_last] < lower[: T - drop_last], 0, 1)
        )

        # — PSAR regimes —
        psar     = np.zeros(T)
        af, max_step = 0.01, 0.10
        trend_up = True
        ep       = high.iloc[0]
        psar[0]  = low.iloc[0]
        for t in range(1, T):
            prev = psar[t-1]
            psar[t] = prev + af * (ep - prev)
            if trend_up:
                if low.iloc[t] < psar[t]:
                    trend_up = False
                    psar[t]  = ep
                    ep       = low.iloc[t]
                    af       = 0.01
                elif high.iloc[t] > ep:
                    ep = high.iloc[t]
                    af = min(af + 0.01, max_step)
            else:
                if high.iloc[t] > psar[t]:
                    trend_up = True
                    psar[t]  = ep
                    ep       = high.iloc[t]
                    af       = 0.01
                elif low.iloc[t] < ep:
                    ep = low.iloc[t]
                    af = min(af + 0.01, max_step)
        features["psar_reg"] = np.where(
            close[: T - drop_last] > psar[: T - drop_last], 2, 0
        )

        # — Z-score regimes —
        ma90 = close.rolling(window=90, min_periods=90).mean()
        sd90 = close.rolling(window=90, min_periods=90).std()
        z    = (close - ma90) / sd90
        features["zscore_reg"] = np.where(
            z[: T - drop_last] > 0.5, 2,
            np.where(z[: T - drop_last] < -0.5, 0, 1)
        )

        # — Weighted-return regimes —
        r       = close.pct_change()
        weights = np.arange(1, 46) ** 0.5
        weights /= weights.sum()
        wr = r.rolling(window=45, min_periods=45).apply(
            lambda x: np.dot(x, weights), raw=True
        )
        features["wret_reg"] = np.where(
            wr[: T - drop_last] >  0, 2,
            np.where(wr[: T - drop_last] <  0, 0, 1)
        )

        # metadata
        features["inst"] = inst
        features["time"] = features.index

        all_rows.append(features.reset_index(drop=True))

    final_df = pd.concat(all_rows, ignore_index=True)
    return final_df

