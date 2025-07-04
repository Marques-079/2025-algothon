# regime_features.py

import numpy as np
import pandas as pd

def _ols_slope(y: np.ndarray) -> float:
    t = np.arange(len(y))
    X = np.vstack([t, np.ones_like(t)]).T
    m, _ = np.linalg.lstsq(X, y, rcond=None)[0]
    return m

def _slope_vol_reg(close: np.ndarray,
                   idx:   int,
                   slope_win: int = 30,
                   vol_win:   int = 100
                  ) -> float | int:
    logp = np.log(close)
    slope_series = (
        pd.Series(logp)
          .rolling(slope_win, min_periods=slope_win)
          .apply(_ols_slope, raw=True)
    )
    rtn        = pd.Series(logp).diff()
    vol_series = rtn.rolling(vol_win, min_periods=1).std()

    slope = slope_series.iloc[idx]
    vol   = vol_series.iloc[idx]
    if np.isnan(slope) or np.isnan(vol):
        return np.nan

    median_vol = vol_series.iloc[: idx + 1].median()
    return 2 if (slope > 0 and vol < median_vol) else 0

def compute_regime_features_window(prices_window: np.ndarray) -> np.ndarray:
    """
    prices_window: (n_inst, win_len)
    returns (n_inst, 9) of indicators
    """
    n_inst, win_len = prices_window.shape
    idx = win_len - 1

    out = np.full((n_inst, 9), np.nan)
    sqrt_weights = np.arange(1, 46, dtype=float) ** 0.5
    sqrt_weights /= sqrt_weights.sum()

    for i in range(n_inst):
        close = prices_window[i]
        logp  = np.log(close)

        # MA
        ma_s = pd.Series(logp).rolling(5).mean().iloc[idx]
        ma_l = pd.Series(logp).rolling(70).mean().iloc[idx]
        ma_reg = 0 if ma_l > ma_s else 2

        # EMA
        ema_s = pd.Series(logp).ewm(span=5,  adjust=False).mean().iloc[idx]
        ema_l = pd.Series(logp).ewm(span=50, adjust=False).mean().iloc[idx]
        ema_reg = 2 if ema_s > ema_l else 0

        # slope/vol
        sv_reg = _slope_vol_reg(close, idx)

        # MACD
        macd_line   = pd.Series(logp).ewm(50, adjust=False).mean() \
                    - pd.Series(logp).ewm(90, adjust=False).mean()
        signal_line = macd_line.ewm(span=40, adjust=False).mean()
        macd_reg    = 2 if macd_line.iloc[idx] > signal_line.iloc[idx] else 0

        # Kalman
        proc_var, meas_var = 0.01, 10.0
        x_est = np.zeros(win_len); P = np.zeros(win_len)
        x_est[0], P[0] = logp[0], 1.0
        for t in range(1, win_len):
            P_pred = P[t-1] + proc_var
            K      = P_pred / (P_pred + meas_var)
            x_est[t] = x_est[t-1] + K*(logp[t] - x_est[t-1])
            P[t]     = (1-K)*P_pred
        kalman_reg = 2 if logp[idx] > x_est[idx] else 0

        # Fibonacci
        if idx >= 50:
            win50 = close[idx-49:idx+1]
            hi, lo = win50.max(), win50.min()
            rng = hi - lo
            upper, lower = lo + 0.786*rng, lo + 0.618*rng
            fib_reg = 2 if close[idx] > upper else 0 if close[idx] < lower else 1
        else:
            fib_reg = np.nan

        # PSAR
        psar = np.empty(win_len)
        trend_up, af, max_af = True, 0.01, 0.10
        ep = close[0]; psar[0] = close[0]
        for t in range(1, win_len):
            psar[t] = psar[t-1] + af*(ep - psar[t-1])
            if trend_up:
                if close[t] < psar[t]:
                    trend_up, psar[t], ep, af = False, ep, close[t], 0.01
                elif close[t] > ep:
                    ep, af = close[t], min(af+0.01, max_af)
            else:
                if close[t] > psar[t]:
                    trend_up, psar[t], ep, af = True, ep, close[t], 0.01
                elif close[t] < ep:
                    ep, af = close[t], min(af+0.01, max_af)
        psar_reg = 2 if close[idx] > psar[idx] else 0

        # Z-score
        ma90 = pd.Series(close).rolling(90).mean().iloc[idx]
        sd90 = pd.Series(close).rolling(90).std().iloc[idx]
        if np.isnan(ma90) or np.isnan(sd90):
            zscore_reg = np.nan
        else:
            z = (close[idx] - ma90) / sd90
            zscore_reg = 2 if z > 0.5 else 0 if z < -0.5 else 1

        # Weighted-return
        if idx >= 45:
            r = pd.Series(close).pct_change().iloc[idx-44:idx+1].values
            wr = np.dot(r, sqrt_weights)
            wret_reg = 2 if wr > 0 else 0 if wr < 0 else 1
        else:
            wret_reg = np.nan

        out[i] = [
            ma_reg, ema_reg, sv_reg, macd_reg, kalman_reg,
            fib_reg, psar_reg, zscore_reg, wret_reg,
        ]

    return out

def infer_from_array(prices: np.ndarray,
                     timestep: int,
                     win_len:   int = 100) -> np.ndarray:
    """
    prices:  (n_inst, T)
    timestep: 0 <= t < T
    win_len: how many bars to include
    returns: (n_inst, 9)
    """
    n_inst, T = prices.shape
    if timestep < win_len-1:
        raise ValueError(f"need at least {win_len} bars, got {timestep+1}")
    window = prices[:, (timestep-win_len+1):(timestep+1)]
    return compute_regime_features_window(window)
