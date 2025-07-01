# ─── build_feature_matrix_rolling.py ─────────────────────────────────────
import numpy as np
import pandas as pd
from precision_labeller import plot_all_regimes_long

# ── helper reused from earlier code (unchanged) ──────────────────────────
def compute_slope_vol(prices: pd.Series,
                      slope_win: int = 30,
                      vol_win: int   = 100) -> pd.DataFrame:
    logp = np.log(prices)
    t    = np.arange(slope_win)
    X    = np.vstack([t, np.ones_like(t)]).T

    def _ols(y):
        m, _ = np.linalg.lstsq(X, y, rcond=None)[0]
        return m

    slope = (
        logp.rolling(window=slope_win, min_periods=slope_win)
            .apply(_ols, raw=True)
    )
    rtn = logp.diff()
    vol = rtn.rolling(window=vol_win, min_periods=vol_win).std()
    return pd.DataFrame({"slope": slope, "vol": vol})


# ───────────────────────── main function ────────────────────────────────
def build_feature_matrix(price_file: str,
                         drop_last: int = 10,
                         output_csv: str = "features_all_modelsFINAL.csv"):
    """
    Produce the training feature matrix (one row per timestep × instrument).

    * `slope_vol_reg` uses a **100-bar rolling median** of volatility
      (causal; no leakage).
    * `ma_reg` mapping is flipped → bull = 2, bear = 0.
    """
    df_raw = pd.read_csv(price_file, sep=r"\s+", header=None)
    n_rows, n_inst = df_raw.shape

    rows = []

    for inst in range(n_inst):
        close = df_raw.iloc[:, inst]
        logp  = np.log(close)
        T     = len(close)

        # ground-truth regimes (trim last drop_last bars)
        true_reg = plot_all_regimes_long(end_point=T, plot_graph=False, inst=inst)
        true_reg = pd.Series(true_reg[: T - drop_last], name="true")

        feats = pd.DataFrame(index=true_reg.index)

        # ── MA regime (FLIPPED) ──────────────────────────────────────────
        ma_s = logp.rolling(5,  min_periods=1).mean()
        ma_l = logp.rolling(70, min_periods=1).mean()
        feats["ma_reg"] = np.where(ma_l > ma_s, 0, 2)[: T - drop_last]

        # ── EMA regime (unchanged) ──────────────────────────────────────
        ema_s = logp.ewm(span=5,  adjust=False).mean()
        ema_l = logp.ewm(span=50, adjust=False).mean()
        feats["ema_reg"] = np.where(ema_s > ema_l, 2, 0)[: T - drop_last]

        # ── Slope / Vol regime  (rolling-median threshold) ──────────────
        sv_df = compute_slope_vol(close).dropna()
        idx   = sv_df.index[sv_df.index < T - drop_last]

        slope = sv_df.loc[idx, "slope"]
        vol   = sv_df.loc[idx, "vol"]

        roll_median = (
            vol.rolling(window=100, min_periods=100)
               .median()
        )

        feats.loc[idx, "slope_vol_reg"] = np.where(
            (slope > 0) & (vol < roll_median.loc[idx]), 2, 0
        )

        # ── MACD regime ─────────────────────────────────────────────────
        ema_s2   = logp.ewm(span=50, adjust=False).mean()
        ema_l2   = logp.ewm(span=90, adjust=False).mean()
        macd     = ema_s2 - ema_l2
        signal   = macd.ewm(span=40, adjust=False).mean()
        feats["macd_reg"] = np.where(
            macd[: T - drop_last] > signal[: T - drop_last], 2, 0
        )

        # ── Kalman regime ───────────────────────────────────────────────
        x_est = np.zeros(T)
        P     = np.zeros(T)
        x_est[0], P[0] = logp.iloc[0], 1.0
        for t in range(1, T):
            x_pred = x_est[t - 1]
            P_pred = P[t - 1] + 0.01
            K      = P_pred / (P_pred + 10.0)
            x_est[t] = x_pred + K * (logp.iloc[t] - x_pred)
            P[t]     = (1 - K) * P_pred
        feats["kalman_reg"] = np.where(
            logp[: T - drop_last] > x_est[: T - drop_last], 2, 0
        )

        # ── Fibonacci regime ────────────────────────────────────────────
        hi50 = close.rolling(50, min_periods=50).max()
        lo50 = close.rolling(50, min_periods=50).min()
        rng  = hi50 - lo50
        upper = lo50 + 0.786 * rng
        lower = lo50 + 0.618 * rng
        feats["fib_reg"] = np.where(
            close[: T - drop_last] > upper[: T - drop_last], 2,
            np.where(close[: T - drop_last] < lower[: T - drop_last], 0, 1)
        )

        # ── PSAR regime (same simplification as before) ─────────────────
        psar = np.zeros(T)
        trend_up, af, max_af = True, 0.01, 0.10
        ep = hi50.iloc[0] if not np.isnan(hi50.iloc[0]) else close.iloc[0]
        psar[0] = lo50.iloc[0] if trend_up else hi50.iloc[0]
        for t in range(1, T):
            psar[t] = psar[t - 1] + af * (ep - psar[t - 1])
            if trend_up:
                if close.iloc[t] < psar[t]:
                    trend_up, psar[t], ep, af = False, ep, close.iloc[t], 0.01
                elif close.iloc[t] > ep:
                    ep, af = close.iloc[t], min(af + 0.01, max_af)
            else:
                if close.iloc[t] > psar[t]:
                    trend_up, psar[t], ep, af = True, ep, close.iloc[t], 0.01
                elif close.iloc[t] < ep:
                    ep, af = close.iloc[t], min(af + 0.01, max_af)
        feats["psar_reg"] = np.where(
            close[: T - drop_last] > psar[: T - drop_last], 2, 0
        )

        # ── Z-score regime ──────────────────────────────────────────────
        ma90 = close.rolling(90, min_periods=90).mean()
        sd90 = close.rolling(90, min_periods=90).std()
        z    = (close - ma90) / sd90
        feats["zscore_reg"] = np.where(
            z[: T - drop_last] > 0.5, 2,
            np.where(z[: T - drop_last] < -0.5, 0, 1)
        )

        # ── Weighted-return regime ──────────────────────────────────────
        r   = close.pct_change()
        w   = (np.arange(1, 46) ** 0.5)
        w  /= w.sum()
        wr = r.rolling(45, min_periods=45).apply(lambda x: np.dot(x, w), raw=True)
        feats["wret_reg"] = np.where(
            wr[: T - drop_last] > 0, 2,
            np.where(wr[: T - drop_last] < 0, 0, 1)
        )

        # metadata & align
        feats["true_regime"] = true_reg.values
        feats["inst"]        = inst
        feats["time"]        = feats.index

        rows.append(feats.reset_index(drop=True))

    final_df = pd.concat(rows, ignore_index=True)
    final_df.to_csv(output_csv, index=False)
    print(f"✅ Features written to {output_csv}")


# ——— run once —————————————————————————————————————————————
if __name__ == "__main__":
    build_feature_matrix("prices.txt",
                         drop_last=10,
                         output_csv="features_all_modelsFINAL.csv")
