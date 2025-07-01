"""
feature_pipelines.py  (Option B: *causal* 100-bar rolling median
for the slope/volatility regime)

This file contains:

• a library of “evaluate_*_accuracy” helpers for quick back-tests
• build_feature_matrix(…) — writes one row per timestep × instrument
  with all nine regime signals **aligned** to the online inference
  logic used in regime_inference.py.
"""

# ────────────────────────────────────────────────────────────────────────
# Imports
# ────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from precision_labeller import plot_all_regimes_long    # adjust import if needed

# ────────────────────────────────────────────────────────────────────────
# 1.  Accuracy helpers
# (unchanged — skip to “build_feature_matrix” to see the Option B edit)
# ────────────────────────────────────────────────────────────────────────

def evaluate_ma_accuracy(price_file: str,
                         short_w: int = 5,
                         long_w: int = 70) -> float:
    df = pd.read_csv(price_file, sep=r"\s+", header=None)
    prices = df.iloc[:, 0]
    T = len(prices)

    true_regs = plot_all_regimes_long(end_point=T, plot_graph=False, inst=0)

    logp  = np.log(prices)
    ma_s  = logp.rolling(window=short_w, min_periods=1).mean()
    ma_l  = logp.rolling(window=long_w,  min_periods=1).mean()
    preds = np.where(ma_l > ma_s, 0, 2)[: len(true_regs)]

    return accuracy_score(true_regs, preds)


def evaluate_ema_accuracy(price_file: str,
                          short_span: int = 5,
                          long_span:  int = 50,
                          inst: int   = 0) -> float:
    df = pd.read_csv(price_file, sep=r"\s+", header=None)
    prices = df.iloc[:, inst]
    T = len(prices)

    true_regs = plot_all_regimes_long(end_point=T, plot_graph=False, inst=inst)

    logp  = np.log(prices)
    ema_s = logp.ewm(span=short_span, adjust=False).mean()
    ema_l = logp.ewm(span=long_span,  adjust=False).mean()
    preds = np.where(ema_s > ema_l, 2, 0)[: len(true_regs)]

    return accuracy_score(true_regs, preds)


# ─── slope/vol helper — re-used in both evaluation & feature builder ───
def compute_slope_vol(prices: pd.Series,
                      slope_win: int,
                      vol_win:   int) -> pd.DataFrame:
    """
    Returns a DataFrame with two columns: 'slope' and 'vol', aligned to
    the original price index.  NaNs appear until each window is ready.
    """
    logp = np.log(prices)

    # ordinary-least-squares slope over `slope_win`
    t = np.arange(slope_win)
    X = np.vstack([t, np.ones_like(t)]).T

    def slope_of_window(y):
        m, _ = np.linalg.lstsq(X, y, rcond=None)[0]
        return m

    slope = (
        logp.rolling(window=slope_win, min_periods=slope_win)
            .apply(slope_of_window, raw=True)
    )

    # rolling σ over `vol_win` log returns
    rtn  = logp.diff()
    vol  = rtn.rolling(window=vol_win, min_periods=vol_win).std()

    return pd.DataFrame({"slope": slope, "vol": vol})


def evaluate_slope_vol_accuracy(price_file: str,
                                slope_win: int = 30,
                                vol_win:   int = 100,
                                inst: int  = 0) -> float:
    df = pd.read_csv(price_file, sep=r"\s+", header=None)
    prices = df.iloc[:, inst]
    T = len(prices)

    true_regs = plot_all_regimes_long(end_point=T, plot_graph=False, inst=inst)

    sv = compute_slope_vol(prices, slope_win, vol_win).dropna()
    sv = sv[sv.index < len(true_regs)]

    # ─────── Option B change: *rolling* median, causal ───────
    median_series = (
        sv["vol"]
        .rolling(window=100, min_periods=100)
        .median()
    )

    preds = np.where(
        (sv["slope"] > 0) & (sv["vol"] < median_series), 2, 0
    )

    return accuracy_score(true_regs[sv.index], preds)


# (evaluate_macd_accuracy, …, evaluate_weighted_return_accuracy
#  are identical to the versions you already had – omitted for brevity)

# ────────────────────────────────────────────────────────────────────────
# 2.  Feature-matrix builder  ➜ **Option B applied here**
# ────────────────────────────────────────────────────────────────────────

def build_feature_matrix(price_file: str,
                         drop_last: int = 10,
                         output_csv: str = "features_all_models2.csv"):
    """
    For each instrument:
      – computes nine technical-regime labels (same order as training)
      – aligns them causally so the last `drop_last` bars are ignored
      – writes a tall DataFrame to `output_csv`.
    Implements **Option B** for slope/vol regime:
      the threshold is a 100-bar *rolling* median of recent volatility,
      matching the online inference code in regime_inference.py.
    """
    df_raw = pd.read_csv(price_file, sep=r"\s+", header=None)
    n_rows, n_cols = df_raw.shape
    all_rows = []

    for inst in range(n_cols):
        close = df_raw.iloc[:, inst]
        high  = close             # placeholder (PSAR needs H/L)
        low   = close
        T     = len(close)

        # ground-truth from autolabeller
        true = plot_all_regimes_long(end_point=T, plot_graph=False, inst=inst)
        true = pd.Series(true[: T - drop_last], name="true")

        features = pd.DataFrame(index=true.index)
        logp = np.log(close)

        # — MA regime —
        ma_s = logp.rolling(5,  min_periods=1).mean()
        ma_l = logp.rolling(70, min_periods=1).mean()
        features["ma_reg"] = np.where(ma_l > ma_s, 0, 2)[: T - drop_last]

        # — EMA regime —
        ema_s = logp.ewm(span=5,  adjust=False).mean()
        ema_l = logp.ewm(span=50, adjust=False).mean()
        features["ema_reg"] = np.where(ema_s > ema_l, 2, 0)[: T - drop_last]

        # — Slope/Vol regime (★ Option B edit ★) —
        sv = compute_slope_vol(close, slope_win=30, vol_win=100).dropna()
        sv = sv[sv.index < T - drop_last]

        median_series = (
            sv["vol"]
            .rolling(window=100, min_periods=100)
            .median()
        )

        features.loc[sv.index, "slope_vol_reg"] = np.where(
            (sv["slope"] > 0) & (sv["vol"] < median_series), 2, 0
        )

        # — MACD regime —
        ema_s2 = logp.ewm(span=50, adjust=False).mean()
        ema_l2 = logp.ewm(span=90, adjust=False).mean()
        macd   = ema_s2 - ema_l2
        signal = macd.ewm(span=40, adjust=False).mean()
        features["macd_reg"] = np.where(
            macd[: T - drop_last] > signal[: T - drop_last], 2, 0
        )

        # — Kalman regime —
        x_est = np.zeros(T); P = np.zeros(T)
        x_est[0], P[0] = logp.iloc[0], 1.0
        for t in range(1, T):
            P_pred = P[t-1] + 0.01
            K = P_pred / (P_pred + 10.0)
            x_est[t] = x_est[t-1] + K * (logp.iloc[t] - x_est[t-1])
            P[t]     = (1 - K) * P_pred
        features["kalman_reg"] = np.where(
            logp[: T - drop_last] > x_est[: T - drop_last], 2, 0
        )

        # — Fibonacci regime —
        hi50 = close.rolling(50, min_periods=50).max()
        lo50 = close.rolling(50, min_periods=50).min()
        rng  = hi50 - lo50
        lower = lo50 + 0.618 * rng
        upper = lo50 + 0.786 * rng
        features["fib_reg"] = np.where(
            close[: T - drop_last] > upper[: T - drop_last], 2,
            np.where(close[: T - drop_last] < lower[: T - drop_last], 0, 1)
        )

        # — PSAR regime (close-only variant) —
        psar = np.zeros(T)
        trend_up, af, max_af = True, 0.01, 0.10
        ep = high.iloc[0]
        psar[0] = low.iloc[0]
        for t in range(1, T):
            psar[t] = psar[t-1] + af * (ep - psar[t-1])
            if trend_up:
                if low.iloc[t] < psar[t]:
                    trend_up, psar[t], ep, af = False, ep, low.iloc[t], 0.01
                elif high.iloc[t] > ep:
                    ep, af = high.iloc[t], min(af + 0.01, max_af)
            else:
                if high.iloc[t] > psar[t]:
                    trend_up, psar[t], ep, af = True, ep, high.iloc[t], 0.01
                elif low.iloc[t] < ep:
                    ep, af = low.iloc[t], min(af + 0.01, max_af)
        features["psar_reg"] = np.where(
            close[: T - drop_last] > psar[: T - drop_last], 2, 0
        )

        # — Z-score regime —
        ma90 = close.rolling(90, min_periods=90).mean()
        sd90 = close.rolling(90, min_periods=90).std()
        z = (close - ma90) / sd90
        features["zscore_reg"] = np.where(
            z[: T - drop_last] > 0.5, 2,
            np.where(z[: T - drop_last] < -0.5, 0, 1)
        )

        # — Weighted-return regime —
        r = close.pct_change()
        w = (np.arange(1, 46) ** 0.5); w /= w.sum()

        wr = r.rolling(45, min_periods=45).apply(
            lambda x: np.dot(x, w), raw=True
        )
        features["wret_reg"] = np.where(
            wr[: T - drop_last] > 0, 2,
            np.where(wr[: T - drop_last] < 0, 0, 1)
        )

        # — Book-keeping —
        features["true_regime"] = true.values
        features["inst"]        = inst
        features["time"]        = features.index

        all_rows.append(features.reset_index(drop=True))

    final = pd.concat(all_rows, ignore_index=True)
    final.to_csv(output_csv, index=False)
    print(f"✅ Features written to: {output_csv}")


# quick one-liner for the default build
if __name__ == "__main__":
    build_feature_matrix("prices.txt",
                         drop_last=10,
                         output_csv="features_all_models_FINAL.csv")
