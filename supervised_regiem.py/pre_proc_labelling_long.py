
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import numpy as np

def flip_small_flat_segments(regimes, prices, width_factor=1.5, price_tol=0.02):
    out = regimes.copy()
    changed = True

    while changed:
        changed = False
        boundaries = np.flatnonzero(out[:-1] != out[1:])
        starts     = np.concatenate(([0], boundaries + 1))
        ends       = np.concatenate((boundaries, [len(out) - 1]))

        for i, (s, e) in enumerate(zip(starts, ends)):
            seg_len = e - s + 1
            left_len  = starts[i] - starts[i - 1] if i > 0 else 0
            right_len = ends[i + 1] - ends[i]   if i < len(starts)-1 else 0

            p0, p1 = prices[s], prices[e]
            price_change = abs(p1 - p0) / abs(p0)

            # debug print
            #print(f"Segment {i}: [{s}–{e}] len={seg_len}, "
            #      f"nbrs=({left_len},{right_len}), Δ={price_change:.1%}")

            # check dominating neighbour
            candidates = []
            if left_len  > width_factor * seg_len:
                candidates.append((left_len,  out[s-1]))
            if right_len > width_factor * seg_len:
                candidates.append((right_len, out[e+1]))
            if not candidates:
                #("  → no flip\n")
                continue

            # pick the largest neighbour
            _, neighbour_val = max(candidates, key=lambda x: x[0])

            if price_change <= price_tol:
                #print(f"  → flipping → {neighbour_val}\n")
                out[s:e+1] = neighbour_val
                changed = True
            else:
                #("  → price tol fail (no flip)\n")
                pass

    return out


def plot_all_regimes_long(end_point, plot_graph=True, inst=None):
    price_df = pd.read_csv("prices.txt", sep=r"\s+", header=None).iloc[:end_point]
    # if inst is given, make it a one-element list; otherwise loop all
    instruments = [inst] if inst is not None else range(price_df.shape[1])

    for inst in instruments:
        df = price_df[[inst]].rename(columns={inst: "price"}).copy()

        # ----- your existing parameters -----
        N = 10
        K = 28
        window = 2
        slope_lower, slope_upper = -2e-5, 2e-5

        # Precompute log-price series
        df["log_price"] = np.log(df["price"])
        prices_arr = df["price"].values  # still use raw prices for slope window

        # ─── SLOPE‐BASED REGIME DISTRIBUTION (on log-returns) ────────────────────────────────
        bear_cnt = neutral_cnt = bull_cnt = total = 0
        for t in range(0, end_point - window):
            y_raw = prices_arr[t : t + window]
            # log-return from first in window
            y = np.log(y_raw) - np.log(y_raw[0])

            Xmat = np.arange(window).reshape(-1, 1)
            slope = LinearRegression().fit(Xmat, y).coef_[0]

            if slope < slope_lower:
                bear_cnt += 1
            elif slope > slope_upper:
                bull_cnt += 1
            else:
                neutral_cnt += 1
            total += 1

        fractions = {
            "bear":    bear_cnt / total,
            "neutral": neutral_cnt / total,
            "bull":    bull_cnt / total,
        }
        down_q = fractions["bear"]
        up_q   = fractions["bear"] + fractions["neutral"]

        # ─── FORWARD LOG-RETURN ─────────────────────────────────────
        # simple N-day log-return:
        df["fwd_ret"] = df["log_price"].shift(-N) - df["log_price"]
        df = df.iloc[:-N]  # drop the last N rows without forward return

        # set thresholds on log-returns
        up_th   = df["fwd_ret"].quantile(up_q)
        down_th = df["fwd_ret"].quantile(down_q)

        # 0 = bear, 1 = neutral, 2 = bull
        df["regime_raw"] = np.where(
            df["fwd_ret"] >  up_th,   2,
            np.where(df["fwd_ret"] < down_th, 0, 1)
        )

        # ─── Smooth by rolling‐mode (majority‐vote) to kill short flips ────────────
        def rolling_mode(s: pd.Series, window: int = 21) -> pd.Series:
            return (
                s.rolling(window, center=True, min_periods=1)
                 .apply(lambda x: x.value_counts().idxmax())
                 .astype(int)
            )

        df["regime_smooth"] = rolling_mode(df["regime_raw"], window=21)

        # ─── Enforce a minimum run length of K days ────────────────────
        def enforce_min_run(regimes: pd.Series, L: int) -> pd.Series:
            arr = regimes.to_numpy().copy()
            n = len(arr)
            i = 0
            while i < n:
                j = i + 1
                while j < n and arr[j] == arr[i]:
                    j += 1
                run_len = j - i
                if 0 < run_len < L:
                    # look left
                    left_val = arr[i - 1] if i > 0 else None
                    left_run = 0
                    k = i - 1
                    while k >= 0 and arr[k] == left_val:
                        left_run += 1
                        k -= 1
                    # look right
                    right_val = arr[j] if j < n else None
                    right_run = 0
                    k = j
                    while k < n and arr[k] == right_val:
                        right_run += 1
                        k += 1
                    # choose longer neighbor
                    fill = left_val if (left_val is not None and left_run >= right_run) else right_val
                    arr[i:j] = fill
                i = j
            return pd.Series(arr, index=regimes.index)

        df["regime_final"] = enforce_min_run(df["regime_smooth"], K)

        raw_flat = flip_small_flat_segments(
        df["regime_final"].values,
        df["price"].values,
        width_factor=1.5,
        price_tol=0.05
        )
        df["regime_final"] = raw_flat

        final_boundaries = np.flatnonzero(raw_flat[:-1] != raw_flat[1:])
        final_starts   = np.concatenate(([0], final_boundaries + 1))
        final_ends     = np.concatenate((final_boundaries, [len(raw_flat)-1]))

        # ─── PLOTTING ───────────────────────────────────────────────────────────────
        if plot_graph:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(df.index, df["price"], color="k", lw=1, label="Close Price")

            color_map = {0: "red", 1: "lightgrey", 2: "green"}
            df["segment"] = (df["regime_final"] != df["regime_final"].shift()).cumsum()
            for _, seg in df.groupby("segment"):
                reg = int(seg["regime_final"].iat[0])
                ax.axvspan(seg.index[0],
                           seg.index[-1],
                           color=color_map[reg],
                           alpha=0.3,
                           lw=0)

            ax.set_title(f"Instrument {inst} regimes (log‐returns)")
            ax.set_xlabel("Time (Days)")
            ax.set_ylabel("Price")
            ax.legend()
            plt.tight_layout()
            plt.show()

    return df["regime_final"].values

