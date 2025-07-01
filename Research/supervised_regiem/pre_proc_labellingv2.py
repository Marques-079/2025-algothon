import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def plot_all_regimes_percent(end_point, plot_graph=True):
    # read all 50 columns once
    price_df = pd.read_csv("../prices.txt", sep=r"\s+", header=None).iloc[:end_point]
    n_instruments = price_df.shape[1]

    for inst in range(n_instruments):
        df = price_df[[inst]].rename(columns={inst: "price"}).copy()

        # ----- your existing parameters -----
        N = 8
        K =28
        window = 2
        slope_lower, slope_upper = -2e-5, 2e-5

    # ─── SLOPE‐BASED REGIME DISTRIBUTION ────────────────────────────────

        bear_cnt    = 0
        neutral_cnt = 0
        bull_cnt    = 0
        total       = 0

        prices_arr = price_df[inst].values

        for t in range(0, end_point - window):
            y_raw = prices_arr[t : t+window]
            y = (y_raw / y_raw[0]) - 1

            Xmat = np.arange(window).reshape(-1,1)
            lr = LinearRegression().fit(Xmat, y)
            slope = lr.coef_[0]

            if slope < slope_lower:
                bear_cnt += 1
                #print(f'{t} slope: {slope:.6f} and classed: Bear')
            elif slope > slope_upper:
                bull_cnt += 1
                #print(f'{t} slope: {slope:.6f} and classed: Bull')
            else:
                neutral_cnt += 1
                #print(f'{t} slope: {slope:.6f} and classed: Neutral')
            total += 1

        fractions = {
            'bear':    bear_cnt    / total,
            'neutral': neutral_cnt / total,
            'bull':    bull_cnt    / total
        }
        #print("Slope‐based regime fractions:", fractions)

        down_q = fractions['bear']
        up_q = fractions["bear"] + fractions["neutral"]
        #print(f"Updated down_q: {down_q}, up_q: {up_q}")
        #down_q = up_q = 0.5

        #------------------------------------------------------------------------------------------------------------------------#
        '''
        df["price_fwd"]  = df["price"].shift(-N)
        df["fwd_ret"]    = df["price_fwd"] / df["price"] - 1
        df = df.iloc[:-N]  # drop the last N rows with no forward return
        '''

        df["avg_price_fwd"] = (
            df["price"]
            .rolling(window=N, min_periods=N)
            .mean()
            .shift(-N)
        )
        df["fwd_ret"] = df["avg_price_fwd"]/df["price"] - 1
        df = df.iloc[:-N]  

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
            n   = len(arr)
            i   = 0

            while i < n:
                j = i + 1
                while j < n and arr[j] == arr[i]:
                    j += 1
                run_len = j - i

                if 0 < run_len < L:
                    left_val, right_val = None, None
                    left_run, right_run = 0, 0

                    if i > 0:
                        left_val = arr[i - 1]
                        k = i - 1
                        while k >= 0 and arr[k] == left_val:
                            left_run += 1
                            k -= 1

                    if j < n:
                        right_val = arr[j]
                        k = j
                        while k < n and arr[k] == right_val:
                            right_run += 1
                            k += 1

                    if left_run >= right_run and left_val is not None:
                        fill = left_val
                    else:
                        fill = right_val

                    arr[i:j] = fill
                i = j
            return pd.Series(arr, index=regimes.index)


        df["regime_final"] = enforce_min_run(df["regime_smooth"], K)


        if plot_graph:
            fig, ax = plt.subplots(figsize=(12,4))
            ax.plot(df.index, df["price"], "k-", lw=1, label="Close Price")

            color_map = {0:"red", 1:"lightgrey", 2:"green"}
            df["segment"] = (df["regime_final"] != df["regime_final"].shift()).cumsum()

            for _, seg in df.groupby("segment"):
                reg = int(seg["regime_final"].iat[0])
                ax.axvspan(seg.index[0],
                           seg.index[-1],
                           color=color_map[reg],
                           alpha=0.3,
                           lw=0)

            ax.set_title(f"Instrument {inst} regimes (percentage %)")
            ax.set_xlabel("Time (Days)")
            ax.set_ylabel("Price")
            ax.legend()
            plt.tight_layout()
            plt.show()     # <-- ensures each fig renders in Jupyter

    return  # nothing, or you could return a dict of arrays if you like


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def plot_all_regimes_logs(end_point, plot_graph=True):
    # read all 50 columns once
    price_df = pd.read_csv("../prices.txt", sep=r"\s+", header=None).iloc[:end_point]
    n_instruments = price_df.shape[1]

    for inst in range(n_instruments):
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

    # end for inst
    return






