import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
        

def plot_all_regimes_short(end_point, plot_graph = True, inst = None):

    price_df = pd.read_csv("prices.txt", sep=r"\s+", header=None).iloc[:end_point]
    
    if inst is None:
        instruments = range(price_df.shape[1])
    else:
        instruments = [inst]
    
    for inst in instruments:
        df = price_df[[inst]].rename(columns={inst: "price"}).copy()

        N = 10
        K = 5
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
            plt.show()    

    return df["regime_final"].values

