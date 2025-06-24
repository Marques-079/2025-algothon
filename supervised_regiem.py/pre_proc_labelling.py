import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

'''
NB: Feel free to change the hyperparameters around and see what works 
Presets are for a longer regiem based autolabeller
'''

presets = {
    #[Instance, Endpoint, N_steps forward, K_min_trend_length, window, slope_lower, slope_upper]

    0 : [5, 750, 10, 27, 6, -0.015, 0.015], #Works well on Range < 12 
    1 : [7, 750, 10, 10, 2, -0.055, 0.055], #Work well on Range > 12 
    2 : [9, 750, 10, 47, 2, -0.055, 0.055], #Work well on Range > 25

}

def get_data(preset_num, instrument, endpoint, plot_graph):
    hypers = presets[preset_num]  # Change the index to select different presets based on RANGE 

    #-------------HYPER PARAMTERS----------------#
    inst = instrument                # instrument index to label
    end_point = endpoint         # how much of data we want to label left -> right

    N = hypers[2]                         # forward‐return window. At N steps forward take the change                  
    K = hypers[3]                         # "What can the smallest trend be in length" Undersized trend merge threshold
    # I advise against a K over 35 as the merging may become too loose which leads to false labelling

    price_df = pd.read_csv("prices.txt", sep=r"\s+", header=None).iloc[:end_point]
    df = price_df[[inst]].rename(columns={inst: "price"}).copy()
    '''
    All tunable parameters:
    Lets say (down, up) = (0.4, 0.6) then we would be taking 40% bear 20% netural and 40% bull proprtions of our data.
    Id would be saying [0, -> 0.4] is bear, [0.4, 0.6] is neutral and [0.6, 1] is bull where 1 is full data.

    So my thoughts with this one was that quartiles seperate on proportion which means we have to
    dynamically adjust for whether a instrument is full bull or bear aka. outliers. 
    Essentially this samples 'window' amount of points ahead and plots a linear regression, depending on grad it will accumulate 
    the amount of pos, netural and neg slopes classified by the hyperparams slope_lower, slope_upper. 
    Thus we can adapt the quartiles based on the rough amount of ups and downs in the instrument.
    '''

    slope_lower = hypers[5]   # gradient threshold below which we call “bear”
    slope_upper = hypers[6]   # gradient threshold above which we call “bull”
    window      = hypers[4]      # same as your forward‐return window 

    # ─── SLOPE‐BASED REGIME DISTRIBUTION ────────────────────────────────

    bear_cnt    = 0
    neutral_cnt = 0
    bull_cnt    = 0
    total       = 0

    prices_arr = price_df[inst].values

    for t in range(0, end_point - window):
        y = prices_arr[t : t+window]
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

    #utilise fractions to form quartiles
    down_q = fractions['bear']
    up_q = fractions["bear"] + fractions["neutral"]
    #print(f"Updated down_q: {down_q}, up_q: {up_q}")

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
        """
        Merge any contiguous “run” shorter than L into the longer of its two neighbors.
        """
        arr = regimes.to_numpy().copy()
        n   = len(arr)
        i   = 0

        while i < n:
            # 1) find end of this run [i:j)
            j = i + 1
            while j < n and arr[j] == arr[i]:
                j += 1
            run_len = j - i

            # 2) if too short, inspect neighbors
            if 0 < run_len < L:
                left_val, right_val = None, None
                left_run, right_run = 0, 0

                # look left
                if i > 0:
                    left_val = arr[i - 1]
                    k = i - 1
                    while k >= 0 and arr[k] == left_val:
                        left_run += 1
                        k -= 1

                # look right
                if j < n:
                    right_val = arr[j]
                    k = j
                    while k < n and arr[k] == right_val:
                        right_run += 1
                        k += 1

                # pick the longer neighbor
                if left_run >= right_run and left_val is not None:
                    fill = left_val
                else:
                    fill = right_val

                arr[i:j] = fill

            # 3) jump to the next run
            i = j

        return pd.Series(arr, index=regimes.index)


    df["regime_final"] = enforce_min_run(df["regime_smooth"], K)

    #print(df["regime_final"].value_counts())

    if plot_graph:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df.index, df["price"], color="k", lw=1, label="Close Price")

        color_map = {0: "red", 1: "lightgrey", 2: "green"}
        df["segment"] = (df["regime_final"] != df["regime_final"].shift()).cumsum()

        for _, seg in df.groupby("segment"):
            reg = seg["regime_final"].iat[0]
            ax.axvspan(seg.index[0],
                    seg.index[-1],
                    color=color_map[reg],
                    alpha=0.3,
                    lw=0)

        ax.set_title("Smoothed Market Regimes (0=bear, 1=neutral, 2=bull)")
        ax.set_xlabel("Time (Days)")
        ax.set_ylabel("Price")
        ax.legend(loc="upper left")
        plt.tight_layout()
        plt.show()
    

    return df["regime_final"].to_numpy()

