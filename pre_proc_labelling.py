#TODO need to tune hyperparamters so model can fit well to all instruments -Marcus

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#-------------HYPER PARAMTERS----------------#
inst = 6                # instrument index to label
end_point = 750         # how much of data we want to label left -> right

N = 10                           # forward‐return window
down_q = 0.60                    # I recommend 0.4
up_q = 0.70                     # I recommend 0.6
K = 10                           # if a trend identfied is under K days then it will merge it with a neighbouring trend 

price_df = pd.read_csv("prices.txt", sep=r"\s+", header=None).iloc[:end_point]
df = price_df[[inst]].rename(columns={inst: "price"}).copy()
'''
All tunable parameters:
Lets say (down, up) = (0.4, 0.6) then we would be taking 40% bear 20% netural and 40% bull proprtions of our data.
Id would be saying [0, -> 0.4] is bear, [0.4, 0.6] is neutral and [0.6, 1] is bull where 1 is full data.
'''

# ─── SLOPE‐BASED REGIME DISTRIBUTION ────────────────────────────────
'''
So my thoughts with this one was that quartiles seperate on proportion which means we have to
dynamically adjust for whether a instrument is full bull or bear aka. outliers. 
Essentially this samples 'window' amount of points ahead and plots a linear regression, depending on grad it will accumulate 
the amount of pos, netural and neg slopes classified by the hyperparams slope_lower, slope_upper. 
Thus we can adapt the quartiles based on the rough amount of ups and downs in the instrument.
'''
slope_lower = -0.005   # gradient threshold below which we call “bear”
slope_upper =  0.005   # gradient threshold above which we call “bull”
window      = 2      # same as your forward‐return window 

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
        print(f'{t} slope: {slope:.6f} and classed: Bear')
    elif slope > slope_upper:
        bull_cnt += 1
        print(f'{t} slope: {slope:.6f} and classed: Bull')
    else:
        neutral_cnt += 1
        print(f'{t} slope: {slope:.6f} and classed: Neutral')
    total += 1

fractions = {
    'bear':    bear_cnt    / total,
    'neutral': neutral_cnt / total,
    'bull':    bull_cnt    / total
}
print("Slope‐based regime fractions:", fractions)

#utilise fractions to form quartiles
down_q = fractions['bear']
up_q = fractions["bear"] + fractions["neutral"]
print(f"Updated down_q: {down_q}, up_q: {up_q}")

#------------------------------------------------------------------------------------------------------------------------#

df["price_fwd"]  = df["price"].shift(-N)
df["fwd_ret"]    = df["price_fwd"] / df["price"] - 1
df = df.iloc[:-N]  # drop the last N rows with no forward return

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
def enforce_min_run(s: pd.Series, K: int = 21) -> pd.Series:
    '''
    Groups trends moving through the array and if trend len < K it will merge the trend into the 
    neighbouring trend group which is larger 
    '''
    arr = s.to_numpy().copy()
    N = len(arr)
    i = 0
    while i < N:
        # find end of this contiguous run
        j = i + 1
        while j < N and arr[j] == arr[i]:
            j += 1
        run_len = j - i
        if 0 < run_len < K:
            # pick neighbor with longer context
            left_len  = i
            right_len = N - j
            if left_len >= right_len and i > 0:
                tgt = arr[i-1]
            elif j < N:
                tgt = arr[j]
            else:
                tgt = arr[i-1]
            arr[i:j] = tgt
        i = j
    return pd.Series(arr, index=s.index)

df["regime_final"] = enforce_min_run(df["regime_smooth"], K)

#---------------------Plotting-----------------#
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
