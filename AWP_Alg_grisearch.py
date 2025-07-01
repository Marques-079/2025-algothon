# grid_search_smoothing.py

import numpy as np
import pandas as pd
from itertools import product
import eval     # make sure eval.py is on your PYTHONPATH
from eval import loadPrices, calcPL

# ─── 1) Hyperparameters & Data ────────────────────────────────────────────────
PRICES_FILE   = "prices.txt"
NUM_TEST_DAYS = 500
NOTIONAL      = 10_000

SHORT_W = 20
LONG_W  = 50
MOM_W   = 10

prcAll = loadPrices(PRICES_FILE)   # shape (nInst, nt)

# ─── 2) Smoothers ──────────────────────────────────────────────────────────────
def confirm_filter(raw, min_consec):
    sm = np.zeros_like(raw)
    sm[0] = raw[0]
    consec = 0
    for t in range(1, len(raw)):
        if raw[t] == sm[t-1]:
            consec = 0
            sm[t] = sm[t-1]
        else:
            consec += 1
            if consec >= min_consec:
                sm[t] = raw[t]
                consec = 0
            else:
                sm[t] = sm[t-1]
    return sm

from collections import Counter
def mode_filter(raw, window):
    sm = np.zeros_like(raw)
    for t in range(len(raw)):
        w0 = max(0, t - window + 1)
        sm[t] = Counter(raw[w0:t+1]).most_common(1)[0][0]
    return sm

def ewma_hysteresis(raw, alpha, thresh):
    sm = np.zeros_like(raw)
    s  = raw[0]
    sm[0] = raw[0]
    for t in range(1, len(raw)):
        s = alpha * raw[t] + (1 - alpha) * s
        if s >= +thresh:
            sm[t] = +1
        elif s <= -thresh:
            sm[t] = -1
        else:
            sm[t] = sm[t-1]
    return sm

# ─── 3) Raw signal builder ─────────────────────────────────────────────────────
import pandas as _pd
def raw_regime_signal(prcSoFar):
    nInst, t = prcSoFar.shape
    df = _pd.DataFrame(prcSoFar.T, columns=range(nInst))
    series_list = []
    for i in range(nInst):
        s      = df[i]
        ma_s   = s.rolling(SHORT_W, min_periods=1).mean()
        ma_l   = s.rolling(LONG_W,  min_periods=1).mean()
        mom    = s.diff(MOM_W).fillna(0)
        combo  = np.where(np.where(ma_l>ma_s,0,2)==np.where(mom>0,2,0),
                          np.where(ma_l>ma_s,0,2),
                          np.where(mom>0,2,0))
        series_list.append(np.where(combo==2, +1, -1))
    return series_list

# ─── 4) Factory for getPosition ────────────────────────────────────────────────
def make_getPosition(smoothing, param):
    def getPosition(prcSoFar):
        nInst, t = prcSoFar.shape
        if t < LONG_W:
            return np.zeros(nInst, dtype=int)

        raw_list = raw_regime_signal(prcSoFar)
        signal   = np.zeros(nInst, dtype=int)

        for i, raw in enumerate(raw_list):
            if smoothing == "confirm":
                sm = confirm_filter(raw, param)
            elif smoothing == "mode":
                sm = mode_filter(raw,   param)
            else:
                alpha, thresh = param
                sm = ewma_hysteresis(raw, alpha, thresh)

            signal[i] = sm[-1]

        last_price = prcSoFar[:, -1]
        max_shares = np.floor(NOTIONAL / last_price).astype(int)
        return signal * max_shares

    return getPosition

# ─── 5) Grid‐search ────────────────────────────────────────────────────────────
results = []
total = 4 + 4 + 3*3
count = 0

# 5a) consecutive‐confirmation
for min_consec in [1,2,3,5]:
    count += 1
    print(f"[{count}/{total}] Testing confirm_filter(min_consec={min_consec})")
    gp = make_getPosition("confirm", min_consec)
    eval.getPosition = gp
    meanpl, ret, plstd, sharpe, dvol = calcPL(prcAll, NUM_TEST_DAYS)
    score = meanpl - 0.1*plstd
    results.append({
        "method":"confirm","param":min_consec,
        "meanPL":meanpl,"stdPL":plstd,"Sharpe":sharpe,"score":score
    })

# 5b) rolling‐mode
for window in [3,5,7,10]:
    count += 1
    print(f"[{count}/{total}] Testing mode_filter(window={window})")
    gp = make_getPosition("mode", window)
    eval.getPosition = gp
    meanpl, ret, plstd, sharpe, dvol = calcPL(prcAll, NUM_TEST_DAYS)
    score = meanpl - 0.1*plstd
    results.append({
        "method":"mode","param":window,
        "meanPL":meanpl,"stdPL":plstd,"Sharpe":sharpe,"score":score
    })

# 5c) EWMA + hysteresis
for alpha, thresh in product([0.1,0.2,0.3],[0.0,0.5,1.0]):
    count += 1
    print(f"[{count}/{total}] Testing ewma(alpha={alpha},thresh={thresh})")
    gp = make_getPosition("ewma", (alpha,thresh))
    eval.getPosition = gp
    meanpl, ret, plstd, sharpe, dvol = calcPL(prcAll, NUM_TEST_DAYS)
    score = meanpl - 0.1*plstd
    results.append({
        "method":"ewma","param":f"α={alpha},θ={thresh}",
        "meanPL":meanpl,"stdPL":plstd,"Sharpe":sharpe,"score":score
    })

# ─── 6) Summarize ──────────────────────────────────────────────────────────────
df = pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)

print("\n=== Top 10 smoothing configurations ===")
print(df.head(10)[["method","param","meanPL","stdPL","Sharpe","score"]].to_string(index=False))
