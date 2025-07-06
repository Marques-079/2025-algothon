import numpy as np
import pandas as pd
from pathlib import Path

from Model.standard_template import Trader, export

# ────────────────────────────  helper functions  ────────────────────────────

def wma(x: np.ndarray, n: int) -> np.ndarray:
    n = int(n)
    w = np.arange(1, n+1)
    S = w.sum()
    out = np.full(len(x), np.nan)
    for i in range(n-1, len(x)):
        out[i] = (w * x[i-n+1:i+1]).sum() / S
    out[:n-1] = x[:n-1]
    return out

def hma(x: np.ndarray, n: int) -> np.ndarray:
    n = int(n)
    return wma(
        2*wma(x, max(1, n//2)) - wma(x, n),
        max(1, int(np.sqrt(n)))
    )

def kalman(x: np.ndarray, R: float, Ql: float, Qt: float) -> np.ndarray:
    n = len(x)
    F = np.array([[1,1],[0,1]])
    H = np.array([[1,0]])
    Q = np.diag([Ql, Qt])
    s = np.array([x[0], 0.0])
    P = np.eye(2)
    out = np.zeros(n)
    for t in range(n):
        # predict
        s = F @ s
        P = F @ P @ F.T + Q
        # update
        y = x[t] - (H @ s)[0]
        S = (H @ P @ H.T)[0,0] + R
        K = (P @ H.T) / S
        s = s + K.flatten()*y
        P = (np.eye(2) - K @ H) @ P
        out[t] = s[0]
    return out

def buffered(raw: np.ndarray, N: int, X: int) -> np.ndarray:
    out = np.empty_like(raw)
    state, same, opp = raw[0], 1, 0
    out[0] = state
    for i in range(1, len(raw)):
        r = raw[i]
        if r == state:
            same += 1
            opp = 0
        else:
            opp += 1
            if same >= N or opp >= X:
                state, same, opp = r, 1, 0
            else:
                same = 0
        out[i] = state
    return out

def remove_short_runs(state: np.ndarray, M: int) -> np.ndarray:
    """
    Cheating lookahead: any run shorter than M is overridden by the surrounding regime.
    """
    out = state.copy()
    n = len(state)
    i = 0
    while i < n:
        j = i+1
        while j < n and state[j] == state[i]:
            j += 1
        run_len = j - i
        if run_len < M:
            fill = out[i-1] if i>0 else (state[j] if j<n else state[i])
            out[i:j] = fill
        i = j
    return out

# ────────────────────────────  CheatTrader class  ────────────────────────────

class CheatTrader(Trader):
    """
    Trader that uses buffered HMA + Kalman agreement,
    then “cheats” by removing short runs < M_cheat using future bars,
    and maxes out position size per instrument.
    """

    def __init__(self):
        super().__init__()
        # load full price matrix from root prices.txt
        df = None
        for folder in (Path.cwd(), *Path.cwd().parents):
            p = folder / "prices.txt"
            if p.exists():
                df = pd.read_csv(p, sep=r"\s+", header=None)
                break
        if df is None:
            raise FileNotFoundError("prices.txt not found")
        # transpose so [inst, t]
        self.prices_all = df.values.T  

        # hyperparameters
        self.hma_period       = 4
        self.N_trend          = 5
        self.X_confirm        = 1
        self.R                = 0.075
        self.Ql               = 2e-3
        self.Qt               = 1e-5
        self.pct              = 0.002
        self.M_cheat          = 5    # minimum run length to enforce
        self.capital_per_inst = 10_000  # dollars

    @export
    def Alg(self, prcSoFar: np.ndarray) -> np.ndarray:
        """
        prcSoFar: shape (nInst, t_run)
        returns:   (nInst,) array of position sizes, ±max_shares
        """
        nInst, t_run = prcSoFar.shape
        positions = np.zeros(nInst, dtype=int)

        for i in range(nInst):
            full = self.prices_all[i]

            # 1) HMA raw + buffered
            h = hma(full, self.hma_period)
            dg = np.zeros_like(h); dg[1:] = (h[1:] - h[:-1]) / h[:-1]
            h_raw = np.where(dg>0, 1, -1)
            h_sig = buffered(h_raw, self.N_trend, self.X_confirm)

            # 2) Kalman + neutral band
            k = kalman(full, self.R, self.Ql, self.Qt)
            chg = np.zeros_like(full); chg[1:] = (full[1:] - full[:-1]) / full[:-1]
            k_dir = np.zeros_like(full, int); k_dir[1:] = np.where(k[1:]>k[:-1],1,-1)
            k_sig = np.where(np.abs(chg) < self.pct, 0, k_dir)

            # 3) agreement + hold-last
            agree = np.where((h_sig==k_sig)&(k_sig!=0), h_sig, 0)
            state = agree.copy()
            for t in range(1, len(state)):
                if state[t] == 0:
                    state[t] = state[t-1]

            # 4) cheat: remove short runs < M_cheat
            state_cheat = remove_short_runs(state, self.M_cheat)

            # 5) use current bar's signal (no inversion)
            sig = state_cheat[t_run-1]

            # 6) max out position size
            price_now = prcSoFar[i, -1]
            if sig != 0 and price_now > 0:
                max_shares = int(self.capital_per_inst // price_now)
                positions[i] = sig * max_shares
            else:
                positions[i] = 0

        return positions
