from pathlib import Path
import numpy as np
import pandas as pd

from Model.standard_template import Trader, export

# ────────────────────────────  helper functions  ────────────────────────────
def wma(x: np.ndarray, n: int) -> np.ndarray:
    """Causal weighted MA with weights 1…n."""
    n = int(n)
    w = np.arange(1, n+1, dtype=float)
    S = w.sum()
    out = np.empty(len(x), dtype=float)
    out[:n-1] = x[:n-1]
    for i in range(n-1, len(x)):
        out[i] = (w * x[i-n+1:i+1]).sum() / S
    return out

def hma(x: np.ndarray, n: int) -> np.ndarray:
    """Hull moving average, causal."""
    n = int(n)
    half  = max(1, n//2)
    sqrtn = max(1, int(np.sqrt(n)))
    return wma(2*wma(x, half) - wma(x, n), sqrtn)

def kalman(x: np.ndarray, R: float, Ql: float, Qt: float) -> np.ndarray:
    """Two‐state (level+trend) causal Kalman smoother."""
    n = len(x)
    F = np.array([[1,1],[0,1]], float)
    H = np.array([[1,0]], float)
    Q = np.diag([Ql, Qt])
    s = np.array([x[0], 0.0], float)
    P = np.eye(2)
    out = np.zeros(n, float)
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
    """Buffered ±1: flip after N same or X opp in a row."""
    out   = np.empty_like(raw)
    state = raw[0]
    same  = 1
    opp   = 0
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
    """Merge any run shorter than M into its neighbor."""
    out = state.copy()
    n = len(state)
    i = 0
    while i < n:
        j = i+1
        while j<n and state[j]==state[i]:
            j += 1
        run_len = j-i
        if run_len < M:
            fill = out[i-1] if i>0 else (state[j] if j<n else state[i])
            out[i:j] = fill
        i = j
    return out

# ────────────────────────────  Trader class  ────────────────────────────
class CheatFilteredTrader(Trader):
    """
    Trader that uses:
      1) BUFFERED HMA vs KALMAN agreement,
      2) ‘cheat’ removal of short runs < M_cheat,
      3) maxes out position at capital_per_inst.
    """

    def __init__(self):
        super().__init__()

        # hyperparameters (matched from grid search & user context)
        self.hma_period       = 4
        self.N_trend          = 5
        self.X_confirm        = 1
        self.R                = 0.075
        self.Ql               = 0.004
        self.Qt               = 1e-5
        self.pct_thresh       = 0.001
        self.M_cheat          = 5
        self.capital_per_inst = 10_000

        # load full price matrix [inst, time]
        df = None
        for folder in (Path.cwd(), *Path.cwd().parents):
            p = folder / "prices.txt"
            if p.exists():
                df = pd.read_csv(p, sep=r"\s+", header=None)
                break
        if df is None:
            raise FileNotFoundError("prices.txt not found")
        self.prices_all = df.values.T  # shape: (nInst, T)

        nInst, T = self.prices_all.shape

        # precompute all signals, fully causal
        # 1) HMA series
        H = np.vstack([
            hma(self.prices_all[i], self.hma_period)
            for i in range(nInst)
        ])
        # 2) HMA-gradient raw ±1
        dH = np.zeros_like(H)
        dH[:,1:] = (H[:,1:] - H[:,:-1]) / H[:,:-1]
        Hraw = np.where(dH>0, 1, -1)
        # 3) buffered HMA signal
        Hsig = np.vstack([
            buffered(Hraw[i], self.N_trend, self.X_confirm)
            for i in range(nInst)
        ])

        # 4) Kalman series
        K = np.vstack([
            kalman(self.prices_all[i], self.R, self.Ql, self.Qt)
            for i in range(nInst)
        ])
        # 5) Kalman ±1 with neutral band
        pctchg = np.zeros_like(self.prices_all)
        pctchg[:,1:] = (self.prices_all[:,1:] - self.prices_all[:,:-1]) / self.prices_all[:,:-1]
        Kdir = np.zeros_like(self.prices_all, int)
        Kdir[:,1:] = np.where(K[:,1:]>K[:,:-1],1,-1)
        Ksig = np.where(np.abs(pctchg) < self.pct_thresh, 0, Kdir)

        # 6) agreement + hold‐last
        agree = np.where((Hsig == Ksig) & (Ksig != 0), Hsig, 0)
        state = agree.copy()
        for i in range(nInst):
            for t in range(1, T):
                if state[i,t] == 0:
                    state[i,t] = state[i,t-1]

        # 7) cheat‐filter short runs
        self.state_cheat = np.vstack([
            remove_short_runs(state[i], self.M_cheat)
            for i in range(nInst)
        ])

    @export
    def Alg(self, prcSoFar: np.ndarray) -> np.ndarray:
        """
        prcSoFar: (nInst, t_run) price history to current bar.
        returns:   (nInst,) integer positions ±max_shares.
        """
        nInst, t_run = prcSoFar.shape
        t = t_run - 1  # current time index

        prices_now = prcSoFar[:,t]
        sigs       = self.state_cheat[:,t]

        positions = np.zeros(nInst, dtype=int)
        valid     = prices_now > 0
        shares    = (self.capital_per_inst // prices_now[valid]).astype(int)
        positions[valid] = sigs[valid] * shares
        return positions
