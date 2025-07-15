#Blscklsit model
#!/usr/bin/env python3
"""
dynamic_pip_trader.py  –  “black-list” + overflow-safe edition
──────────────────────────────────────────────────────────────
Trades the rolling-window causal-PIP strategy **except** it stays flat
on the eight instruments that were chronic losers in the back-test:

    BLACK_LIST = {4, 5, 8, 22, 23, 28, 48, 49}

The only code change from the previous file is a one-line safety cast
inside `best_threshold()` so the interim PnL calculation never overflows
(`int8 * 10 000` → `int32`).

Everything else (N/W pairs, re-calibration cadence, position sizing)
stays exactly the same.
"""
from typing import List, Optional, Tuple
import numpy as np

# ─────────────── instruments to ignore completely ────────────────
BLACK_LIST = {1,4 ,5 , 8, 13, 18, 19, 20, 21, 22, 23, 33, 34, 35, 40, 45, 47, 48, 49 }

# ─────────────── grid-search parameters (unchanged) ──────────────
BEST_N: List[int] = [
    200, 200, 400, 200, 400, 400, 100, 300, 100, 200,
    100, 500, 100, 300, 200, 100, 200, 100, 200, 300,
    100, 200, 500, 100, 400, 100, 100, 400, 100, 100,
    400, 500, 300, 200, 400, 500, 100, 400, 100, 100,
    100, 300, 400, 500, 100, 400, 500, 100, 500, 300,
]
BEST_W: List[int] = [
     10,  10, 100,  10, 100,  10, 100,  25,  10, 100,
    100,  50,  25,  10,  50,  25,  10,  25,  50, 100,
    100,  10,  10,  50,  25,  25,  10,  50,  25, 100,
     25,  10, 100,  50,  25,  10,  50, 100,  50,  25,
     25,  10, 100, 100,  50,  50,  10,  25,  25,  25,
]

POSITION_LIMIT   = 10_000
THR_MIN, THR_MAX, THR_STEP = 0.0, 0.700, 0.0005
THR_GRID         = np.arange(THR_MIN, THR_MAX + THR_STEP, THR_STEP)

print = lambda x: x
# ───────────── helper functions ──────────────────────────────────
def causal_update(px: float,
                  last_extreme: float,
                  direction: Optional[int],
                  thr: float) -> Tuple[int, float]:
    if direction in (0, None):
        move = (px - last_extreme) / last_extreme
        if abs(move) >= thr:
            return (1 if move > 0 else -1), px
        return 0, last_extreme

    if direction == 1:                         # up-swing
        if px > last_extreme:
            return 1, px
        if (last_extreme - px) / last_extreme >= thr:
            return -1, px
        return 1, last_extreme

    # direction == −1 (down-swing)
    if px < last_extreme:
        return -1, px
    if (px - last_extreme) / last_extreme >= thr:
        return 1, px
    return -1, last_extreme


def best_threshold(window: np.ndarray) -> float:
    """
    Grid search that is now overflow-safe:
    dir_vec (int8) → cast to int32 before multiplying by 10 000.
    """
    best_thr, best_pnl = THR_GRID[0], -np.inf
    for thr in THR_GRID:
        dir_vec = np.zeros(len(window), np.int8)
        le, d = window[0], 0
        for k in range(1, len(window)):
            d, le = causal_update(window[k], le, d, thr)
            dir_vec[k] = d
        # ---------- SAFETY CAST -----------------------------------
        pnl = np.sum(dir_vec[:-1].astype(np.int32) *
                     POSITION_LIMIT *
                     np.diff(window))
        # -----------------------------------------------------------
        if pnl > best_pnl:
            best_pnl, best_thr = pnl, thr
    return best_thr

# ───────────── trader class ─────────────────────────────────────
class CausalPIPTrader:
    class _State:
        __slots__ = ("n","w","next_cal","thr","dir","le")
        def __init__(self, n:int, w:int):
            self.n, self.w = n, w
            self.next_cal  = n
            self.thr       = None
            self.dir       = 0
            self.le        = None

    def __init__(self):
        super().__init__()
        self._states = [self._State(n,w) for n,w in zip(BEST_N, BEST_W)]
        print("CausalPIPTrader (black-list edition) ready.")

    def Alg(self, prcSoFar: np.ndarray) -> np.ndarray:
        n_inst, t_seen = prcSoFar.shape
        if n_inst != 50:
            raise ValueError("Expected 50 instruments.")
        positions = np.zeros(n_inst, np.int32)

        for i in range(n_inst):
            if i in BLACK_LIST:
                continue                       # stay flat forever

            st  = self._states[i]
            px  = prcSoFar[i, -1]

            if st.le is None:
                st.le = px

            if t_seen >= st.next_cal:
                win = prcSoFar[i, t_seen - st.n : t_seen]
                st.thr = best_threshold(win)
                st.next_cal += st.w
                print(f"[t={t_seen:4d}] Inst {i:2d} thr={st.thr:.4f}")

            if st.thr is None:
                continue

            st.dir, st.le = causal_update(px, st.le, st.dir, st.thr)
            positions[i] = st.dir * POSITION_LIMIT

        return positions

trader = CausalPIPTrader()
getMyPosition = trader.Alg