import numpy as np
from typing import Set, Optional, Tuple
from Model.standard_template import Trader, export

# ───────────────────────── global constants ────────────────────────────── #
POSITION_LIMIT = 10_000

BEST_N = np.array([
    200,200,400,200,400,400,100,300,100,200,
    100,500,100,300,200,100,200,100,200,300,
    100,200,500,100,400,100,100,400,100,100,
    400,500,300,200,400,500,100,400,100,100,
    100,300,400,500,100,400,500,100,500,300
], dtype=int)

BEST_W = np.array([
     10, 10,100, 10,100, 10,100, 25, 10,100,
    100, 50, 25, 10, 50, 25, 10, 25, 50,100,
    100, 10, 10, 50, 25, 25, 10, 50, 25,100,
     25, 10,100, 50, 25, 10, 50,100, 50, 25,
     25, 10,100,100, 50, 50, 10, 25, 25, 25
], dtype=int)

THR_MIN, THR_MAX = 0.00, 0.70
THR_COARSE, THR_FINE, THR_ULTRA = 0.05, 0.005, 0.001

# Default “worst” instruments to zero out
DEFAULT_WORST: Set[int] = {22, 48, 6, 29, 9}


# ───────────────────────── helper functions ────────────────────────────── #
def _pivot_update(px: float, le: float, d: int, thr: float) -> Tuple[int, float]:
    if d == 0:
        mv = (px - le) / le
        if abs(mv) >= thr:
            return (1 if mv > 0 else -1), px
        return 0, le
    if d == 1:
        if px > le:
            return 1, px
        if (le - px) / le >= thr:
            return -1, px
        return 1, le
    # d == -1
    if px < le:
        return -1, px
    if (px - le) / le >= thr:
        return 1, px
    return -1, le

def _pnl(series: np.ndarray, thr: float) -> float:
    d, le = 0, series[0]
    pnl = 0.0
    for p0, p1 in zip(series[:-1], series[1:]):
        d, le = _pivot_update(p1, le, d, thr)
        pnl += d * POSITION_LIMIT * (p1 - p0)
    return pnl

def _best_thr(series: np.ndarray) -> float:
    # coarse grid
    grid = np.arange(THR_MIN, THR_MAX + THR_COARSE*0.1, THR_COARSE)
    best = max(grid, key=lambda t: _pnl(series, t))
    # fine grid
    lo, hi = max(THR_MIN, best - THR_COARSE), min(THR_MAX, best + THR_COARSE)
    grid = np.arange(lo, hi + THR_FINE*0.1, THR_FINE)
    best = max(grid, key=lambda t: _pnl(series, t))
    # ultra-fine
    lo, hi = max(THR_MIN, best - THR_FINE), min(THR_MAX, best + THR_FINE)
    grid = np.arange(lo, hi + THR_ULTRA*0.1, THR_ULTRA)
    return max(grid, key=lambda t: _pnl(series, t))


# ───────────────────────── trader class ────────────────────────────────── #
class DynamicPIPTrader(Trader):
    """
    Dynamic PIP trader with hierarchical threshold re-calibration
    and optional zero exposure on a given set of instruments.
    """

    class _State:
        __slots__ = ("n", "w", "next_calib", "thr", "dir", "last_extreme")
        def __init__(self, n: int, w: int):
            self.n            = n
            self.w            = w
            self.next_calib   = n
            self.thr          = None
            self.dir          = 0
            self.last_extreme = None

    def __init__(self, worst: Optional[Set[int]] = None):
        """
        Parameters
        ----------
        worst : Set[int], optional
            Indices of instruments to keep flat (zero position).
            Default = {22,48,6,29,9}.
        """
        super().__init__()
        self.worst = worst if worst is not None else DEFAULT_WORST
        self._states = [
            DynamicPIPTrader._State(n, w)
            for n, w in zip(BEST_N, BEST_W)
        ]

    @export
    def Alg(self, prc: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        prc : np.ndarray, shape (50, t_seen)
            Price history up to current bar (per instrument).
        Returns
        -------
        np.ndarray(int32), shape (50,)
            Target positions: +limit, −limit, or 0.
        """
        n_inst, t_seen = prc.shape
        if n_inst != 50:
            raise ValueError("Expected 50 instruments")

        pos = np.zeros(n_inst, dtype=np.int32)

        for i, state in enumerate(self._states):
            # always flat for worst instruments
            if i in self.worst:
                continue

            price = prc[i, -1]
            if state.last_extreme is None:
                state.last_extreme = price

            # recalibrate threshold if due
            if t_seen >= state.next_calib:
                window = prc[i, max(0, t_seen - state.n) : t_seen]
                state.thr = _best_thr(window)
                state.next_calib = t_seen + state.w

            if state.thr is None:
                continue

            # one-step pivot update
            state.dir, state.last_extreme = _pivot_update(
                price,
                state.last_extreme,
                state.dir,
                state.thr
            )

            pos[i] = state.dir * POSITION_LIMIT

        return pos


__all__ = ["DynamicPIPTrader"]
