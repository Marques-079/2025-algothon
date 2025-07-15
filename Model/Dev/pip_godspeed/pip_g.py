
#!/usr/bin/env python3
"""
timeexit_trader.py
──────────────────
An incremental **Time‑Exit Trader** that keeps the original
HMA → Kalman signal logic while enforcing a *per‑instrument* time‑based
stop‑loss.  The class conforms to the standard template:

```python
from Model.standard_template import Trader, export
```

* **Alg(prcSoFar)** expects an `np.ndarray` of shape `(50, t_seen)` and
  returns a **single** `np.ndarray[int32]` of positions (no confidences).
* Position size = `CAPITAL // price` (sign‑reversed; `+` = long in book).
* Trade is closed (set to `0`) once it has been open for
  `TIME_EXIT_MAX[i]` consecutive bars.
* Commission is not applied inside the class – handle externally if
  needed.
"""
import numpy as np
from collections import deque
from typing import Tuple, List
from Model.standard_template import Trader, export

# ───────────────────── parameters ────────────────────────────────
CAPITAL    = 10_000
HMA_PERIOD = 100
# Per‑instrument time stop (bars) – derived from your grid‑search
TIME_EXIT_MAX: List[int] = [
    200, 100,  50,  50,  50,  50, 100,  50, 250, 250,
    200, 250, 250, 250, 100,  50,  50, 250,  50, 100,
    250,  50,  50, 150, 150, 150, 250,  50, 250,  50,
    250, 200, 250, 150,  50,  50, 200,  50,  50, 200,
    100, 250, 150, 150, 250, 100, 150,  50, 100, 200,
]
if len(TIME_EXIT_MAX) != 50:
    raise ValueError("Need 50 TimeExit parameters (one per instrument)")

# ───────────────────── helper filters ────────────────────────────
class IncWMA:
    def __init__(self, n: int):
        self.n   = n
        self.buf = deque(maxlen=n)
        self.w   = np.arange(1, n + 1, dtype=float)
        self.S   = self.w.sum()
    def update(self, x: float) -> float:
        self.buf.append(x)
        if len(self.buf) < self.n:
            return x
        arr = np.fromiter(self.buf, float, len(self.buf))
        return float((self.w * arr).sum() / self.S)

class IncHMA:
    def __init__(self, n: int):
        self.w_full  = IncWMA(n)
        self.w_half  = IncWMA(max(1, n // 2))
        self.w_final = IncWMA(max(1, int(np.sqrt(n))))
    def update(self, price: float) -> float:
        full  = self.w_full.update(price)
        half  = self.w_half.update(price)
        return self.w_final.update(2.0 * half - full)

class IncKalman:
    def __init__(self, R=0.075, Ql=4e-3, Qt=1e-5):
        self.F = np.array([[1, 1], [0, 1]], float)
        self.H = np.array([[1, 0]], float)
        self.Q = np.diag([Ql, Qt])
        self.R = R
        self.s = None
        self.P = np.eye(2)
    def update(self, x: float) -> float:
        if self.s is None:
            self.s = np.array([x, 0.0], float)
            return x
        self.s = self.F @ self.s
        self.P = self.F @ self.P @ self.F.T + self.Q
        y   = x - (self.H @ self.s)[0]
        S   = (self.H @ self.P @ self.H.T)[0, 0] + self.R
        K   = (self.P @ self.H.T) / S
        self.s += K.flatten() * y
        self.P  = (np.eye(2) - K @ self.H) @ self.P
        return self.s[0]

# ───────────────────── trader class ───────────────────────────────
class TimeExitTrader(Trader):
    """HMA→Kalman with time‑based exit per instrument."""

    class _State:
        __slots__ = ("hma", "kal", "signal", "position", "bars")
        def __init__(self):
            self.hma      = IncHMA(HMA_PERIOD)
            self.kal      = IncKalman()
            self.signal   = 0   # +1 short / -1 long (model convention)
            self.position = 0   # actual signed size
            self.bars     = 0   # bars in current trade

    def __init__(self):
        super().__init__()
        self._states = [self._State() for _ in range(50)]

    # ------------------------------------------------------------
    @export
    def Alg(self, prcSoFar: np.ndarray) -> np.ndarray:
        """Return positions for current bar (shape (50,))."""
        nInst, _ = prcSoFar.shape
        if nInst != 50:
            raise ValueError("Expecting 50 instruments")
        prices_now = prcSoFar[:, -1]
        out        = np.zeros(50, dtype=np.int32)

        for i, price in enumerate(prices_now):
            st = self._states[i]
            h  = st.hma.update(price)
            k  = st.kal.update(price)
            sig = 1 if h > k else -1      # +1 = short, -1 = long (match old conv.)

            # flip -> enter new trade and reset timer
            if sig != st.signal:
                st.signal = sig
                max_shares = int(CAPITAL // price) if price > 0 else 0
                st.position = -sig * max_shares   # sign‑reversed => +long
                st.bars = 0
            else:
                # still same trade – advance timer
                if st.position != 0:
                    st.bars += 1
                    if st.bars >= TIME_EXIT_MAX[i]:
                        st.position = 0
                        st.signal   = 0
                        st.bars     = 0

            out[i] = st.position

        return out



#  import numpy as np
# from typing import Set, Optional, Tuple
# from Model.standard_template import Trader, export

# # ───────────────────────── global constants ────────────────────────────── #
# POSITION_LIMIT = 10_000

# BEST_N = np.array([
#     200,200,400,200,400,400,100,300,100,200,
#     100,500,100,300,200,100,200,100,200,300,
#     100,200,500,100,400,100,100,400,100,100,
#     400,500,300,200,400,500,100,400,100,100,
#     100,300,400,500,100,400,500,100,500,300
# ], dtype=int)

# BEST_W = np.array([
#      10, 10,100, 10,100, 10,100, 25, 10,100,
#     100, 50, 25, 10, 50, 25, 10, 25, 50,100,
#     100, 10, 10, 50, 25, 25, 10, 50, 25,100,
#      25, 10,100, 50, 25, 10, 50,100, 50, 25,
#      25, 10,100,100, 50, 50, 10, 25, 25, 25
# ], dtype=int)

# THR_MIN, THR_MAX = 0.00, 0.70
# THR_COARSE, THR_FINE, THR_ULTRA = 0.05, 0.005, 0.001

# # Default “worst” instruments to zero out
# DEFAULT_WORST: Set[int] = {22, 48, 6, 29, 9}


# # ───────────────────────── helper functions ────────────────────────────── #
# def _pivot_update(px: float, le: float, d: int, thr: float) -> Tuple[int, float]:
#     if d == 0:
#         mv = (px - le) / le
#         if abs(mv) >= thr:
#             return (1 if mv > 0 else -1), px
#         return 0, le
#     if d == 1:
#         if px > le:
#             return 1, px
#         if (le - px) / le >= thr:
#             return -1, px
#         return 1, le
#     # d == -1
#     if px < le:
#         return -1, px
#     if (px - le) / le >= thr:
#         return 1, px
#     return -1, le

# def _pnl(series: np.ndarray, thr: float) -> float:
#     d, le = 0, series[0]
#     pnl = 0.0
#     for p0, p1 in zip(series[:-1], series[1:]):
#         d, le = _pivot_update(p1, le, d, thr)
#         pnl += d * POSITION_LIMIT * (p1 - p0)
#     return pnl

# def _best_thr(series: np.ndarray) -> float:
#     # coarse grid
#     grid = np.arange(THR_MIN, THR_MAX + THR_COARSE*0.1, THR_COARSE)
#     best = max(grid, key=lambda t: _pnl(series, t))
#     # fine grid
#     lo, hi = max(THR_MIN, best - THR_COARSE), min(THR_MAX, best + THR_COARSE)
#     grid = np.arange(lo, hi + THR_FINE*0.1, THR_FINE)
#     best = max(grid, key=lambda t: _pnl(series, t))
#     # ultra-fine
#     lo, hi = max(THR_MIN, best - THR_FINE), min(THR_MAX, best + THR_FINE)
#     grid = np.arange(lo, hi + THR_ULTRA*0.1, THR_ULTRA)
#     return max(grid, key=lambda t: _pnl(series, t))


# # ───────────────────────── trader class ────────────────────────────────── #
# class DynamicPIPTrader(Trader):
#     """
#     Dynamic PIP trader with hierarchical threshold re-calibration
#     and optional zero exposure on a given set of instruments.
#     """

#     class _State:
#         __slots__ = ("n", "w", "next_calib", "thr", "dir", "last_extreme")
#         def __init__(self, n: int, w: int):
#             self.n            = n
#             self.w            = w
#             self.next_calib   = n
#             self.thr          = None
#             self.dir          = 0
#             self.last_extreme = None

#     def __init__(self, worst: Optional[Set[int]] = None):
#         """
#         Parameters
#         ----------
#         worst : Set[int], optional
#             Indices of instruments to keep flat (zero position).
#             Default = {22,48,6,29,9}.
#         """
#         super().__init__()
#         self.worst = worst if worst is not None else DEFAULT_WORST
#         self._states = [
#             DynamicPIPTrader._State(n, w)
#             for n, w in zip(BEST_N, BEST_W)
#         ]

#     @export
#     def Alg(self, prc: np.ndarray) -> np.ndarray:
#         """
#         Parameters
#         ----------
#         prc : np.ndarray, shape (50, t_seen)
#             Price history up to current bar (per instrument).
#         Returns
#         -------
#         np.ndarray(int32), shape (50,)
#             Target positions: +limit, −limit, or 0.
#         """
#         n_inst, t_seen = prc.shape
#         if n_inst != 50:
#             raise ValueError("Expected 50 instruments")

#         pos = np.zeros(n_inst, dtype=np.int32)

#         for i, state in enumerate(self._states):
#             # always flat for worst instruments
#             if i in self.worst:
#                 continue

#             price = prc[i, -1]
#             if state.last_extreme is None:
#                 state.last_extreme = price

#             # recalibrate threshold if due
#             if t_seen >= state.next_calib:
#                 window = prc[i, max(0, t_seen - state.n) : t_seen]
#                 state.thr = _best_thr(window)
#                 state.next_calib = t_seen + state.w

#             if state.thr is None:
#                 continue

#             # one-step pivot update
#             state.dir, state.last_extreme = _pivot_update(
#                 price,
#                 state.last_extreme,
#                 state.dir,
#                 state.thr
#             )

#             pos[i] = state.dir * POSITION_LIMIT

#         return pos


# __all__ = ["DynamicPIPTrader"]
