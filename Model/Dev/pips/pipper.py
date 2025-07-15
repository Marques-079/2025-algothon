


# #Blscklsit model
# #!/usr/bin/env python3
# """
# dynamic_pip_trader.py  –  “black-list” + overflow-safe edition
# ──────────────────────────────────────────────────────────────
# Trades the rolling-window causal-PIP strategy **except** it stays flat
# on the eight instruments that were chronic losers in the back-test:

#     BLACK_LIST = {4, 5, 8, 22, 23, 28, 48, 49}

# The only code change from the previous file is a one-line safety cast
# inside `best_threshold()` so the interim PnL calculation never overflows
# (`int8 * 10 000` → `int32`).

# Everything else (N/W pairs, re-calibration cadence, position sizing)
# stays exactly the same.
# """
# from typing import List, Optional, Tuple
# import numpy as np
# from Model.standard_template import Trader, export

# # ─────────────── instruments to ignore completely ────────────────
# BLACK_LIST = {4, 5, 8, 22, 23, 28, 48, 49}

# # ─────────────── grid-search parameters (unchanged) ──────────────
# BEST_N: List[int] = [
#     200, 200, 400, 200, 400, 400, 100, 300, 100, 200,
#     100, 500, 100, 300, 200, 100, 200, 100, 200, 300,
#     100, 200, 500, 100, 400, 100, 100, 400, 100, 100,
#     400, 500, 300, 200, 400, 500, 100, 400, 100, 100,
#     100, 300, 400, 500, 100, 400, 500, 100, 500, 300,
# ]
# BEST_W: List[int] = [
#      10,  10, 100,  10, 100,  10, 100,  25,  10, 100,
#     100,  50,  25,  10,  50,  25,  10,  25,  50, 100,
#     100,  10,  10,  50,  25,  25,  10,  50,  25, 100,
#      25,  10, 100,  50,  25,  10,  50, 100,  50,  25,
#      25,  10, 100, 100,  50,  50,  10,  25,  25,  25,
# ]

# POSITION_LIMIT   = 10_000
# THR_MIN, THR_MAX, THR_STEP = 0.0, 0.700, 0.0005
# THR_GRID         = np.arange(THR_MIN, THR_MAX + THR_STEP, THR_STEP)

# # ───────────── helper functions ──────────────────────────────────
# def causal_update(px: float,
#                   last_extreme: float,
#                   direction: Optional[int],
#                   thr: float) -> Tuple[int, float]:
#     if direction in (0, None):
#         move = (px - last_extreme) / last_extreme
#         if abs(move) >= thr:
#             return (1 if move > 0 else -1), px
#         return 0, last_extreme

#     if direction == 1:                         # up-swing
#         if px > last_extreme:
#             return 1, px
#         if (last_extreme - px) / last_extreme >= thr:
#             return -1, px
#         return 1, last_extreme

#     # direction == −1 (down-swing)
#     if px < last_extreme:
#         return -1, px
#     if (px - last_extreme) / last_extreme >= thr:
#         return 1, px
#     return -1, last_extreme


# def best_threshold(window: np.ndarray) -> float:
#     """
#     Grid search that is now overflow-safe:
#     dir_vec (int8) → cast to int32 before multiplying by 10 000.
#     """
#     best_thr, best_pnl = THR_GRID[0], -np.inf
#     for thr in THR_GRID:
#         dir_vec = np.zeros(len(window), np.int8)
#         le, d = window[0], 0
#         for k in range(1, len(window)):
#             d, le = causal_update(window[k], le, d, thr)
#             dir_vec[k] = d
#         # ---------- SAFETY CAST -----------------------------------
#         pnl = np.sum(dir_vec[:-1].astype(np.int32) *
#                      POSITION_LIMIT *
#                      np.diff(window))
#         # -----------------------------------------------------------
#         if pnl > best_pnl:
#             best_pnl, best_thr = pnl, thr
#     return best_thr

# # ───────────── trader class ─────────────────────────────────────
# class CausalPIPTrader(Trader):
#     class _State:
#         __slots__ = ("n","w","next_cal","thr","dir","le")
#         def __init__(self, n:int, w:int):
#             self.n, self.w = n, w
#             self.next_cal  = n
#             self.thr       = None
#             self.dir       = 0
#             self.le        = None

#     def __init__(self):
#         super().__init__()
#         self._states = [self._State(n,w) for n,w in zip(BEST_N, BEST_W)]
#         print("CausalPIPTrader (black-list edition) ready.")

#     @export
#     def Alg(self, prcSoFar: np.ndarray) -> np.ndarray:
#         n_inst, t_seen = prcSoFar.shape
#         if n_inst != 50:
#             raise ValueError("Expected 50 instruments.")
#         positions = np.zeros(n_inst, np.int32)

#         for i in range(n_inst):
#             if i in BLACK_LIST:
#                 continue                       # stay flat forever

#             st  = self._states[i]
#             px  = prcSoFar[i, -1]

#             if st.le is None:
#                 st.le = px

#             if t_seen >= st.next_cal:
#                 win = prcSoFar[i, t_seen - st.n : t_seen]
#                 st.thr = best_threshold(win)
#                 st.next_cal += st.w
#                 print(f"[t={t_seen:4d}] Inst {i:2d} thr={st.thr:.4f}")

#             if st.thr is None:
#                 continue

#             st.dir, st.le = causal_update(px, st.le, st.dir, st.thr)
#             positions[i] = st.dir * POSITION_LIMIT

#         return positions


# __all__ = ["CausalPIPTrader"]


"""
dynamic_pip_trader.py
─────────────────────
Live-trading implementation of the **window-N / step-W** re-calibration
logic.  The class keeps a tiny, per-instrument state object so each call
to `Alg()` is **O(1)**.

Strategy recap
--------------
│  • For instrument *i* we fix (N[i], W[i]) found in the prior grid-search.
│  • Once we have at least N[i] bars of history:
│        – Find the best pivot-threshold on the last N[i] bars
│          (grid 0 → 0.700 step 0.0005).
│        – Trade the NEXT W[i] bars with that fixed threshold.
│        – Repeat.
│  • Position = ±POSITION_LIMIT shares using the causal PIP signal.
│  • Commission/slippage handled by the back-tester / broker layer.
"""
from pathlib import Path
from typing  import List, Optional, Tuple

import numpy as np
from Model.standard_template import Trader, export

# ─────────────────── hyper-parameters from the grid-search ──────────────
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

POSITION_LIMIT = 10_000
THR_MIN, THR_MAX, THR_STEP = 0.0, 0.700, 0.0005
THR_GRID = np.arange(THR_MIN, THR_MAX + THR_STEP, THR_STEP, dtype=float)


# ────────────────────────── helper functions ────────────────────────────
def causal_update(px: float,
                  last_extreme: float,
                  direction: Optional[int],
                  thr: float) -> Tuple[int, float]:
    """
    O(1) causal PIP update for a single new price.
    direction: None (0), +1 (up), -1 (down).
    Returns (new_direction, new_last_extreme).
    """
    if direction == 0 or direction is None:
        move = (px - last_extreme) / last_extreme
        if abs(move) >= thr:
            return (1 if move > 0 else -1), px
        return 0, last_extreme

    if direction == 1:  # up-swing
        if px > last_extreme:
            return 1, px
        if (last_extreme - px) / last_extreme >= thr:
            return -1, px
        return 1, last_extreme

    # direction == -1  (down-swing)
    if px < last_extreme:
        return -1, px
    if (px - last_extreme) / last_extreme >= thr:
        return 1, px
    return -1, last_extreme


def best_threshold(window: np.ndarray) -> float:
    """
    Grid search for the threshold that maximises net PnL on `window`.
    Uses the same causal-PnL definition as the research phase.
    """
    best_thr = THR_GRID[0]
    best_pnl = -np.inf
    for thr in THR_GRID:
        dir_vec = np.zeros_like(window, dtype=np.int8)
        last_extreme = window[0]
        d = 0
        for i in range(1, len(window)):
            d, last_extreme = causal_update(window[i], last_extreme, d, thr)
            dir_vec[i] = d
        pos = dir_vec.astype(np.int32) * POSITION_LIMIT
        pnl = np.sum(pos[:-1] * np.diff(window))
        if pnl > best_pnl:
            best_pnl, best_thr = pnl, thr
    return best_thr


# ───────────────────────────── trader class ──────────────────────────────
class CausalPIPTrader(Trader):
    """
    Per-instrument rolling-calibration causal-PIP trader.
    """

    class _InstrState:
        __slots__ = ("n", "w", "next_calib", "thr",
                     "dir", "last_extreme")

        def __init__(self, n: int, w: int):
            self.n = n
            self.w = w
            self.next_calib = n          # first calibration after n bars
            self.thr = None              # threshold (float) once calibrated
            self.dir = 0                 # current direction (-1/0/+1)
            self.last_extreme = None     # float

    # ─────────────────────── initialisation ────────────────────────────
    def __init__(self):
        super().__init__()
        if len(BEST_N) != 50 or len(BEST_W) != 50:
            raise ValueError("Need 50 (N,W) pairs.")
        self._states = [self._InstrState(N, W)
                        for N, W in zip(BEST_N, BEST_W)]
        print("DynamicPIPTrader initialised with per-instrument (N,W) pairs.")

    # ───────────────────────── core algorithm ───────────────────────────
    @export
    def Alg(self, prcSoFar: np.ndarray) -> np.ndarray:
        """
        Called once per bar by the back-tester / broker.
        Parameters
        ----------
        prcSoFar : np.ndarray  shape (nInst, t_seen)
            Complete price history up to **and including** the current bar.
        Returns
        -------
        np.ndarray(int32) of target share positions.
        """
        nInst, t_seen = prcSoFar.shape
        if nInst != 50:
            raise ValueError("Expecting 50 instruments.")

        positions = np.zeros(nInst, dtype=np.int32)

        for i in range(nInst):
            state  = self._states[i]
            price  = prcSoFar[i, -1]

            # first bar initialisation
            if state.last_extreme is None:
                state.last_extreme = price

            # ── (re)calibrate threshold when due ────────────────────
            if t_seen >= state.next_calib:
                window = prcSoFar[i, t_seen - state.n : t_seen]
                state.thr = best_threshold(window)
                state.next_calib += state.w
                print(f"[t={t_seen:4d}] Inst {i:2d} recalibrated: "
                      f"N={state.n} W={state.w} thr={state.thr:.4f}")

            # ── if not yet calibrated, stay flat ────────────────────
            if state.thr is None:
                positions[i] = 0
                continue

            # ── causal one-step update ──────────────────────────────
            state.dir, state.last_extreme = causal_update(
                price, state.last_extreme, state.dir, state.thr
            )
            positions[i] = state.dir * POSITION_LIMIT

        return positions


__all__ = ["DynamicPIPTrader"]





# """
# causal_pip_trader_optimized.py
# ──────────────────────────────
# Trader that applies **instrument-specific pivot thresholds** optimised
# via the grid-search you ran (0.005 → 0.500, step 0.0005) on the full
# 1 000-bar training window.

# • Position = ±10 000 shares when the causal PIP direction is up/down,
#   else 0.
# • Uses a *different* threshold for each of the 50 instruments to decide
#   when a swing has reversed.

# Drop this file into `Model/Dev/…` (or your preferred location) and
# register the class with your back-tester.
# """
# from typing import Sequence

# import numpy as np
# from Model.standard_template import Trader, export


# # # ───────────────── instrument-specific thresholds (length 50) ─────────────
# # #  index : 0       1       2       3       4       5       6       7       8
# # THRESHOLDS: Sequence[float] = [
# #     0.0195, 0.0675, 0.0275, 0.0260, 0.0550, 0.0200, 0.1290, 0.0075, 0.1055,
# #     0.0780, 0.0900, 0.0140, 0.0585, 0.0365, 0.0485, 0.0805, 0.0425, 0.0295,
# #     0.0625, 0.0065, 0.0560, 0.0865, 0.0730, 0.0975, 0.0175, 0.0400, 0.1430,
# #     0.0850, 0.0690, 0.0055, 0.0220, 0.1010, 0.0435, 0.1010, 0.0500, 0.0760,
# #     0.0250, 0.0660, 0.0190, 0.0155, 0.0420, 0.0075, 0.1235, 0.0935, 0.0150,
# #     0.0540, 0.0760, 0.0540, 0.0200, 0.0795,
# # ]

# # #500

# # THRESHOLDS: Sequence[float] = [
# #     0.0195, 0.0505, 0.1145, 0.0175, 0.0505, 0.0435, 0.1185, 0.0075, 0.0395, 0.0330,
# #     0.0900, 0.0250, 0.0175, 0.0270, 0.0350, 0.0455, 0.0425, 0.0210, 0.0375, 0.0065,
# #     0.0225, 0.0535, 0.0635, 0.0925, 0.0300, 0.0390, 0.0285, 0.0770, 0.0640, 0.0095,
# #     0.0525, 0.0820, 0.0430, 0.0485, 0.0500, 0.0870, 0.0100, 0.0660, 0.0075, 0.0150,
# #     0.0545, 0.0010, 0.0775, 0.0920, 0.0180, 0.0540, 0.0495, 0.0635, 0.0940, 0.0675
# # ]

# #700

# THRESHOLDS: Sequence[float] = [
#     0.0290, 0.0645, 0.1325, 0.0260, 0.0550, 0.0465, 0.1185, 0.1370, 0.0400, 0.0330,
#     0.0900, 0.0075, 0.0585, 0.0580, 0.0350, 0.0385, 0.0425, 0.0315, 0.0540, 0.0065,
#     0.0560, 0.0605, 0.0635, 0.0925, 0.0175, 0.0400, 0.0285, 0.0790, 0.0690, 0.0100,
#     0.0525, 0.1010, 0.0430, 0.0540, 0.0500, 0.0870, 0.0240, 0.0660, 0.0205, 0.0155,
#     0.0420, 0.0010, 0.1840, 0.0125, 0.0265, 0.0540, 0.0495, 0.0180, 0.0825, 0.1265
# ]




# POSITION_LIMIT: int = 10_000


# # ────────────────────────── helper: causal signal ─────────────────────────
# def _causal_signal(series: np.ndarray, thr: float) -> int:
#     """
#     Return the latest position direction (+1 / 0 / -1) for a single
#     instrument, given its price *history* and a pivot threshold `thr`.
#     """
#     last_extreme = series[0]
#     direction    = None  # 'up' | 'down' | None

#     for px in series[1:]:
#         move = (px - last_extreme) / last_extreme

#         if direction is None:
#             if abs(move) >= thr:
#                 direction    = 'up' if move > 0 else 'down'
#                 last_extreme = px
#         else:
#             if direction == 'up':
#                 if px > last_extreme:                                   # extend rally
#                     last_extreme = px
#                 elif (last_extreme - px) / last_extreme >= thr:         # reversal
#                     direction, last_extreme = 'down', px
#             else:                                                       # direction == 'down'
#                 if px < last_extreme:                                   # extend decline
#                     last_extreme = px
#                 elif (px - last_extreme) / last_extreme >= thr:         # reversal
#                     direction, last_extreme = 'up', px

#     return 1 if direction == 'up' else (-1 if direction == 'down' else 0)


# # ───────────────────────── trader implementation ──────────────────────────
# class CausalPIPTrader(Trader):
#     """
#     Uses the pre-optimised threshold list to generate target positions
#     of ±10 000 shares (or 0) for each of the 50 instruments.
#     """

#     def __init__(self,
#                  thresholds: Sequence[float] = THRESHOLDS,
#                  position_limit: int = POSITION_LIMIT):
#         """
#         Parameters
#         ----------
#         thresholds : sequence(float), length 50
#             Instrument-specific pivot thresholds.
#         position_limit : int
#             Maximum absolute share count per instrument.
#         """
#         super().__init__()
#         if len(thresholds) != 50:
#             raise ValueError("Expecting 50 threshold values (one per instrument)")
#         self.thresholds     = np.asarray(thresholds, dtype=float)
#         self.position_limit = int(position_limit)

#     # called once per bar by the back-tester
#     @export
#     def Alg(self, prcSoFar: np.ndarray) -> np.ndarray:
#         """
#         Parameters
#         ----------
#         prcSoFar : ndarray (nInst × t_seen)
#             Price history up to *and including* the current bar.

#         Returns
#         -------
#         ndarray(int)
#             Target share positions: +10 000, −10 000, or 0 for each instrument.
#         """
#         nInst, _ = prcSoFar.shape
#         if nInst != 50:
#             raise ValueError("Price matrix must contain 50 instruments")

#         pos_dir = np.zeros(nInst, dtype=np.int8)

#         # compute per-instrument direction with its own threshold
#         for i in range(nInst):
#             pos_dir[i] = _causal_signal(prcSoFar[i], self.thresholds[i])

#         # scale ±1/0 to full share size
#         return pos_dir.astype(np.int32) * self.position_limit


# __all__ = ["CausalPIPTraderOptimised"]




# import numpy as np
# from Model.standard_template import Trader, export   # back-tester hooks

# # ────────────────────────── helpers ──────────────────────────
# def causal_signal(series: np.ndarray, threshold: float) -> int:
#     """
#     Return the latest position direction (+1 / 0 / -1) for a single instrument.
#     """
#     last_extreme = series[0]
#     direction    = None                      # 'up', 'down', or None

#     for px in series[1:]:
#         move = (px - last_extreme) / last_extreme

#         if direction is None:
#             if abs(move) >= threshold:
#                 direction    = 'up' if move > 0 else 'down'
#                 last_extreme = px
#         else:
#             if direction == 'up':
#                 if px > last_extreme:                       # extend rally
#                     last_extreme = px
#                 elif (last_extreme - px) / last_extreme >= threshold:
#                     direction    = 'down'                   # reversal
#                     last_extreme = px
#             else:                                           # direction == 'down'
#                 if px < last_extreme:                       # extend decline
#                     last_extreme = px
#                 elif (px - last_extreme) / last_extreme >= threshold:
#                     direction    = 'up'                     # reversal
#                     last_extreme = px

#     return 1 if direction == 'up' else (-1 if direction == 'down' else 0)


# # ───────────────────────── trader class ──────────────────────
# class CausalPIPTrader(Trader):
#     """
#     Computes a causal swing direction for each instrument and
#     returns a *target* position of ±position_limit (or 0) shares.
#     """

#     def __init__(self, threshold: float = 0.018, position_limit: int = 10_000):
#         """
#         Parameters
#         ----------
#         threshold      : float
#             Relative swing that defines a pivot, e.g. 0.03 = 3 %.
#         position_limit : int
#             Absolute maximum share count per instrument (long or short).
#         """
#         super().__init__()
#         self.threshold      = float(threshold)
#         self.position_limit = int(position_limit)

#     @export
#     def Alg(self, prcSoFar: np.ndarray) -> np.ndarray:
#         """
#         Called once per bar by the back-tester.

#         Parameters
#         ----------
#         prcSoFar : ndarray  (nInst × t_seen)
#             Price history up to *and including* the current bar.

#         Returns
#         -------
#         ndarray(int)
#             Target share positions: +position_limit, -position_limit, or 0.
#         """
#         nInst, _ = prcSoFar.shape
#         pos_dir  = np.zeros(nInst, dtype=int)

#         for i in range(nInst):
#             pos_dir[i] = causal_signal(prcSoFar[i], self.threshold)

#         # scale the ±1/0 direction to full size
#         return pos_dir * self.position_limit


# __all__ = ["CausalPIPTrader"]

