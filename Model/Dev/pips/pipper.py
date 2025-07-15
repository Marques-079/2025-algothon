"""
causal_pip_trader_optimized.py
──────────────────────────────
Trader that applies **instrument-specific pivot thresholds** optimised
via the grid-search you ran (0.005 → 0.500, step 0.0005) on the full
1 000-bar training window.

• Position = ±10 000 shares when the causal PIP direction is up/down,
  else 0.
• Uses a *different* threshold for each of the 50 instruments to decide
  when a swing has reversed.

Drop this file into `Model/Dev/…` (or your preferred location) and
register the class with your back-tester.
"""
from typing import Sequence

import numpy as np
from Model.standard_template import Trader, export


# # ───────────────── instrument-specific thresholds (length 50) ─────────────
# #  index : 0       1       2       3       4       5       6       7       8
# THRESHOLDS: Sequence[float] = [
#     0.0195, 0.0675, 0.0275, 0.0260, 0.0550, 0.0200, 0.1290, 0.0075, 0.1055,
#     0.0780, 0.0900, 0.0140, 0.0585, 0.0365, 0.0485, 0.0805, 0.0425, 0.0295,
#     0.0625, 0.0065, 0.0560, 0.0865, 0.0730, 0.0975, 0.0175, 0.0400, 0.1430,
#     0.0850, 0.0690, 0.0055, 0.0220, 0.1010, 0.0435, 0.1010, 0.0500, 0.0760,
#     0.0250, 0.0660, 0.0190, 0.0155, 0.0420, 0.0075, 0.1235, 0.0935, 0.0150,
#     0.0540, 0.0760, 0.0540, 0.0200, 0.0795,
# ]


THRESHOLDS: Sequence[float] = [
    0.0195, 0.0505, 0.1145, 0.0175, 0.0505, 0.0435, 0.1185, 0.0075, 0.0395, 0.0330,
    0.0900, 0.0250, 0.0175, 0.0270, 0.0350, 0.0455, 0.0425, 0.0210, 0.0375, 0.0065,
    0.0225, 0.0535, 0.0635, 0.0925, 0.0300, 0.0390, 0.0285, 0.0770, 0.0640, 0.0095,
    0.0525, 0.0820, 0.0430, 0.0485, 0.0500, 0.0870, 0.0100, 0.0660, 0.0075, 0.0150,
    0.0545, 0.0010, 0.0775, 0.0920, 0.0180, 0.0540, 0.0495, 0.0635, 0.0940, 0.0675
]



POSITION_LIMIT: int = 10_000

commRate = 0.0005
dlrPosLimit = 10000

# ────────────────────────── helper: causal signal ─────────────────────────
def _causal_signal(series: np.ndarray, thr: float) -> int:
    """
    Return the latest position direction (+1 / 0 / -1) for a single
    instrument, given its price *history* and a pivot threshold `thr`.
    """
    last_extreme = series[0]
    direction    = None  # 'up' | 'down' | None

    for px in series[1:]:
        move = (px - last_extreme) / last_extreme

        if direction is None:
            if abs(move) >= thr:
                direction    = 'up' if move > 0 else 'down'
                last_extreme = px
        else:
            if direction == 'up':
                if px > last_extreme:                                   # extend rally
                    last_extreme = px
                elif (last_extreme - px) / last_extreme >= thr:         # reversal
                    direction, last_extreme = 'down', px
            else:                                                       # direction == 'down'
                if px < last_extreme:                                   # extend decline
                    last_extreme = px
                elif (px - last_extreme) / last_extreme >= thr:         # reversal
                    direction, last_extreme = 'up', px

    return 1 if direction == 'up' else (-1 if direction == 'down' else 0)


# ───────────────────────── trader implementation ──────────────────────────
class CausalPIPTrader(Trader):
    def __init__(self,
                 thresholds: Sequence[float] = THRESHOLDS,
                 position_limit: int = POSITION_LIMIT):
        super().__init__()
        if len(thresholds) != 50:
            raise ValueError("Expecting 50 threshold values (one per instrument)")

        self.thresholds     = np.asarray(thresholds, dtype=float)
        self.position_limit = int(position_limit)
        self.nInst          = 50
        
        self.first = True

        self.curPos = np.zeros(50)
        self.cash = 0
        self.totDVolume = 0
        self.value = 0
        self.todayPLL = []
        
        self.bl = np.full(50,1.0)
    @export
    def Alg(self, prcSoFar: np.ndarray) -> np.ndarray:
        nInst, t = prcSoFar.shape
        if self.first:
            self.first = False
            for i in range(1,t):
                simPrc = prcSoFar[:,:i]
                nInst, _ = simPrc.shape

                pos_dir = np.zeros(nInst)

                for i in range(nInst):
                    pos_dir[i] = np.int64(_causal_signal(simPrc[i], self.thresholds[i]))

                new_position = pos_dir * 10_000/prcSoFar[:,-1]

                # Update performance stats incrementally
                self.update_performance(simPrc, new_position)
            

        pos_dir = np.zeros(nInst)

        for i in range(nInst):
            pos_dir[i] = _causal_signal(prcSoFar[i], self.thresholds[i])

        new_position = pos_dir * self.position_limit
        new_position*=self.bl

        # Update performance stats incrementally
        self.update_performance(prcSoFar, new_position)
        cpnl = self.cum_pnl_per()
        new_position[cpnl < 10_000]  = 0

        return new_position

    def update_performance(self, prices: np.ndarray, new_position: np.ndarray):
        prcHistSoFar = prices
        curPrices = prcHistSoFar[:,-1]
        # Trading, do not do it on the very last day of the test
        newPosOrig = new_position
        posLimits = np.array([int(x) for x in dlrPosLimit / curPrices])
        newPos = np.clip(newPosOrig, -posLimits, posLimits)
        deltaPos = newPos - self.curPos
        dvolumes = curPrices * np.abs(deltaPos)
        dvolume = np.sum(dvolumes)
        self.totDVolume += np.sum(dvolumes)

        comms = ( dvolumes * commRate )
        deltaValue = curPrices * deltaPos

        self.cash -= deltaValue + comms
        self.curPos = np.array(newPos)
        posValue = self.curPos*curPrices
        todayPL = self.cash + posValue - self.value
        self.value = self.cash + posValue

        self.todayPLL.append(todayPL)

    def cum_pnl_per(self):
        pll = np.array(self.todayPLL)
        cpnl = pll.sum(axis=0)
        return cpnl

    

__all__ = ["CausalPIPTraderOptimised"]




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

