from Model.standard_template import Trader, export
import numpy as np
import pandas as pd

class MAStrategy(Trader): #EMA
    def __init__(
        self,
        short_window: int = 15,
        long_window:  int = 30,
        threshold_up: float = 0.0,
        threshold_down: float = 0.0,
        notional:     float = 10_000
    ):
        super().__init__()
        self.short_w        = short_window
        self.long_w         = long_window
        self.threshold_up   = threshold_up
        self.threshold_down = threshold_down
        self.notional       = notional

        self.first          = True
        self.nInst          = None
        self.prev_dirs      = None
        self.prev_positions = None

    @export
    def getPosition(self, prcSoFar: np.ndarray) -> np.ndarray:
        """
        prcSoFar: shape (nInst, t) array of price history (close only)
        returns:  array of length nInst, integer share counts
        """
        nInst, t = prcSoFar.shape
        if self.first:
            self.nInst         = nInst
            self.prev_dirs     = np.zeros(nInst, dtype=int)
            self.prev_positions = np.zeros(nInst, dtype=int)
            self.first          = False

        # Build a DataFrame with timesteps as rows, instruments as columns
        df = pd.DataFrame(prcSoFar.T)

        # Compute EMAs along the time axis
        ema_s = df.ewm(span=self.short_w, adjust=False).mean()
        ema_l = df.ewm(span=self.long_w,  adjust=False).mean()

        # Grab the latest values
        last_price = prcSoFar[:, -1]
        es_last    = ema_s.iloc[-1].values
        el_last    = ema_l.iloc[-1].values

        # Unit-less gap
        norm_gap = (es_last - el_last) / el_last

        # Regime: +1 = long zone, -1 = short zone, 0 = neutral
        desired_dirs = np.where(
            norm_gap >  self.threshold_up,  1,
            np.where(norm_gap < -self.threshold_down, -1, 0)
        )

        # Maximum shares per instrument
        max_shares = np.floor(self.notional / last_price).astype(int)

        # Base positions: direction * max shares
        positions = desired_dirs * max_shares

        # If regime didn’t change, hold the previous position
        same = desired_dirs == self.prev_dirs
        positions[same] = self.prev_positions[same]

        # Save for next tick
        self.prev_dirs       = desired_dirs.copy()
        self.prev_positions  = positions.copy()

        return positions


# from Model.standard_template import Trader, export
# import numpy as np
# import pandas as pd

# class MAStrategy(Trader):
#     def __init__(
#         self,
#         short_window: int = 20,
#         long_window:  int = 50,
#         threshold_up: float = 0.0015,
#         threshold_down: float = 0.010,
#         notional: float = 10_000
#     ):
#         super().__init__()
#         self.short_w        = short_window
#         self.long_w         = long_window
#         self.threshold_up   = threshold_up
#         self.threshold_down = threshold_down
#         self.notional       = notional

#         self.first          = True
#         self.nInst          = None
#         self.prev_dirs      = None
#         self.prev_positions = None

#     @export
#     def getPosition(self, prcSoFar: np.ndarray) -> np.ndarray:
#         """
#         prcSoFar: shape (nInst, t) array of prices history
#         returns:  array of length nInst, integer share counts
#         """
#         # determine dimensions
#         nInst, t = prcSoFar.shape
#         if self.first:
#             self.nInst     = nInst
#             self.prev_dirs = np.zeros(nInst, dtype=int)
#             self.prev_positions = np.zeros(nInst, dtype=int)
#             self.first = False

#         # build DataFrame to compute rolling MAs over time axis
#         df = pd.DataFrame(prcSoFar.T)  # rows = timesteps, cols = instruments
#         ma_s   = df.rolling(window=self.short_w, min_periods=1).mean()
#         ma_l   = df.rolling(window=self.long_w,  min_periods=1).mean()

#         # grab last values
#         last_price = prcSoFar[:, -1]
#         ms_last    = ma_s.iloc[-1].values
#         ml_last    = ma_l.iloc[-1].values

#         # normalized gap
#         norm_gap = (ms_last - ml_last) / ml_last

#         # desired direction: +1=long, -1=short, 0=neutral
#         desired_dirs = np.where(
#             norm_gap >  self.threshold_up,  1,
#             np.where(norm_gap < -self.threshold_down, -1, 0)
#         )

#         # compute maximum shares per instrument
#         max_shares = np.floor(self.notional / last_price).astype(int)

#         # initial positions based on desired direction
#         positions = desired_dirs * max_shares

#         # if direction unchanged, keep previous absolute position
#         same_mask = desired_dirs == self.prev_dirs
#         positions[same_mask] = self.prev_positions[same_mask]

#         # save for next call
#         self.prev_dirs      = desired_dirs.copy()
#         self.prev_positions = positions.copy()

#         return positions
