#!/usr/bin/env python3
"""
fourier_arima_trader.py

A live‐trading implementation of a Fourier‐regression “ARIMA‐style” one‐step
forecasting strategy.  At each bar, for each instrument:

  1. Fit a ridge‐regularized Fourier‐basis linear regression on the historical first differences.
  2. Forecast the next difference, integrate to get next price.
  3. Go long if price is expected to rise, short if expected to fall.

This class is fully causal and retrains on every new bar (“walk‐forward”) but
never fails due to rank‐deficiency thanks to the ridge penalty.
"""

import numpy as np
from sklearn.linear_model import Ridge
from Model.standard_template import Trader, export

# ──────────────────── Hyper‐parameters ─────────────────────
POSITION_LIMIT = 10_000   # shares per instrument
PERIOD         = 100      # assumed cycle length
TERMS          = 20        # number of Fourier harmonics
ALPHA          = 1.0      # ridge regularization strength

# ──────────── Fourier Feature Generator ────────────────
def make_fourier(t: np.ndarray, T: float, K: int) -> np.ndarray:
    """
    Builds a design matrix of shape (len(t), 1 + 2K) with columns:
      [1,
       sin(2π·1·t/T), cos(2π·1·t/T),
       …,
       sin(2π·K·t/T), cos(2π·K·t/T)]
    """
    features = [np.ones_like(t)]
    for k in range(1, K + 1):
        features.append(np.sin(2 * np.pi * k * t / T))
        features.append(np.cos(2 * np.pi * k * t / T))
    return np.column_stack(features)

# ─────────────────── Trader Class ────────────────────────
class FourierARIMATrader(Trader):
    """
    Trader that forecasts the next bar’s price via a ridge‐regularized
    Fourier‐regression on the first differences, and sets position = ±POSITION_LIMIT.
    """
    def __init__(self,
                 period: float = PERIOD,
                 terms: int = TERMS,
                 alpha: float = ALPHA,
                 position_limit: int = POSITION_LIMIT):
        self.period = period
        self.terms = terms
        self.alpha = alpha
        self.position_limit = position_limit

    @export
    def Alg(self, prcSoFar: np.ndarray) -> np.ndarray:
        """
        prcSoFar : np.ndarray, shape (n_inst, t)
                   price history up to the current bar (inclusive).

        Returns
        -------
        positions : np.ndarray of shape (n_inst,), dtype=int
                    +position_limit for expected rise,
                    -position_limit for expected fall,
                     0 if not enough history.
        """
        n_inst, t = prcSoFar.shape
        positions = np.zeros(n_inst, dtype=int)

        # Need at least two points to diff
        if t < 2:
            return positions

        # time‐index for diffs: [1,2,…,t-1]
        time_idx = np.arange(1, t)
        X = make_fourier(time_idx, self.period, self.terms)
        
        for i in range(n_inst):
            hist  = prcSoFar[i]
            diffs = np.diff(hist)  # length t-1

            # ridge‐regularized fit
            model = Ridge(alpha=self.alpha).fit(X, diffs)

            # predict next diff at time = t
            X_next    = make_fourier(np.array([t]), self.period, self.terms)
            diff_pred = model.predict(X_next)[0]
            price_pred = hist[-1] + diff_pred

            positions[i] = self.position_limit if price_pred > hist[-1] else -self.position_limit

        return positions
