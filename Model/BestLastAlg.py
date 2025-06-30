import numpy as np
import pandas as pd
import math
from StandardTemplate import Trader, export
from market_condition.indicators import atr_close_to_close

class ChooseBestLastBar(Trader):
    def __init__(self):
        super(Trader).__init__()
        self.last_positions = np.zeros(50, dtype=int)
        self.entry_prices = np.full(50, np.nan)
        self.take_profits = np.full(50, np.nan)
        self.stop_losses = np.full(50, np.nan)
        self.tp_ratio = 2
        self.sl_ratio = 1
        self.atr_window = 1
    
    @export
    def Alg(self, prcSoFar: np.ndarray):
        nInst, t = prcSoFar.shape

        # Not enough time to calculate 14 day ATR
        if t < self.atr_window + 1:
            self.last_positions[:] = 0
            self.entry_prices[:] = np.nan
            self.take_profits[:] = np.nan
            self.stop_losses[:] = np.nan
            return np.zeros(nInst, dtype=int)
        
        current_prices = prcSoFar[:, -1]
        prev_prices = prcSoFar[:, -2]
        returns = current_prices - prev_prices
        
        # Compute 14-day ATR using closing prices
        price_diffs = np.abs(np.diff(prcSoFar[:, -self.atr_window-1:], axis=1))  # shape: (nInst, 14)
        atr = np.mean(price_diffs, axis=1)  # shape: (nInst,)
        # atr = abs(returns)

        positions = self.last_positions.copy()

        # Check exits for active positions
        for i in range(nInst):
            if positions[i] != 0:
                # Long position
                if positions[i] > 0:
                    if current_prices[i] >= self.take_profits[i] or current_prices[i] <= self.stop_losses[i]:
                        positions[i] = 0
                        self.entry_prices[i] = np.nan
                        self.take_profits[i] = np.nan
                        self.stop_losses[i] = np.nan
                # Short position
                elif positions[i] < 0:
                    if current_prices[i] <= self.take_profits[i] or current_prices[i] >= self.stop_losses[i]:
                        positions[i] = 0
                        self.entry_prices[i] = np.nan
                        self.take_profits[i] = np.nan
                        self.stop_losses[i] = np.nan

        N = 10
        # Enter new positions in slots where we have none
        num_new_trades = N - np.sum(positions != 0)

        # Get candidates: sort by abs return, highest first
        candidate_indices = np.argsort(np.abs(returns))[::-1]
        selected = 0
        for idx in candidate_indices:
            if positions[idx] == 0:
                # Enter trade based on direction of last return
                direction = np.sign(returns[idx])
                if direction == 0:
                    continue  # no movement, skip

                positions[idx] = math.floor(10000 * direction / current_prices[idx])
                self.entry_prices[idx] = current_prices[idx]
                if direction > 0:
                    self.take_profits[idx] = current_prices[idx] + self.tp_ratio * atr[idx]
                    self.stop_losses[idx] = current_prices[idx] - self.sl_ratio * atr[idx]
                else:
                    self.take_profits[idx] = current_prices[idx] - self.tp_ratio * atr[idx]
                    self.stop_losses[idx] = current_prices[idx] + self.sl_ratio * atr[idx]

                selected += 1
                if selected >= num_new_trades:
                    break

        # Save and return
        self.last_positions = positions.copy()
        return positions
        
