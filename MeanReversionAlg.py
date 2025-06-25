import numpy as np
import pandas as pd
import math
from templates.StandardTemplate import Trader, export

class MeanReversionTrader(Trader):
    def __init__(self, pairs, beta_matrix, alpha_matrix, z_entry=2, z_exit=0.5):
        """
        Parameters:
            pairs (list of tuple): list of (i, j) instrument index pairs
            beta_matrix (ndarray): matrix of hedge ratios (betas)
            alpha_matrix (ndarray): matrix of intercepts (alphas)
            z_entry (float): z-score threshold to enter a trade
            z_exit (float): z-score threshold to exit a trade
        """
        self.pairs = pairs
        self.beta_matrix = beta_matrix
        self.alpha_matrix = alpha_matrix
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.active_positions = {}  # key: (i, j), value: 1 (long spread) or -1 (short spread)
        self.max_hold = 10000
        self.positions = np.zeros(50, dtype=int)

    @export
    def step(self, price_history):
        """
        Called each time step with full price history.

        Parameters:
            price_history (ndarray): shape (n_instruments, n_days), historical prices

        Returns:
            positions (ndarray): shape (n_instruments,), position for each instrument
        """
        n_instruments = price_history.shape[0]
        positions = np.zeros(n_instruments)

        if price_history.shape[1] < 30:
            return positions  # Not enough data yet

        latest_prices = price_history[:, -1]  # Get the last day's prices

        for i, j in self.pairs:
            series_i = price_history[i, -30:]  # Last 30 days of instrument i
            series_j = price_history[j, -30:]  # Last 30 days of instrument j
            beta = self.beta_matrix[i, j]
            alpha = self.alpha_matrix[i, j]

            spread_series = series_i - (beta * series_j + alpha)
            mean_spread = np.mean(spread_series)
            std_spread = np.std(spread_series)

            if std_spread == 0:
                continue

            latest_spread = latest_prices[i] - (beta * latest_prices[j] + alpha)
            z_score = (latest_spread - mean_spread) / std_spread

            active = self.active_positions.get((i, j), 0)
            max_pos = math.floor( ((self.max_hold / max(abs(beta), 1)) / max(latest_prices[i], latest_prices[j]))* 0.75) 
            if active == 0:
                if z_score > self.z_entry:
                    self.positions[i] = -1 * max_pos
                    self.positions[j] = beta * max_pos
                    self.active_positions[(i, j)] = -1
                elif z_score < -self.z_entry:
                    self.positions[i] = 1 * max_pos
                    self.positions[j] = -beta * max_pos
                    self.active_positions[(i, j)] = 1
            else:
                if abs(z_score) < self.z_exit:
                    self.active_positions[(i, j)] = 0
                    self.positions[i] = 0
                    self.positions[j] = 0
        
        return self.positions


    # def run(self, prc_matrix):
    #     T = prc_matrix.shape[1]
    #     for t in range(T):
    #         self.step(prc_matrix, t)

    # def get_trade_log(self):
    #     return pd.DataFrame(self.trades, columns=["Day", "Asset A", "Asset B", "Action", "Z-Score"])

    # def current_positions(self):
    #     return {pair: pos for pair, pos in self.positions.items() if pos != 0}