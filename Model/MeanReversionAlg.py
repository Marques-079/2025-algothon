import numpy as np
import pandas as pd
import math
from StandardTemplate import Trader, export
import matplotlib.pyplot as plt
import seaborn as sns

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
        self.trades = []  # Log of trades for analysis
        self.entry_prices = {}
        self.spread_log = np.full(10000, np.nan)  # Preallocate for performance
        self.raw_spread_log = np.full(10000, np.nan)

    @export
    def step(self, price_history):
        """
        Called each time step with full price history.

        Parameters:
            price_history (ndarray): shape (n_instruments, n_days), historical prices

        Returns:
            positions (ndarray): shape (n_instruments,), position for each instrument
        """

        n_instruments, t = price_history.shape
        day = t-1
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

            # Add to logs!
            self.spread_log[day] = z_score  # Log z-score by day index
            self.raw_spread_log[day] = latest_spread

            active = self.active_positions.get((i, j), 0)
            max_pos = math.floor(min(self.max_hold / latest_prices[i], self.max_hold / (latest_prices[j]*abs(beta)))*0.95)
            if active == 0:
                if z_score > self.z_entry:
                    self.positions[i] = -1 * max_pos
                    self.positions[j] = beta * max_pos
                    self.entry_prices[(i, j)] = {'i': latest_prices[i], 'j': latest_prices[j]}
                    self.active_positions[(i, j)] = -1
                    self.trades.append([day, i, j, "Enter Short", z_score, None])
                    print(self.positions[i]*latest_prices[i], self.positions[j]*latest_prices[j])
                elif z_score < -self.z_entry:
                    self.positions[i] = 1 * max_pos
                    self.positions[j] = -beta * max_pos
                    self.entry_prices[(i, j)] = {'i': latest_prices[i], 'j': latest_prices[j]}
                    self.active_positions[(i, j)] = 1
                    self.trades.append([day, i, j, "Enter Long", z_score, None])
            else:
                if abs(z_score) < self.z_exit:
                    self.active_positions[(i, j)] = 0
                    self.positions[i] = 0
                    self.positions[j] = 0
                    if active == 1:
                        # Long spread: i - (beta * j + alpha)
                        entry_price_i = self.entry_prices[(i, j)]['i']
                        entry_price_j = self.entry_prices[(i, j)]['j']
                        pnl = (latest_prices[i] - entry_price_i) - beta * (latest_prices[j] - entry_price_j)
                    else:
                        # Short spread
                        entry_price_i = self.entry_prices[(i, j)]['i']
                        entry_price_j = self.entry_prices[(i, j)]['j']
                        pnl = (entry_price_i - latest_prices[i]) - beta * (entry_price_j - latest_prices[j])
                    capital = abs(entry_price_i) + abs(beta * entry_price_j)
                    pct_return = pnl / capital * 100

                    self.trades.append([
                        day, i, j, "Exit", z_score, f"{pct_return:.2f}%"
                    ])
        # Plotting during run!
        # if(t % 50  == 0):
        #     self.plot_spread()
                
        return self.positions


    def run(self, prc_matrix):
        T = prc_matrix.shape[1]
        for t in range(T):
            self.step(prc_matrix[:, :t+1])

    def get_trade_log(self):
        return pd.DataFrame(self.trades, columns=["Day", "Asset A", "Asset B", "Action", "Z-Score", "Return"])

    def plot_spread(self):
        valid = ~np.isnan(self.spread_log)
        days = np.where(valid)[0]
        z_scores = self.spread_log[valid]
        raw_spreads = self.raw_spread_log[valid]

        fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Plot 1: Z-score
        axs[0].plot(days, z_scores, label="Spread Z-Score", color='orange', zorder=1)
        axs[0].axhline(self.z_entry, color='red', linestyle='--', label='Entry Threshold', zorder=0)
        axs[0].axhline(-self.z_entry, color='red', linestyle='--', zorder=0)
        axs[0].axhline(self.z_exit, color='green', linestyle='--', label='Exit Threshold', zorder=0)
        axs[0].axhline(-self.z_exit, color='green', linestyle='--', zorder=0)
        axs[0].axhline(0, color='black', linestyle=':', zorder=0)

        for day, a, b, action, z, ret in self.trades:
            if "Enter" in action:
                color = 'red' if "Short" in action else 'green'
                marker = 'v' if "Short" in action else '^'
                axs[0].scatter(day, self.spread_log[day], marker=marker, color=color, s=80, label=action, zorder=2)
            elif action == "Exit":
                axs[0].scatter(day, self.spread_log[day], marker='D', color='black', s=60, label='Exit', zorder=3)

        axs[0].set_ylabel("Z-Score")
        axs[0].set_title("Z-Score of Spread Over Time")
        axs[0].grid(True)
        handles, labels = axs[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axs[0].legend(by_label.values(), by_label.keys())

        # Plot 2: Raw Spread
        axs[1].plot(days, raw_spreads, label="Raw Spread", color='blue', zorder=1)
        axs[1].axhline(np.mean(raw_spreads), color='black', linestyle='--', label='Mean Spread', zorder=0)

        for day, a, b, action, z, ret in self.trades:
            if "Enter" in action and not np.isnan(self.raw_spread_log[day]):
                color = 'red' if "Short" in action else 'green'
                marker = 'v' if "Short" in action else '^'
                axs[1].scatter(day, self.raw_spread_log[day], marker=marker, color=color, s=80, label=action, zorder=2)
            elif action == "Exit":
                axs[1].scatter(day, self.raw_spread_log[day], marker='D', color='black', s=60, label='Exit', zorder=3)

        axs[1].set_ylabel("Spread Value")
        axs[1].set_title("Actual Spread Over Time")
        axs[1].grid(True)
        handles, labels = axs[1].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axs[1].legend(by_label.values(), by_label.keys())

        axs[1].set_xlabel("Day")
        plt.tight_layout()
        plt.show()

    def current_positions(self):
        return {pair: pos for pair, pos in self.positions.items() if pos != 0}
    
    def plot_trade_returns_histogram(self, bins=20):
        """
        Plots a histogram of trade returns (in percentage).
        
        Parameters:
            bins (int): Number of histogram bins
        """
        if not self.trades:
            print("No trades logged.")
            return

        # Extract only 'Exit' trades with returns
        returns = [trade[5] for trade in self.trades if trade[3] == "Exit" and len(trade) > 5]

        if not returns:
            print("No return data available in trades.")
            return

        plt.figure(figsize=(8, 4))
        plt.hist(returns, bins=bins, color='steelblue', edgecolor='black')
        plt.axvline(np.mean(returns), color='red', linestyle='--', label=f"Mean = {np.mean(returns):.2f}%")
        plt.title("Histogram of Trade Returns (%)")
        plt.xlabel("Return (%)")
        plt.ylabel("Number of Trades")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_trade_returns_violin(self):
        """
        Plots a violin plot of trade returns to visualize distribution and density.
        """
        if not self.trades:
            print("No trades logged.")
            return

        returns = [trade[5] for trade in self.trades if trade[3] == "Exit" and len(trade) > 5]
        if not returns:
            print("No return data available.")
            return

        sns.violinplot(data=returns, inner='quartile', color='skyblue')
        plt.title("Violin Plot of Trade Returns (%)")
        plt.ylabel("Return (%)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()