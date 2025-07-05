import numpy as np
from collections import defaultdict

def get_pivot_highs_lows(prices: np.ndarray, left: int = 5, right: int = 5):
    """
    Collects all pivot highs and lows in the price series.
    
    Returns:
    - List of tuples: (index, price, 'high' or 'low')
    """
    pivots = []

    for i in range(left, len(prices) - right):
        is_high = all(prices[i] > prices[i - j] for j in range(1, left + 1)) and \
                  all(prices[i] > prices[i + j] for j in range(1, right + 1))
        if is_high:
            pivots.append((i, prices[i], 'high'))

        is_low = all(prices[i] < prices[i - j] for j in range(1, left + 1)) and \
                 all(prices[i] < prices[i + j] for j in range(1, right + 1))
        if is_low:
            pivots.append((i, prices[i], 'low'))

    return pivots

def cluster_levels(pivots, tolerance=0.01):
    """
    Clusters pivot prices into support/resistance levels within a given tolerance.
    
    Parameters:
    - pivots: list of (index, price, type)
    - tolerance: max price difference to be in same cluster (as fraction of price)

    Returns:
    - support_levels, resistance_levels (lists of prices)
    """
    levels = defaultdict(list)

    for _, price, typ in pivots:
        matched = False
        for key in levels:
            if abs(price - key) / key < tolerance:
                levels[key].append(price)
                matched = True
                break
        if not matched:
            levels[price].append(price)

    supports = []
    resistances = []

    for key, cluster in levels.items():
        avg_price = np.mean(cluster)
        if len(cluster) >= 2:
            if np.mean([p[2] == 'low' for p in pivots if abs(p[1] - avg_price) / avg_price < tolerance]) > 0.5:
                supports.append(avg_price)
            else:
                resistances.append(avg_price)

    return sorted(supports), sorted(resistances)

def detect_support_resistance(prices: np.ndarray, left: int = 5, right: int = 5, tolerance=0.01):
    """
    Main function to detect support and resistance using pivot clustering.

    Parameters:
    - prices: 1D numpy array of prices
    - left/right: pivot detection window (strictness of the pivot)
    - tolerance: price clustering tolerance (grouping similar pivots into a single support/resistance point)

    Returns:
    - supports, resistances: Lists of price levels
    """
    pivots = get_pivot_highs_lows(prices, left, right)
    supports, resistances = cluster_levels(pivots, tolerance)
    return supports, resistances

def plot_support_resistance(prices, supports, resistances):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(prices, label='Price', color='black')

    for level in supports:
        plt.axhline(level, color='green', linestyle='--', alpha=0.5, label='Support' if level == supports[0] else "")

    for level in resistances:
        plt.axhline(level, color='red', linestyle='--', alpha=0.5, label='Resistance' if level == resistances[0] else "")

    plt.title("Support and Resistance Levels")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
