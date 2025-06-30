import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.stats import percentileofscore, pearsonr, spearmanr
import seaborn as sns
from statsmodels.stats.multitest import fdrcorrection
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.stattools import grangercausalitytests, adfuller, coint
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.regression.linear_model import OLS
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
import statsmodels.api as sm
import os
from matplotlib.cm import coolwarm, get_cmap
from statsmodels.tools.tools import add_constant
from matplotlib.colors import Normalize
from collections import defaultdict

# Other functions
def loadPrices():
    fn="../prices.txt"
    global nt, nInst
    df=pd.read_csv(fn, sep=r'\s+', header=None, index_col=None)
    (nt,nInst) = df.shape
    return nt, nInst, (df.values).T


def average_windowed_correlation(prc_matrix, 
                                  window_size=7,
                                  method='pearson', 
                                  use_log_returns=True):
    """
    Computes an average correlation matrix by converting daily prices to windowed data 
    with multiple offsets (0 to window_size - 1).

    Parameters:
        prc_matrix (ndarray): (n_instruments, n_days) daily price matrix
        window_size (int): number of days in each window (e.g. 7 for weekly, 30 for monthly)
        method (str): 'pearson' or 'spearman'
        use_log_returns (bool): Use log returns (True) or percent returns (False)

    Returns:
        avg_corr_matrix (ndarray): Averaged correlation matrix (n_instruments x n_instruments)
    """
    n_instruments, n_days = prc_matrix.shape
    all_corrs = []

    for offset in range(window_size):
        # Step 1: Resample using window size and offset
        idx = np.arange(offset, n_days, window_size)
        if len(idx) < 3:
            continue  # skip if not enough data points

        windowed_prices = prc_matrix[:, idx]

        # Step 2: Compute returns
        if use_log_returns:
            returns = np.diff(np.log(windowed_prices), axis=1)
        else:
            returns = np.diff(windowed_prices, axis=1) / windowed_prices[:, :-1]

        # Step 3: Compute correlation matrix
        corr_matrix = np.zeros((n_instruments, n_instruments))
        for i in range(n_instruments):
            for j in range(n_instruments):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    if method == 'pearson':
                        r, _ = pearsonr(returns[i], returns[j])
                    elif method == 'spearman':
                        r, _ = spearmanr(returns[i], returns[j])
                    else:
                        raise ValueError("method must be 'pearson' or 'spearman'")
                    corr_matrix[i, j] = r

        all_corrs.append(corr_matrix)

    avg_corr_matrix = np.mean(all_corrs, axis=0)
    return avg_corr_matrix

def compute_linkage_matrix(corr_matrix, use_abs=True, method='ward'):
    """
    Converts a correlation matrix into a linkage matrix for hierarchical clustering.

    Parameters:
        corr_matrix (ndarray): correlation matrix
        method (str): linkage method (default 'ward')

    Returns:
        linkage_matrix
    """
    if use_abs:
        distance_matrix = 1 - np.round(abs(corr_matrix), decimals=10)
    else:
        distance_matrix = 1 - np.round(corr_matrix, decimals=10)
    return linkage(squareform(distance_matrix), method=method)

def get_clusters(linkage_matrix, cutoff):
    """
    Cuts a linkage matrix into clusters.

    Parameters:
        linkage_matrix: output from `linkage()`
        cutoff (float): distance threshold

    Returns:
        ndarray of cluster labels
    """
    return fcluster(linkage_matrix, t=cutoff, criterion='distance')

def group_instruments_by_cluster(clusters):
    """
    Groups instrument indices by their cluster labels.

    Parameters:
        clusters (array-like): Cluster labels for each instrument (length = n_instruments)

    Returns:
        grouped (list of lists): Each sublist contains indices belonging to the same cluster
    """
    cluster_dict = defaultdict(list)
    for idx, cluster_id in enumerate(clusters):
        cluster_dict[cluster_id].append(idx)
    return list(cluster_dict.values())

def plot_clustered_correlation_matrix(corr_matrix, clusters):
    """
    Plots a heatmap of the correlation matrix sorted by cluster.

    Parameters:
        corr_matrix (ndarray): correlation matrix
        clusters (ndarray): cluster labels
    """
    sorted_idx = np.argsort(clusters)
    sorted_corr = corr_matrix[sorted_idx][:, sorted_idx]

    plt.figure(figsize=(10, 8))
    sns.heatmap(sorted_corr, cmap='coolwarm', vmin=-1, vmax=1, square=True)
    plt.title("Correlation Matrix (Clustered)")
    plt.tight_layout()
    plt.show()

def plot_dendrogram(linkage_matrix, cutoff=1.0):
    """
    Plots a dendrogram from a linkage matrix.

    Parameters:
        linkage_matrix: output from `linkage()`
        cutoff (float): distance threshold for coloring clusters
    """
    plt.figure(figsize=(12, 6))
    dendrogram(linkage_matrix, no_labels=True, color_threshold=cutoff)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Instrument Index")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.show()

def compute_atr(prices, window=14):
    """
    Compute Average True Range (ATR) for each price series.
    Assumes prices is a (n_instruments, n_days) ndarray of close prices.
    """
    highs = prices[:, 1:]
    lows = prices[:, :-1]
    closes = prices[:, :-1]
    
    tr = np.abs(highs - lows)  # approximate true range as abs(diff)
    atr = np.zeros_like(prices)
    atr[:, window:] = np.apply_along_axis(
        lambda x: np.convolve(x, np.ones(window)/window, mode='valid'), 
        axis=1, arr=tr
    )
    atr[:, :window] = np.nan  # pad initial values
    return atr

def volatility_normalized_prices(prices, atr_window=14, use_log=False):
    """
    Normalize price series by ATR.
    Returns volatility-adjusted cumulative return.
    """
    if use_log:
        log_returns = np.diff(np.log(prices), axis=1)
    else:
        log_returns = np.diff(prices, axis=1)
    atr = compute_atr(prices, window=atr_window)[:, 1:]  # align shape

    # Avoid divide by zero
    atr[atr == 0] = np.nan
    norm_returns = log_returns / atr

    # Cumulative sum to reconstruct a path
    norm_prices = np.nancumsum(norm_returns, axis=1)
    norm_prices = np.hstack([np.zeros((prices.shape[0], 1)), norm_prices])  # add zero at start

    return norm_prices

def log_returns_normalized_prices(prices):
    log_returns = np.log(prices)
    transformed = log_returns - log_returns[:, [0]]
    return transformed

def z_score_normalized_prices(prices):
    return (prices- prices.mean(axis=1, keepdims=True)) / prices.std(axis=1, keepdims=True)

def percent_change_normalized_prices(prices):
    return prices/ prices[:, [0]]

def plot_pca_projection(pca_coords, clusters):
    n_clusters = len(np.unique(clusters))
    # Step 5: Plot PCA with cluster coloring
    plt.figure(figsize=(10, 8))
    palette = sns.color_palette("hsv", n_clusters)
    for cluster_id in range(1, n_clusters + 1):
        idx = np.where(clusters == cluster_id)[0]
        plt.scatter(
            pca_coords[idx, 0], pca_coords[idx, 1],
            s=80, edgecolor='k', label=f"Cluster {cluster_id}",
            color=palette[cluster_id - 1]
        )
        for i in idx:
            plt.text(pca_coords[i, 0], pca_coords[i, 1], str(i), fontsize=8,
                     ha='center', va='center', color='white', weight='bold')

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Projection Colored by Correlation Clusters")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_cluster_all_normalizations(
    data,
    cluster_labels,
    cluster_id,
    save_plot=False,
    output_dir="cluster_all_normalizations",
    tickers=None,
    show_volatility_band=True
):
    """
    Plots a single cluster's price series using 4 normalization methods:
    - Log Percent Change from first value
    - Z-Score normalization
    - ATR Normalised
    - ATR Log Normalised
    Each with a cluster average line and optional volatility band.

    Parameters:
        data (ndarray): (n_instruments, n_days) price matrix
        cluster_labels (ndarray): cluster ID array (len = n_instruments)
        cluster_id (int): The cluster ID to plot
        save_plot (bool): If True, saves the plot to disk
        output_dir (str): Folder to save plots in
        tickers (list of str): Optional list of instrument names
        show_volatility_band (bool): If True, display ±1 std band around cluster average
    """
    idx = np.where(cluster_labels == cluster_id)[0]
    if len(idx) <= 1:
        print(f"Cluster {cluster_id} skipped (singleton or empty).")
        return

    cluster_data = data[idx]
    if tickers is not None:
        tickers = [tickers[i] for i in idx]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Cluster {cluster_id} - Normalization Comparison with Average and Volatility Band", fontsize=16)

    norm_methods = ["Percent Change", "Z-Score", "ATR Change", "ATR Log Change"]

    for k, method in enumerate(norm_methods):
        ax = axes[k // 2, k % 2]

        if method == "Percent Change":
            transformed = percent_change_normalized_prices(cluster_data)
        elif method == "Z-Score":
            transformed = z_score_normalized_prices(cluster_data)
        elif method == "ATR Change":
            transformed = volatility_normalized_prices(cluster_data, 14, False)
        elif method == "ATR Log Change":
            transformed = volatility_normalized_prices(cluster_data, 14, True)
            # returns = np.diff(cluster_data, axis=1) / cluster_data[:, :-1]
            # returns = np.hstack([np.zeros((returns.shape[0], 1)), returns])
            # transformed = np.cumsum(returns, axis=1)

        # Plot individual lines
        for i, series in enumerate(transformed):
            label = tickers[i] if tickers else f"Instrument {idx[i]}"
            ax.plot(series, label=label, linewidth=1.2)

        # Cluster average
        cluster_avg = np.nanmean(transformed, axis=0)
        ax.plot(cluster_avg, label="Cluster Average", linewidth=2.5, color="black", zorder=10)

        # Volatility band (±1 std)
        if show_volatility_band:
            cluster_std = np.nanstd(transformed, axis=0)
            ax.fill_between(
                np.arange(transformed.shape[1]),
                cluster_avg - cluster_std,
                cluster_avg + cluster_std,
                color='gray',
                alpha=0.3,
                label="±1 Std Dev"
            )

        ax.set_title(method)
        ax.grid(True)

    # Legend (outside grid so it's shared across all subplots)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=min(len(labels), 6), bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    if save_plot:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"cluster_{cluster_id}_normalizations.png")
        plt.savefig(filename, dpi=300)
        print(f"Saved to {filename}")

    plt.show()


def plot_clusters_price_series(
    data,
    cluster_labels,
    save_plots=False,
    output_dir="cluster_plots_colored",
    show_colorbar=True,
    color_by_correlation=True,
    normalization_method="percent"
):
    """
    Plots clustered instrument price series with either:
    - heatmap coloring by correlation to cluster average, OR
    - distinct colors for each instrument (default style).

    Parameters:
        data (ndarray): (n_instruments, n_days) price matrix
        cluster_labels (ndarray): cluster ID array (len = n_instruments)
        normalize (bool): Normalize each series to start at 1
        save_plots (bool): Save plots to disk
        output_dir (str): Folder to save plots in (created if not exists)
        show_colorbar (bool): Show correlation colorbar if heatmap used
        color_by_correlation (bool): If True, use heatmap coloring
    """
    unique_clusters = np.unique(cluster_labels)

    if save_plots and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for cluster_id in unique_clusters:
        idx = np.where(cluster_labels == cluster_id)[0]
        if len(idx) <= 1:
            continue  # skip singleton clusters

        cluster_data = data[idx]

        # Normalize if requested
        if normalization_method == "percent":
            cluster_data = cluster_data / cluster_data[:, [0]]
        elif normalization_method == "zscore":
            cluster_data = (cluster_data - cluster_data.mean(axis=1, keepdims=True)) / cluster_data.std(axis=1, keepdims=True)
        elif normalization_method == "log":
            cluster_data = np.log(cluster_data)
            cluster_data = cluster_data - cluster_data[:, [0]]
        elif normalization_method == "cumulative_return":
            returns = np.diff(cluster_data, axis=1) / cluster_data[:, :-1]
            returns = np.hstack([np.zeros((returns.shape[0], 1)), returns])
            cluster_data = np.cumsum(returns, axis=1)


        avg_series = cluster_data.mean(axis=0)

        # Determine colors
        if color_by_correlation:
            corr_to_avg = []
            for series in cluster_data:
                if np.std(series) == 0 or np.std(avg_series) == 0:
                    corr = 0
                else:
                    corr = np.corrcoef(series, avg_series)[0, 1]
                corr_to_avg.append(corr)
            corr_to_avg = np.clip(corr_to_avg, -1, 1)

            cmap = coolwarm
            norm = Normalize(vmin=-1, vmax=1)
            colors = [cmap(norm(c)) for c in corr_to_avg]
        else:
            # Use a qualitative colormap for distinct colors
            cmap = get_cmap("tab10")  # good for ≤10 items; use 'tab20' for larger clusters
            colors = [cmap(i % cmap.N) for i in range(len(idx))]

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, (series, color) in zip(idx, zip(cluster_data, colors)):
            ax.plot(series, label=f"Instrument {i}", linewidth=1.2, alpha=0.85, color=color)

        ax.plot(avg_series, color='black', linewidth=3, label="Cluster Avg")

        ax.set_title(f"Cluster {cluster_id} — {len(idx)} Instruments")
        ax.set_xlabel("Time (Days)")
        ax.set_ylabel(f"{normalization_method} - Price")
        ax.grid(True)
        ax.legend()

        # Optional colorbar (only for heatmap-style)
        if color_by_correlation and show_colorbar:
            sm = plt.cm.ScalarMappable(cmap=coolwarm, norm=Normalize(vmin=-1, vmax=1))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, orientation='vertical', shrink=0.8)
            cbar.set_label('Correlation with Cluster Avg', fontsize=10)

        plt.tight_layout()

        if save_plots:
            filename = f"cluster_{cluster_id}.png"
            plt.savefig(os.path.join(output_dir, filename))
            plt.close()
        else:
            plt.show()