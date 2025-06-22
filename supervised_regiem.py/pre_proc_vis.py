import pandas as pd
import matplotlib.pyplot as plt

# ─── Load your saved features and prices ─────────────────────────────────
tab_df = pd.read_csv(
    "features_heavy1.csv",
    index_col=[0, 1],    # restore the (inst, time) MultiIndex
    header=0
)
tab_df.index.names = ["inst", "time"]

price_df = pd.read_csv("prices.txt", sep=r"\s+", header=None).iloc[:500]

# ─── Pick one instrument and line things up ───────────────────────────────

def plot_inst_feature(inst: int, feat_idx: int):
    """
    Plot the close price and a single indicator for one instrument.
    inst: instrument index (0–49)
    feat_idx: index into the list of features (0 = first feature, 1 = second, …)
    """
    # slice out that instrument
    inst_df = tab_df.xs(inst, level="inst")
    time    = inst_df.index
    
    # grab and align the price
    price_series  = price_df[inst].reindex(time).values
    
    # pick the feature name by integer index
    feats = list(inst_df.columns)
    feat = feats[feat_idx]
    feat_series = inst_df[feat].values
    
    # build the plot
    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()
    ax1.plot(time, price_series, color="black", label="Close Price")
    ax1.set_ylabel("Price")
    
    ax2 = ax1.twinx()
    ax2.plot(time, feat_series, color="tab:blue", label=feat)
    ax2.set_ylabel(feat)
    
    plt.title(f"Instrument {inst} — {feat}")
    # combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example usage:
plot_inst_feature(0, 47)
#Instrument, feature
