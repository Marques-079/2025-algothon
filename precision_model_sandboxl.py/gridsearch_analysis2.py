import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from precision_labeller import plot_all_regimes_long  # must exist locally

def compute_zscore_regimes(prices: pd.Series, window: int, threshold: float, drop_last: int = 10):
    ma = prices.rolling(window=window, min_periods=window).mean()
    sd = prices.rolling(window=window, min_periods=window).std()
    z = (prices - ma) / sd

    regs_full = np.where(z > threshold, 2,
                         np.where(z < -threshold, 0, 1))

    N = len(prices) - drop_last
    regimes = regs_full[:N]
    z_trim = z[:N]

    valid = ~np.isnan(z_trim)
    return regimes[valid], valid

def shared_param_grid_search(price_file: str, window_range: range, threshold_range: list, drop_last: int = 10, n_inst: int = 50):
    df = pd.read_csv(price_file, sep=r"\s+", header=None)
    records = []
    best_acc = 0.0
    cutoff = 500  # Use only the first 500 timesteps

    for w in window_range:
        for th in threshold_range:
            print(f"\n🔍 Testing params: window={w}, threshold={th}")
            all_preds, all_trues = [], []

            for inst in range(n_inst):
                prices = df.iloc[:cutoff, inst]
                true_regs = plot_all_regimes_long(cutoff, plot_graph=False, inst=inst)[:cutoff - drop_last]

                pred, valid = compute_zscore_regimes(prices, w, th, drop_last)
                true_trimmed = np.array(true_regs)[valid]

                mask = pred != 1
                pred_bin = pred[mask]
                true_bin = true_trimmed[mask]

                print(f"  • Inst {inst:02d} → compared: {len(pred_bin)} samples")

                if len(pred_bin) > 0:
                    all_preds.extend(pred_bin)
                    all_trues.extend(true_bin)

            if len(all_preds) > 0:
                acc = accuracy_score(all_trues, all_preds)
                print(f"  ✅ Total samples compared: {len(all_preds)}, Accuracy: {acc:.4f}")

                if acc > best_acc:
                    print(f"  🎯 New best so far → window={w}, threshold={th}, acc={acc:.4f}")
                    best_acc = acc

                records.append({
                    "window": w,
                    "threshold": th,
                    "accuracy": acc,
                    "samples": len(all_preds)
                })
            else:
                print("  ⚠️  No valid predictions for this parameter set.")

    results = pd.DataFrame(records).sort_values("accuracy", ascending=False)
    return results

if __name__ == "__main__":
    windows = range(80, 106, 2)
    thresholds = [0.5, 1.0, 1.5, 2.0]
    results = shared_param_grid_search("prices.txt", windows, thresholds, drop_last=10, n_inst=50)

    print("\n📊 Top shared Z-score parameters:")
    print(results.head(10).to_string(index=False))

    best = results.iloc[0]
    print(f"\n✅ Best Params → window={best.window}, threshold={best.threshold}, acc={best.accuracy:.4f}")

    print("\n📋 Full Summary of All Hyperparameter Combinations:")
    print(results.to_string(index=False))
