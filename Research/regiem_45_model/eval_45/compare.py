#!/usr/bin/env python3
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib     import Path
from regiem_45_model.pre_proc_labelling_long import plot_all_regimes_long

'''
python -m regiem_45_model.eval_45.compare
'''

N_INST  = 50     # number of instruments
T_FULL  = 750    # full history length expected by plot_all_regimes_long
START   = 160    # first timestep both true & preds overlap
END     = 739    # last   timestep both true & preds overlap

HERE = Path(__file__).parent


df_pred       = pd.read_csv(HERE / "predictions.csv", index_col=0)
pred_matrix   = df_pred.to_numpy()             # shape (50, nt_pred)

#If > X then True and will assign 2.0 (Bull) else 0.0 (Bear)
pred_regimes  = np.where(pred_matrix > 0.72, 0, 2) 


true_list = [plot_all_regimes_long(T_FULL, False, i) for i in range(N_INST)]
true2d = np.vstack(true_list)               


length       = END - START + 1
true_aligned = true2d[:,  START:END+1]         # (50, length)
pred_aligned = pred_regimes[:, :length]        # (50, length)
dates        = np.arange(START, END+1)         # x‐axis for plotting


price_df = pd.read_csv(HERE.parent / "prices.txt", sep=r"\s+", header=None)


shift = 59

p_len = pred_regimes.shape[1]
t_len = true2d.shape[1]

# overlap length after we lop off `shift` columns from the left of truth
# and from the right of predictions
overlap = min(p_len - shift, t_len - shift)

agree_A = (pred_regimes[:, :overlap]          == true2d[:, shift:shift + overlap]).mean()
agree_B = (pred_regimes[:, shift:shift+overlap] == true2d[:, :overlap]).mean()

print(f"If predictions LEAD truth by {shift}: {agree_A:.2%}")
print(f"If truth LEADS predictions by {shift}: {agree_B:.2%}")





def plot_inst(inst: int):

    prices = price_df.iloc[START:END+1, inst].values

    def shade(ax, regs):
        cur = regs[0]; seg0 = dates[0]
        for idx, r in enumerate(regs[1:], 1):
            if r != cur:
                ax.axvspan(seg0, dates[idx],
                           facecolor=("green" if cur==2 else "red"),
                           alpha=0.3)
                cur, seg0 = r, dates[idx]
        ax.axvspan(seg0, dates[-1],
                   facecolor=("green" if cur==2 else "red"),
                   alpha=0.3)

    fig, (ax1, ax2) = plt.subplots(2,1,sharex=True,figsize=(10,6))
    # True regimes
    ax1.plot(dates, prices, color="black", lw=1)
    shade(ax1, true_aligned[inst])
    ax1.set_ylabel("Price")
    ax1.set_title(f"Inst {inst+1} — TRUE regimes")

    # Pred regimes
    ax2.plot(dates, prices, color="black", lw=1)
    shade(ax2, pred_aligned[inst])
    ax2.set_ylabel("Price")
    ax2.set_xlabel("Time (Days)")
    ax2.set_title(f"Inst {inst+1} — PREDICTED regimes")

    plt.tight_layout()
    plt.show()


#plot_inst(2)
