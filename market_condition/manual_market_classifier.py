import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.stats import percentileofscore

def interactive_labeler(close):
    labels = np.full(len(close), 'unlabeled', dtype=object)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(close, color='black', label='Close Price')
    ax.set_title("Interactive Market Regime Labeling")
    ax.set_xlabel("Day")
    ax.set_ylabel("Price")
    ax.grid(True)

    selection = {'start': None, 'end': None}
    colored_regions = []

    def on_press(event):
        if event.inaxes != ax: return
        selection['start'] = int(event.xdata)

    def on_release(event):
        if event.inaxes != ax: return
        selection['end'] = int(event.xdata)

        i, j = sorted((selection['start'], selection['end']))
        label = input(f"Label days {i} to {j} as (bullish/bearish/stagnant): ").strip().lower()

        if label in ['bullish', 'bearish', 'stagnant']:
            labels[i:j+1] = label
            color = {'bullish': 'green', 'bearish': 'red', 'stagnant': 'gray'}[label]
            patch = ax.axvspan(i, j, color=color, alpha=0.3)
            colored_regions.append(patch)
            fig.canvas.draw()
        else:
            print("Invalid label. Try again.")

    cid_press = fig.canvas.mpl_connect('button_press_event', on_press)
    cid_release = fig.canvas.mpl_connect('button_release_event', on_release)

    plt.legend()
    plt.show()

    return labels

### IMPORTING AND CHECKING ###
def loadPrices(fn):
    global nt, nInst
    df=pd.read_csv(fn, sep=r'\s+', header=None, index_col=None)
    (nt,nInst) = df.shape
    return (df.values).T

def main():
    pricesFile="prices.txt"
    prcAll = loadPrices(pricesFile)
    inst = 0
    labels = interactive_labeler(prcAll[inst, :])
main()
