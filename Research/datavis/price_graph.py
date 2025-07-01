#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

nInst = 0
nt = 0
commRate = 0.0005
dlrPosLimit = 10000

def loadPrices(fn):
    global nt, nInst
    df=pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    return df

def plot_dataframe(df,r1,r2, title="DataFrame Plot"):

    plt.figure(figsize=(12, 6))
    for column in df.columns[r1:r2]:
        plt.plot(df.index, df[column], label=column)

    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Values")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


pricesFile="../prices.txt"
prcAll = loadPrices(pricesFile)
print(prcAll)