import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

fn="../prices.txt"
df=pd.read_csv(fn, sep='\s+', header=None, index_col=None)
prices = (df.values).T

ynInst ,T = 50,75
inst = 33
instrPrice = prices[inst]

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    x = np.asarray(x)
    exps = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exps / np.sum(exps)


    
max_freq_pred = []

density_mat = []
pmax = instrPrice.max()
pmin = instrPrice.min()
padding = (pmax - pmin) * 0.1
price_grid = np.linspace(pmin - padding, pmax + padding, 500)

lookback = 0
for i in range(1,len(instrPrice)):
    x_current = instrPrice[i]
    x_prev = instrPrice[lookback:i]
    diff = x_current-x_prev
    tdiff = list(reversed(range(1,len(x_prev)+1)))
    gradient = diff/tdiff
    p = gradient.mean()

    
    x_next = gradient + x_current 
    x_next = np.round(x_next,2)

    
    # # Softmax PMF
    scaled = np.round(x_next * 100).astype(int)
    min_val = scaled.min()
    max_val = scaled.max()
    full_range = np.arange(min_val, max_val + 1)/100
    counts = np.bincount(scaled)[min_val:]
    idx = np.argsort(counts)
    max_freq_pred.append(np.round(full_range[idx[-1]],decimals=1))
    prob = softmax(counts)

    if i == 1:
        continue
    # Gaussian KDE
    # kde = gaussian_kde(instrPrice[max(i-10,0):i+1])
    kde = gaussian_kde(x_next)
    density = kde(price_grid)
    y = density/np.sum(density)
    density_mat.append(y)

density_mat = np.array(density_mat).T
plt.figure(figsize=(20,10))
plt.plot(instrPrice,alpha=0.4)
plt.plot(range(1,750),max_freq_pred,alpha=0.4)
plt.imshow(
    density_mat,
    aspect='auto',
    origin='lower',
    extent=[1, 751, instrPrice.min(), instrPrice.max()],
    cmap='viridis'
)
plt.show()