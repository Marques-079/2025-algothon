import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from NN import RegressionNetWork

fn="../prices.txt"
df=pd.read_csv(fn, sep='\s+', header=None, index_col=None)
prices = (df.values).T

ynInst ,T = 50,75
inst = 42
instrPrice = prices[inst]

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    x = np.asarray(x)
    exps = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exps / np.sum(exps)


    
max_freq_pred = []
network_pred = []

lookback = 0

network = RegressionNetWork()

for i in range(1,len(instrPrice)):
    x_current = instrPrice[i]
    lookback = max( i-10, 0)
    x_prev = instrPrice[lookback:i]
    X = np.array(range(lookback,i+1))
    network.train_1(X,instrPrice[lookback:i+1],i)
    network_pred.append(network.predict(i+1)[0])
    
    #--------------------conventional methods--------------  
    diff = x_current-x_prev
    tdiff = list(reversed(range(1,len(x_prev)+1)))
    gradient = diff/tdiff

    
    x_next = gradient + x_current 
    x_next = np.round(x_next,2)

    
    # # Softmax PMF
    scaled = np.round(x_next * 100).astype(int)
    min_val = scaled.min()
    max_val = scaled.max()
    full_range = np.arange(min_val, max_val + 1)/100
    counts = np.bincount(scaled)[min_val:]

    idx = np.argsort(counts)
    max_val = full_range[idx[-1]]
    max_val = np.round(max_val,decimals=1) 
    
    max_freq_pred.append(max_val)
    


plt.figure(figsize=(20,10))
plt.plot(instrPrice,label="true")
plt.plot(range(2,len(instrPrice)+1),max_freq_pred,label="forecast max")
x_range = range(len(instrPrice)-len(network_pred)+1,len(instrPrice)+1)
plt.plot(x_range,network_pred,label="network")
plt.legend()
plt.show()