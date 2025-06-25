import numpy as np
from regiem_45_model.online_inference import OnlineModel

prices = np.loadtxt("prices.txt").T      # (50, 750)  ← transpose!
print("prices shape =", prices.shape)   
model  = OnlineModel(n_inst=prices.shape[0])

model.warmup(prices, till_t=159)         # fills caches

for t in range(160, prices.shape[1]):
    probs = model.update(prices[:, t])
    if probs is not None:
        print(probs.shape)        
        print(probs[:5])
