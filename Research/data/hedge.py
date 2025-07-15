import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

fn="../prices.txt"
df=pd.read_csv(fn, sep='\s+', header=None, index_col=None)
prices = (df.values).T
market_value = prices.sum(0)

nInst,T = prices.shape

returns = np.diff(prices)
for i in range(50):
    returns[i,:] /= prices[i,1:]

returns*=100

# returns = np.where(np.abs(returns) > 3,returns,0)
print(returns)

returns_rank = [np.arange(50)]

for i in range(T-1):
    returns_rank.append(np.argsort(returns[:,i]))
returns_rank = np.stack(returns_rank)

volume_rank = []
for i in range(T-1):
    volume_rank.append(np.argsort(prices[:,i]))
volume_rank = np.stack(volume_rank)

# plt.plot(market_value,label="market_value")
# plt.plot(prices.T,color="red",alpha=0.3)
# plt.legend()
plt.figure(figsize=(10,5))
plt.plot(returns_rank[:,:],marker='o',alpha=0.3,linewidth=3)
# plt.plot(volume_rank[:,:5],marker='o',alpha=0.3,linewidth=0.05)
plt.show()