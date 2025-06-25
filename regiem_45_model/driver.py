import numpy as np
import pandas as pd
from regiem_45_model.online_inference import OnlineModel

'''
python -m regiem_45_model.driver
'''

prices = np.loadtxt("prices.txt").T      
n_inst, nt = prices.shape
print("prices shape =", prices.shape)

model = OnlineModel(n_inst=n_inst)

model.warmup(prices, till_t=159)
print("✅ Warmup complete (price+feature buffers full at t=159)")


inst_preds = [[] for _ in range(n_inst)]
time_steps = []

for t in range(160, nt):
    print(f"Processing t={t}... ")
    probs = model.update(prices[:, t])
    if probs is not None:
        time_steps.append(t)
        for i, p in enumerate(probs):
            inst_preds[i].append(p)

P = np.vstack([np.array(lst) for lst in inst_preds])
print("Collected predictions shape:", P.shape) 

df = pd.DataFrame(
    data    = P,
    index   = [f"inst_{i}"    for i in range(n_inst)],
    columns = [str(t)          for t in time_steps]
)

csv_path = "predictions.csv"
df.to_csv(csv_path)
print(f"✅ Predictions saved to {csv_path}")




#--------------------------------------------------------------------------------#
# import numpy as np
# import pandas as pd
# from regiem_45_model.online_inference import OnlineModel

# prices = np.loadtxt("prices.txt").T      # (50, 750)  ← transpose!
# print("prices shape =", prices.shape)   
# model  = OnlineModel(n_inst=prices.shape[0])

# model.warmup(prices, till_t=60)         # fills caches
# print('Price Buffers filled, waiting for Features Buffer...')

# for t in range(61, prices.shape[1]):
#     probs = model.update(prices[:, t])
#     if probs is not None:
#         print(probs.shape)        
#         print(probs[:5])

