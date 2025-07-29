from ast import mod
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

fn="../prices.txt"
df=pd.read_csv(fn, sep='\s+', header=None, index_col=None)
prices = (df.values).T

nInst ,T = 50,750
# Simulate a 1D system: constant velocity + noise

instID = 0
instPrice = prices[instID]

window_length = 20

transition_table = []
score_table = np.array([])

mean_start = 0
mean_start_table = [mean_start]*window_length
for i in range(window_length+1,len(instPrice)):
    current_price = instPrice[i]
    price_window = instPrice[i-window_length-1:i]
    model = LinearRegression(fit_intercept=True)
    model.fit(np.linspace(i-window_length-1,i,window_length+1).reshape(-1,1),price_window)

    score = model.coef_
    score_table = np.append(score_table,score)
    
    std = score_table[max(0,mean_start-window_length//2):].std()
    scorezerocs = 0
    if std != 0:
        scorezerocs = ( score - score_table[max(0,mean_start-window_length//2):].mean() )/std

    scaled_z = np.abs(scorezerocs/3)
    if scaled_z > 1:
        scaled_z = 1
    
    if scaled_z > 0.7:
        print(i,scaled_z)
        transition_table.append((i,current_price,scaled_z))
        mean_start = i
    mean_start_table.append(mean_start)
    

    
for t in transition_table:
    plt.scatter(t[0],t[1],alpha=t[2])

lag_mean = []
stability_start = []
for i in range(1,len(instPrice)):
    lag_mean.append(np.mean(instPrice[mean_start_table[i-1]-1:i]))

plt.plot(instPrice)
plt.plot(lag_mean)
plt.show()
