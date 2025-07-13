import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fn="../prices.txt"
df=pd.read_csv(fn, sep='\s+', header=None, index_col=None)
prices = (df.values).T

nInst ,T = 50,750
# Simulate a 1D system: constant velocity + noise

instID = 15
instPrice = prices[instID]


stop = 501
measurements = instPrice[:stop]

# Kalman filter setup
x = 0.0           # initial estimate
v = 0.5           # initial velocity
P = 1.0           # initial uncertainty
Q = 0.05         # process noise (model uncertainty)
R = 4           # measurement noise
A = 1.0           # state transition (no change to state, just incremented by v)
H = 1.0           # measurement matrix
x_estimates = []

for z in measurements:
    # Predict step
    x = x + v               # predict next state
    P = P + Q               # update uncertainty

    # Update step
    v = P * (z-x) 
    K = P / (P + R)          # Kalman gain
    x = x + K * (z - x) + v/2     # update estimate
    P = (1 - K) * P         # update uncertainty

    x_estimates.append(x)

# Forecast future steps
n = len(measurements)
future_steps = 10
forecast = [x + v * i for i in range(1, future_steps + 1)]

# --- EMA calculation (manual) ---
def compute_ema(series, span):
    alpha = 2 / (span + 1)
    ema = np.zeros_like(series)
    ema[0] = series[0]  # Initialize with first value
    for t in range(1, len(series)):
        ema[t] = alpha * series[t] + (1 - alpha) * ema[t-1]
    return ema

emas = []
N = 50
for i in range(0,N):
    emas.append(compute_ema(instPrice,span=2*( i+1 )))
emas = np.array(emas).T
convergence_measure = np.mean(emas,axis=1)
stdT = np.std(emas,axis=1) 
stdB = -stdT
std = np.stack([stdT+convergence_measure,stdB+convergence_measure])
print(std.shape)

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(instPrice, label='Noisy Measurements', alpha=0.5)
plt.plot(emas,alpha=0.1,color="red")
plt.plot(convergence_measure,label="convergence measure")
plt.plot(std.T,color="yellow")
# plt.plot(x_estimates, label='Kalman Estimate', linewidth=2)
plt.plot(np.arange(n, n+future_steps), forecast, label='Forecast', linewidth=2)
plt.legend()
plt.title("Kalman Filter Forecasting")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.grid(True)
plt.show()


