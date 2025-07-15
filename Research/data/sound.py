from scipy.signal import savgol_filter
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt
fn="../prices.txt"

df=pd.read_csv(fn, sep='\s+', header=None, index_col=None)
prices = (df.values).T

nInst ,T = 50,750
# Simulate a 1D system: constant velocity + noise

instID = 33
instPrice = prices[instID]

denoised = savgol_filter(instPrice, window_length=10, polyorder=5)

focus_length = 1000
period = 2* np.pi
terms = 1000

# Generate time variable and target

# Fourier + Polynomial Feature Generator
def make_fourier(t, T, K):
    features = [np.ones_like(t)]  # bias term
    # Fourier terms
    for k in range(1, K + 1):
        features.append(np.sin(2*np.pi*k*t/T))
        features.append(np.cos(2*np.pi*k*t/T))
    return np.column_stack(features)
#
# --- EMA calculation (manual) ---
def compute_ema(series, span):
    alpha = 2 / (span + 1)
    ema = np.zeros_like(series)
    ema[0] = series[0]  # Initialize with first value
    for t in range(1, len(series)):
        ema[t] = alpha * series[t] + (1 - alpha) * ema[t-1]
    return ema


# Params
T = period
K = terms

t = np.linspace(1, focus_length-1, focus_length-1)
y = np.diff(denoised[:focus_length])
y_ema = compute_ema(y,span=10)
y -= y_ema
# plt.plot(y_ema)

#Split data into train/test
window = (0,750)
t_train = t[window[0]:window[1]]
y_train = y[window[0]:window[1]]
X_train = make_fourier(t_train, T, K)

t_test = t[window[1]:]
y_test = y[window[1]:]
X_test = make_fourier(t_test, T, K)

# Fit on training data
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


# results = []
# x_ax = range(window[1],999)
# for i in x_ax:
#     tp = t[:i]
#     yp = y[:i]
#     Xt = make_fourier(tp, T, K)
#     model = LinearRegression()
#     model.fit(Xt, yp)
#     X_test = make_fourier(i+1, T, K)
#     y_test_pred = model.predict(X_test)
#     results.append(y_test_pred)
#     print(i)

# plt.plot(x_ax,results)
# plt.plot(denoised,alpha=0.1)
# plt.plot(instPrice,alpha=0.1)
# plt.show()


# Evaluate
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Train MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
print(f"Test MSE:  {test_mse:.4f}, R²: {test_r2:.4f}")

# Plot predictions
# plt.figure(figsize=(12,5))
plt.plot(t, y, label='True', linewidth=1)
plt.plot(t_train, y_train_pred, label='Train Pred', linestyle='--', color='blue')
plt.plot(t_test, y_test_pred, label='Test Pred', linestyle='--', color='orange')
plt.axvline(t[window[1]], linestyle=':', color='black', label='Train/Test Split')
plt.title("Fourier + Polynomial Regression with Train/Test Split")
plt.legend()
plt.tight_layout()
plt.show()