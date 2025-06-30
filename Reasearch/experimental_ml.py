import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# ─── 0) Hyperparameters ─────────────────────────────────────────────────────
n_train_days = 500
n_test_days  = 250
inst         = 1          # 0-based index of the instrument/column you want to model
arima_order  = (5,1,0)    # (p,d,q)

# ─── 1) Load & split ───────────────────────────────────────────────────────
df         = pd.read_csv("prices.txt", sep=r"\s+", header=None)
data       = df.values
assert data.shape[0] >= n_train_days + n_test_days, "Need ≥750 rows"

train_full = data[:n_train_days, inst]
test_full  = data[n_train_days : n_train_days + n_test_days, inst]

# ─── 2) Fit ARIMA on the training series ────────────────────────────────────
arima_model = ARIMA(train_full, order=arima_order)
arima_fit   = arima_model.fit()
print(arima_fit.summary())

# ─── 3) In-sample (training) MSE ────────────────────────────────────────────
# 'predict' with typ='levels' gives you back the series in price‐units
train_pred = arima_fit.predict(start=0, end=n_train_days-1, typ='levels')
train_mse  = mean_squared_error(train_full, train_pred)
print(f"Train MSE: {train_mse:.6f}")

# ─── 4) Rolling one-step forecasts on the 250-day test window ───────────────
history = list(train_full)
preds   = []

for t in range(n_test_days):
    # fit on all available history each step (small data!)
    step_model = ARIMA(history, order=arima_order).fit()
    yhat       = step_model.forecast()[0]    # one‐step ahead forecast
    preds.append(yhat)
    history.append(test_full[t])             # append the true price

# ─── 5) Test MSE ─────────────────────────────────────────────────────────────
test_mse = mean_squared_error(test_full, preds)

naive_preds = np.concatenate(([train_full[-1]], test_full[:-1]))
print("Naive MSE:", mean_squared_error(test_full, naive_preds))
print("ARIMA MSE:", test_mse)

'''
# ─── 6) Plot Predicted vs Actual ────────────────────────────────────────────
days = np.arange(n_train_days, n_train_days + n_test_days)

plt.figure(figsize=(10,5))
plt.plot(days, test_full, label="Actual Price")
plt.plot(days, preds,     linestyle="--", label="ARIMA Predicted")
plt.xlabel("Day index")
plt.ylabel("Price")
plt.title(f"ARIMA({arima_order}) Forecast vs Actual — Instrument {inst}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

'''
