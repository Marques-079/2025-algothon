from cProfile import label
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression

fn="../prices.txt"
df=pd.read_csv(fn, sep='\s+', header=None, index_col=None)
prices = (df.values).T
market_value = prices.sum(0)

nInst,T = prices.shape

returns = np.diff(prices,axis=1)
returns /= prices[:,:-1]

ret_market = returns.mean(0)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def plot_expanding_alpha_beta(returns, ret_market, min_window=50,instrumentID=0):
    """
    Calculate and plot expanding window alpha and beta for all instruments.
    
    Parameters:
        returns: np.array, shape (nInst, T), instrument returns
        ret_market: np.array, shape (T,), market returns
        min_window: int, minimum number of days before regression starts
        alpha_transparency: float, transparency for plot lines (0-1)
    """
    nInst, T = returns.shape
    times = np.arange(min_window, T)

    alphas_all = np.zeros((len(times), nInst))
    betas_all = np.zeros((len(times), nInst))

    X_full = ret_market.reshape(-1, 1)  # market returns reshaped

    for i, t in enumerate(times):
        X_exp = X_full[:t]
        Y_exp = returns[:, :t].T  # shape (t, nInst)

        model = LinearRegression(fit_intercept=True)
        model.fit(X_exp, Y_exp)

        alphas_all[i, :] = model.intercept_
        betas_all[i, :] = model.coef_.flatten()

    # Plot Alphas
    plt.subplot(2,1,1)
    plt.plot(times, alphas_all[:, instrumentID],label="alpha")
    plt.title(f'Expanding Window Alpha for {instrumentID} Instruments')
    plt.xlabel('Time (days)')
    plt.ylabel('Alpha')
    plt.legend()
    # Plot betas
    plt.subplot(2,1,2)
    plt.plot(times, betas_all[:, instrumentID],label="beta",color="orange")
    plt.title(f'Expanding Window Beta for {instrumentID} Instruments')
    plt.xlabel('Time (days)')
    plt.ylabel('Beta')
    plt.legend()



def plot_rolling_alpha_beta(returns, ret_market, instrumentID=0, window=50):
    """
    Calculate and plot rolling window alpha and beta for one instrument.
    
    Parameters:
        returns: np.array, shape (nInst, T), instrument returns
        ret_market: np.array, shape (T,), market returns
        instrumentID: int, index of instrument to plot
        window: int, rolling window size
    """
    nInst, T = returns.shape
    n_windows = T - window + 1

    betas = np.zeros(n_windows)
    alphas = np.zeros(n_windows)
    times = np.arange(n_windows) + window - 1

    for i in range(n_windows):
        X_window = ret_market[i:i+window].reshape(-1, 1)
        y_window = returns[instrumentID, i:i+window]

        model = LinearRegression(fit_intercept=True)
        model.fit(X_window, y_window)

        alphas[i] = model.intercept_
        betas[i] = model.coef_[0]

    plt.subplot(2,1,1)
    plt.plot(times, alphas, label='Alpha')
    plt.title(f'Rolling Alpha (window={window}) for instrument {instrumentID}')
    plt.xlabel('Time')
    plt.ylabel('Alpha')
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(times, betas, label='Beta')
    plt.title(f'Rolling Beta (window={window}) for instrument {instrumentID}')
    plt.xlabel('Time')
    plt.ylabel('Beta')
    plt.legend()

    plt.tight_layout()

ID = 48
w = 50
plt.figure()
plt.plot(prices[ID])
plt.figure()
plot_rolling_alpha_beta(returns,ret_market,instrumentID=ID,window=w)
plot_expanding_alpha_beta(returns,ret_market,instrumentID=ID,min_window=w)
plt.show()