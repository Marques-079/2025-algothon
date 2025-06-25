import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib import cm
import pandas as pd

class PriceForecastDensityPlotter:
    def __init__(self, prices, price_grid=None, window_size=20):
        self.prices = np.array(prices)
        self.window_size = window_size
        self.time_steps = len(prices) - window_size
        self.price_grid = price_grid or self._init_price_grid()
        self.fine_grid_size = 750
    
        pmin, pmax = np.min(self.prices), np.max(self.prices)
        padding = (pmax - pmin) * 0.1
        self.fine_price_grid = np.linspace(pmin - padding, pmax + padding, 500)

    def _init_price_grid(self):
        pmin, pmax = np.min(self.prices), np.max(self.prices)
        padding = (pmax - pmin) * 0.1
        return np.linspace(pmin - padding, pmax + padding, 200)
    

    def forecast_density_gmm(self, past_prices, n_components=2):
        """
        Fit a Gaussian Mixture Model to past prices and return PDF over fine grid.
        """
        if len(past_prices) < n_components:
            # Too little data to fit GMM — fallback to KDE
            return self.forecast_density_kde(past_prices)

        prices_reshaped = past_prices.reshape(-1, 1)
        gmm = GaussianMixture(n_components=n_components, covariance_type='full')
        gmm.fit(prices_reshaped)

        # Evaluate the PDF on the fixed price grid
        pdf = np.exp(gmm.score_samples(self.fine_price_grid.reshape(-1, 1)))
        pdf /= np.sum(pdf)  # normalize
        return pdf

    def build_density_matrix_gmm(self, window_size=40, n_components=2):
        density_matrix = []
        for t in range(2, self.time_steps + 1):
            start = max(0, t - window_size)
            past_window = self.prices[start:t]
            density = self.forecast_density_gmm(past_window, n_components=n_components)
            density_matrix.append(density)
        return np.array(density_matrix).T

    def run_gmm(self, window_size=40, n_components=2):
        density_matrix = self.build_density_matrix_gmm(window_size=window_size, n_components=n_components)
        self.plot_density_heatmap(density_matrix)


    def forecast_density_kde(self, past_prices):
        """
        KDE over the provided past_prices.
        Handles early time steps by allowing small sample sizes.
        """
        kde = gaussian_kde(past_prices)
        density = kde(self.fine_price_grid)
        return density / np.sum(density)

    def build_density_matrix_kde(self, window_size=40):
        density_matrix = []
        for t in range(1, len(self.prices)):
            # Use the last 'window_size' prices, or fewer if early
            past_window = self.prices[max(0,t-window_size):t+1]
            density = self.forecast_density_kde(past_window)
            density_matrix.append(density)
        return np.array(density_matrix).T
    

    def forecast_density_kalman(self, temperature=1.0):
        """
        Kalman filter forecasting: builds Gaussian predictive distribution at each step.
        """
        # Kalman model parameters
        mu = self.prices[0]  # initial estimate
        sigma2 = 1.0  # initial variance
        Q = 1.0       # process noise
        R = 1.0       # observation noise

        density_matrix = []

        for t in range(self.time_steps):
            z = self.prices[t]

            # === Kalman update ===
            # Kalman gain
            K = sigma2 / (sigma2 + R)
            # Update estimate with observation
            mu = mu + K * (z - mu)
            # Update variance
            sigma2 = (1 - K) * sigma2

            # === Kalman predict next ===
            mu_pred = mu
            sigma2_pred = sigma2 + Q

            # Evaluate predicted Gaussian over fine price grid
            pdf = norm.pdf(self.fine_price_grid, loc=mu_pred, scale=np.sqrt(sigma2_pred / temperature))
            pdf /= np.sum(pdf)  # normalize

            density_matrix.append(pdf)

        return np.array(density_matrix).T  # shape: [price_bins, time_steps]


    def plot_density_heatmap(self, density_matrix):
        t = np.linspace(1, len(self.prices), len(self.prices))

        f,ax = plt.subplots(subplot_kw={"projection":"3d"})
        price_bins, time_steps = density_matrix.shape
        # Create meshgrid for plotting
        time_grid = np.arange(time_steps)
        price_grid = self.fine_price_grid
        T, P = np.meshgrid(time_grid, price_grid)
        surf = ax.plot_surface(T, P, density_matrix, cmap=cm.coolwarm)

        plt.figure(figsize=(10, 6))
        # plt.plot(t,self.prices,color="orange",linestyle="--",marker='o')
        plt.imshow(
            density_matrix,
            aspect='auto',
            origin='lower',
            extent=[0, len(self.prices), self.price_grid[0], self.price_grid[-1]],
            cmap='viridis'
        )
        plt.colorbar(label='Density')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.title('Forecasted Price Density Over Time')
        plt.show()

    def run_kde(self, window_size=20):
        density_matrix = self.build_density_matrix_kde(window_size=window_size)
        self.plot_density_heatmap(density_matrix)
    
    def run_kalman(self, temperature=1.0):
        density_matrix = self.forecast_density_kalman(temperature=temperature)
        self.plot_density_heatmap(density_matrix)


def loadPrices(fn):
    df = pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    return df

prices = loadPrices("prices.txt")
focus_length = 750
instrument = prices[0][:focus_length].to_numpy()  # limit to 750 samples if larger

# Sample usage
prices = instrument  # Simulated price series
plotter = PriceForecastDensityPlotter(prices)
plotter.run_kde()
