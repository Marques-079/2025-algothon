# 2025 algothon 📊

## Overview

This repository was built for participation in the **2025 SIG Algothon Trading Competition**, where the goal is to develop and showcase innovative and effective trading strategies in the elusive field of quantitative research. 

## System architecture

* **Modular architecture:** 
The repository is built around a modular architecture, segregated into distinct compartments for testing, research, and model development.

* **Flexible model framework:** 
Model development utilizes a custom-built framework designed to dynamically scan active models under development or selectively test pre-existing models. This
framework facilitates fine-grained state management within the trading algorithm by seamlessly wrapping default functions into a class. 

* **Backtesting and evaluation tools:**
The **testing** library, tdx, offers multiple data sourcing opportunities, including synthetic and custom price data. The backtesting tool is sourced from an open-source repository, providing robust performance analysis.

### Technologies used

*   **NumPy:** Fundamental for numerical computation, array manipulation, and FFT operations.
*   **Pandas:** Used for data manipulation, handling dataframes, and time series data.
*   **Scikit-learn:** Employed by various components for linear regression.
*   **Pytorch/TensorFlow:**   Multiple approaches leverages these for building and training the neural network to investigate projection logistics.
*   **Matplotlib/Seaborn:** Used for visualization in many of these files.
*   **SciPy:** Used for some of the signal processing and numerical calculations (e.g., convolution).

## Research Endeavors

* **Neural networks & Machine learning:** 
The project utilizes a TensorFlow feedforward network with a Huber loss for robust time series forecasting and a supervised bidirectional LSTM (PyTorch) that analyzes
historical stock price regimes, incorporating a novel attention mechanism for increased predictive accuracy.

*  **Stochastic Modeling – Fokker-Planck Equation:** The Fokker-Planck equation is implemented to model the stochastic evolution of an asset’s price. This sophisticated
approach estimates the drift and diffusion coefficients, representing the probability density of price states over time – a core element of building a more realistic and
nuanced stochastic model.

*  **Regression Models for Market Beta Estimation:** The project uses linear regression, both in its standard form and with expanding/rolling windows, to estimate market
beta – a critical measure of an asset’s volatility and correlation to the overall market. This approach relies on the assumption that historical price data can be
effectively used to predict future beta values.

* **Identifying Market Shifts:** We use statistical anomaly detection to pinpoint significant price shifts. This method calculates Z-scores to identify outliers and applies a
carefully calibrated threshold.  This robust approach distinguishes genuine price movements from normal market fluctuations.

* **Uncovering Trends with FFT:** We then decompose the price data using Fourier Analysis (FFT), revealing underlying trends and cyclical patterns.  By fitting an exponential
decay model to this data, we capture the principle of mean reversion – the market’s tendency to return to its long-term average. This powerful technique, however, requires
careful adjustment to minimize spectral leakage and ensure the best possible results."

* **Kalman Filtering:** Kalman filtering is employed to track moving objects and predict future price values by combining noisy measurements with models incorporating constant velocity assumptions.

* **Statistical Analysis & Density Estimation:** Statistical analysis, including Kernel Density Estimation and Gaussian Mixture Models, alongside ranked returns and trading volume data, is utilized to generate probabilistic
price forecasts and identify profitable trading opportunities.

*  **Volatility Analysis – Rolling Statistics & Signal Filtering:** Rolling mean, standard deviation, and z-score calculations are used to assess volatility and identify
trends. The inclusion of a Savitzky-Golay filter suggests an effort to denoise the price data, likely to sharpen trends and improve the accuracy of subsequent analysis.

*   **Technical Analysis – Parabolic SAR Pivot Detection:**  A Parabolic SAR indicator, coupled with pseudo-ADX calculation, is used to identify "breaking pivot" points –
signals interpreted as shifts in market momentum. This leverages a technical analysis approach, relying on the assumption that significant price changes represent trend
changes.

*   **Clustering – Financial Time Series:**  Clustering techniques are applied to group financial assets based on similarities in their price movements.  This provides an
exploratory data analysis strategy for identifying potentially related assets.

## Requirements

* **Python3.x**: The repository requires Python3.x to run.
* **Dependencies**: The repository has a number of dependencies, including `numpy`, `pandas`, and `matplotlib`, which are listed in `requirements.txt`.

## Installation

To install the repository, simply clone it and run `pip install -r requirements.txt` to install the dependencies.

## Acknowledgements

The repository was developed by [Your Name] with contributions from [Other Contributors].


