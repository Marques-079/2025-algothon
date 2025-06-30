from .MeanReversionAlg import MeanReversionTrader
import pickle
with open("Research/market_condition/mean_reversion_params.pkl", "rb") as f:
    data = pickle.load(f)

beta_matrix = data["beta_matrix"]
alpha_matrix = data["alpha_matrix"]
pairs = data["pairs"]
if __name__ == "__main__":
    print(pairs)

# Initialize trader
trader = MeanReversionTrader(pairs, beta_matrix, alpha_matrix)