from .MeanReversionAlg import MeanReversionTrader
import pickle

# Algorithm's current working directory is relative to where it is executed so 
# so use paths as if it is at the root of the directory
with open("Research/market_condition/mean_reversion_params.pkl", "rb") as f:
    data = pickle.load(f)

beta_matrix = data["beta_matrix"]
alpha_matrix = data["alpha_matrix"]
pairs = data["pairs"]
if __name__ == "__main__":
    print(pairs)

# Initialize trader
trader = MeanReversionTrader(pairs, beta_matrix, alpha_matrix)