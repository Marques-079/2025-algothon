import numpy as np
from Model.InsiderTrading import InsideTrader
from Model.BestLastAlg import ChooseBestLastBar
from Model.superiorBaseline import superiorBaseline
from Model.ExampleMain import Baseline
from Model.MeanReversionAlg import MeanReversionTrader  # if it's in a separate file
import pickle

##### TODO #########################################
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

IT = InsideTrader()
# IT.export_trader()

example = Baseline()
# example.export_trader()

BLB = ChooseBestLastBar()
# BLB.export_trader()


# Load the parameters
with open("market_condition/mean_reversion_params.pkl", "rb") as f:
    data = pickle.load(f)

beta_matrix = data["beta_matrix"]
alpha_matrix = data["alpha_matrix"]
pairs = data["pairs"]
if __name__ == "__main__":
    print(pairs)

# Initialize trader
MV = MeanReversionTrader(pairs, beta_matrix, alpha_matrix)
# MV.export_trader()

sbase = superiorBaseline()
sbase.export_trader()