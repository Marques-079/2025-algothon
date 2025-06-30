import numpy as np
# Development models
from Model.Dev.Insider import InsideTrader
from Model.Dev.demo import Baseline
from Model.Dev.BestLast import ChooseBestLastBar
from Model.Dev.superiorBaseline import superiorBaseline
from Model.Dev.MeanReversion.MeanReversionAlg import MeanReversionTrader

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