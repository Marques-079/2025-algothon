import numpy as np
from InsiderTrading import InsideTrader
from BestLastAlg import ChooseBestLastBar
from templates.ExampleMain import Baseline

##### TODO #########################################
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

IT = InsideTrader()
# IT.export_trader()

example = Baseline()
# example.export_trader()

BLB = ChooseBestLastBar()
BLB.export_trader()