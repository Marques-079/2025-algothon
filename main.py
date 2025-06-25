import numpy as np
from InsiderTrading import InsideTrader
# Where did rsi_exponential go??
# from market_condition.indicators import rsi_exponential, atr_close_to_close
from NN import NN

from templates.ExampleMain import Baseline

##### TODO #########################################
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################
IT = InsideTrader()
IT.export_trader()

# example = Baseline()
# example.export_trader()

nn = NN()
nn.export_trader()

nInst = 50
currentPos = np.zeros(nInst)



# Template Indicator Starting
def MachineOrchestra(prcSoFar):
    global currentPos
    bought = False

    price_buy = None
    day_bought = None
    tp = None
    sl = None
    tp_ratio = 10
    sl_ratio = 5

    currentPos = np.zeros(nins)

    # nins is number of instruments
    # nt is number of days in
    (nins, nt) = prcSoFar.shape

    # Get all indicators
    rsi = rsi_exponential(prcSoFar[0, :])
    atr = atr_close_to_close(prcSoFar[0, :])

    # Plug into ML models
    ## basic for now
    if rsi[-1] < 30 and not bought:
        bought = True
        price_buy = prcSoFar[0, -1]
        day_bought = nt
        tp = price_buy + tp_ratio*atr[-1]
        sl = price_buy - sl_ratio*atr[-1]
        currentPos[0] = 50
    

    # Adjust from current pos

    return currentPos
