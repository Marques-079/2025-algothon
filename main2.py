from math import fabs
from pandas import DataFrame
from requests import ReadTimeout
from scipy.special import softmax
import numpy as np

from indicators import rsi_exponential, atr_close_to_close

nInst = 50
currentPos = np.zeros(nInst)
bought = False
price_buy = None
day_bought = None
tp = None
sl = None
tp_ratio = 3
sl_ratio = 3



# Template Indicator Starting
def MachineOrchestra(prcSoFar):
    ### Define vars
    global currentPos
    global bought 
    global price_buy 
    global day_bought 
    global tp 
    global sl 
    global tp_ratio 
    global sl_ratio 

    # nins is number of instruments
    # nt is number of days in
    (nins, nt) = prcSoFar.shape

    inst = 1

    ### Get all indicators
    rsi = rsi_exponential(prcSoFar[inst, :])
    atr = atr_close_to_close(prcSoFar[inst, :])

    ### Plug into ML models & Getting buy signal
    last_price = prcSoFar[inst, -1]
    
    # if rsi below 30 we buy in
    buy_signal = rsi[-1] < 40

    ### Adjust from current pos
    if buy_signal and not bought:
        bought = True
        price_buy = last_price
        day_bought = nt
        tp = price_buy + tp_ratio*atr[-1]
        sl = price_buy - sl_ratio*atr[-1]
        currentPos[inst] = 250
        print(f"Bought 250 shares at {price_buy} with TP: {tp} & SL: {sl}\n\n")
    
    if bought:
        if last_price >= tp or last_price <= sl:
            bought = False
            price_buy = None
            day_bought = None
            tp = None
            sl = None
            currentPos[inst] = 0

    return currentPos

getMyPosition = MachineOrchestra