import numpy as np


from StandardTemplate import Trader, export, export_trading_funcs
from indicators import rsi_exponential, atr_close_to_close

##### TODO #########################################
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

nInst = 50
currentPos = np.zeros(nInst)

#-------------------------------------------------- 
#      Example method using the standard template
#-------------------------------------------------- 

class Baseline(Trader):
    def __init__(self):
        super(Trader).__init__()
        self.first = True
    
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    @export
    def base_model(self,prcSoFar: np.ndarray):
        nInst, t = prcSoFar.shape
        lp = prcSoFar[:,-1]
        lpmax = 10_000/lp

        if self.first:
            self.first = False
            return lpmax
        kernel = np.array([-1,1])  #profit kernel
        convolved_array = np.zeros((nInst,t))
        for i in range(50):
            convolved_array[i,:] = np.convolve(prcSoFar[i,:], kernel, mode='same')  # or 'same', 'full'
        

        
        d_dist = []
        d = 15
        n = int(d*self.sigmoid(t/d))
        for i in range(1,n+1):
            diff = convolved_array[:,-i:].sum(axis=1)/i
            d_dist.append(diff)

        d_dist = np.array(d_dist)
        m = np.tanh( np.arange(1,n+1) )
        m = m[:,np.newaxis]
        d_dist *= m
        index_delta = d_dist.sum()/n
        index_delta/=abs(index_delta)

        r = -index_delta*lpmax
        return r

# Exporting the function
traderExample = Baseline()
traderExample.export_trader()




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
