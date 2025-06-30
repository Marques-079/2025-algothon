# Reference for using templates

import numpy as np

from Model.standard_template.StandardTemplate import Trader, export, export_trading_funcs

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
        kernel = np.array([1,-1])  #profit kernel
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

        r = index_delta*lpmax
        return r

# Uncomment below 

# # Exporting the function
# traderExample = Baseline()
# traderExample.export_trader()


#-------------------------------------------------- 
#      Example function the standard template
#-------------------------------------------------- 

@export
def GodTierTrading(_):
    return np.random.random_integers(-10_000,10_000,50)

# Uncomment below

# # Note: this would NOT override the previous traderExample export so make sure only 1 export at a time
# export_trading_funcs()
