from templates.helper import loadPrices
from templates.StandardTemplate import Trader,export
    
import numpy as np

class mmx(Trader):
    def __init__(self):
        super(Trader).__init__()
        self.first = True
    
    @export
    def minimax(self,prcSoFar: np.ndarray):
        nInst, t = prcSoFar.shape
        lp = prcSoFar[:,-1]
        lpmax = 10_000/lp

        if self.first:
            self.first = False
            return lpmax