from Model.standard_template import Trader,export
import numpy as np
import matplotlib.pyplot as plt

class LiX(Trader):
    def __init__(self):
        super(Trader).__init__()
        self.lpmax = np.zeros(50)
        self.emaRange = 50
        self.emaTable = []
        self.stdTable = None
        self.first = True
    
    @export
    def make_pos(self,prcSoFar: np.ndarray):
        _, t = prcSoFar.shape
        current_price = prcSoFar[:,-1]
        self.lpmax = 10_000/current_price
        N =  100
        if self.first:
            self.first = False
            for i in range(0,N):
                self.emaTable.append(self.pregen_ema(prcSoFar,span=i))
            emas = np.array( self.emaTable )
            stdT = np.std(emas,axis=0) 
            self.stdTable = stdT

        curr_emas = []
        for i in range(0,N):
            emas = self.emaTable[i]
            new_ema = self.get_next_ema(prcSoFar,span=i)
            curr_emas.append(new_ema)
            new_ema = new_ema[:,np.newaxis]
            self.emaTable[i] = np.concat(( emas,new_ema ),axis=1)
        curr_emas = np.array(curr_emas)
        cm_mean = np.mean(curr_emas,axis=0)
        cm_std = np.std(curr_emas,axis=0)

        self.stdTable = np.append(self.stdTable,cm_std[:,np.newaxis],1)
        min_val = np.min(self.stdTable,1)
        max_val = np.max(self.stdTable,1)
        centered_std = cm_std - min_val
        diff  = max_val-min_val
        centered_std /= diff
        
        trade_signal = current_price - cm_mean
        trade_signal /= np.abs(trade_signal)
        
        weight = centered_std
        threshold = np.percentile(weight, 30)
        weight[weight <= threshold] = 0

        pos = trade_signal * weight * self.lpmax
        return pos

    def pregen_ema(self,mat,span):
        span = self.span_transform(span)
        alpha = 2 / (span + 1)
        nInst,T = mat.shape
        T -= 1
        ema = np.zeros(shape=(nInst,T))
        ema[:,0] = mat[:,0]  # Initialize with first value
        for t in range(1, T):
            ema[:,t] = alpha * mat[:,t] + (1 - alpha) * ema[:,t-1]
        return ema
    
    def get_next_ema(self,mat,span):
        spandex = span
        span = self.span_transform(span)
        alpha = 2 / (span + 1)
        ema = self.emaTable[spandex]
        new_ema = alpha * mat[:,-1] + (1 - alpha) * ema[:,-1]
        return new_ema

    def span_transform(self,s):
        return 1*(s+1)