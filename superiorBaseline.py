from templates.StandardTemplate import Trader, export
import numpy as np

class superiorBaseline(Trader):
    def __init__(self):
        super(Trader).__init__()
        self.first = True
    
    @export
    def position(self,prcSoFar: np.ndarray):
        nInst,t = prcSoFar.shape
        lPrice = prcSoFar[:,-1]
        posMax = 10_000/lPrice

        if self.first:
            self.first = False
            return posMax

        marketTrend = self.getMarketTrend(prcSoFar)
        return posMax*marketTrend

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def getMarketTrend(self,prcSoFar: np.ndarray):
        nInst,t = prcSoFar.shape
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
        return index_delta