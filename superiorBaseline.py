from templates.StandardTemplate import Trader, export
import numpy as np

class superiorBaseline(Trader):
    def __init__(self):
        super(Trader).__init__()
        self.first = True
        
        self.nInst = 50
        self.PLTable = np.zeros((self.nInst,1000))
    
    @export
    def position(self,prcSoFar: np.ndarray):
        nInst,t = prcSoFar.shape

        lPrice = prcSoFar[:,-1]
        posMax = 10_000/lPrice

        if self.first:
            self.first = False
            return posMax
        
        pPrice = prcSoFar[:,-2]
        self.PLTable[:,t] = self.getProfitLoss(lPrice,pPrice) + self.PLTable[:,t-1]
        self.rankedCumulativeWinLoss()
        
        marketTrend = self.getMarketTrend(prcSoFar)
        position = np.full(50,marketTrend)

        return posMax*position

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    
    def getProfitLoss(self,lPrice:np.ndarray,pPrice:np.ndarray):
        diff = lPrice-pPrice
        return diff / pPrice * 100

    def rankedCumulativeWinLoss(self):
        K = 10
        gdList = []
        for i in range(self.nInst):
            wr = np.where( self.PLTable[i,:] > 0, 1, 0 ).sum()
            gdList.append((i,wr))
        gdList.sort(key=lambda x:x[1],reverse=True)
        Ws = [i[0] for i in gdList[:K] ]
        Ls = [ i[0] for i in gdList[-K:] ]
        print(Ws,Ls)

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