from templates.StandardTemplate import Trader, export
import numpy as np

class superiorBaseline(Trader):
    def __init__(self):
        super(Trader).__init__()
        self.first = True
        
        self.nInst = 50
        self.K = 3
        self.lookback = [0]*50
        self.LBpenalty = [[0,0]]*50
        self.PLTable = np.zeros((self.nInst,1000))
    
    @export
    def position(self,prcSoFar: np.ndarray):
        nInst,t = prcSoFar.shape

        lPrice = prcSoFar[:,-1]
        posMax = 10_000/lPrice

        if self.first:
            self.first = False
            for i in range(1,t):
                pp = prcSoFar[:,i-1]
                lp = prcSoFar[:,i]
                diff = lp-pp
                self.PLTable[:,i]  = diff + self.PLTable[:,i-1]
            return posMax
        
        pPrice = prcSoFar[:,-2]

        marketTrend = self.getMarketTrend(prcSoFar)
        position = np.full(50,marketTrend)

        self.PLTable[:,t] = lPrice-pPrice 
        Ws,Ls = self.rankedCumulativeWinLoss(t)
        print(t,Ws,Ls)
        for win in Ws:
            if position[win] != 1:
                self.LBpenalty[win][0] += 1
            else:
                self.LBpenalty[win][0] -= 1
            self.LBpenalty[win][1] += 1

            if self.LBpenalty[win][0] > 10:
                self.lookback[win] = t-10
                self.LBpenalty[win] = [0,0]
                
            accuracy = 1 - self.LBpenalty[win][0] / self.LBpenalty[win][1]
            if accuracy > 0.7:
                position[win] = 1

        for loss in Ls:
            if position[loss] != -1:
                self.LBpenalty[loss][0] += 1
            else:
                self.LBpenalty[loss][0] -= 1
            self.LBpenalty[loss][1] += 1

            if self.LBpenalty[loss][0] > 10:
                self.lookback[loss] = t-10
                self.LBpenalty[loss] = [0,0]
                

            accuracy = 1 - self.LBpenalty[loss][0] / self.LBpenalty[loss][1]
            if accuracy > 0.7:
                position[loss] = -1

        

        return posMax*position

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    
    def rankedCumulativeWinLoss(self,t:int):
        Llist = []
        Wlist = []
        for i in range(self.nInst):
            lbIndex = self.lookback[i]
            end = t-lbIndex
            differential = self.PLTable[i,lbIndex:].copy()
            
            # Accumulating win loss in range
            for j in range(1,end+1):
                differential[j] += differential[j-1]

            wr = np.where( differential > 0, 1, 0 ).sum()
            rwr = np.where( differential < 0, 1, 0 ).sum()
            # kinda arb but just sets minimum threshold 
            # of consistent & significant wins before a decision is made 
            if wr+rwr > 125:
                Llist.append((i,wr))
                Wlist.append((i,rwr))
        # Idk why but this sorting seems better
        Llist.sort(key=lambda x:x[1])
        Wlist.sort(key=lambda x:x[1])
        Ls = [ i[0] for i in Llist[:self.K] ]
        Ws = [i[0] for i in Wlist[:self.K] ]
        return (Ws,Ls)

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