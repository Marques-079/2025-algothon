from templates.StandardTemplate import Trader, export
import numpy as np
import pickle


class superiorBaseline(Trader):
    def __init__(self):
        super(Trader).__init__()
        self.first = True
        
        self.nInst = 50
        self.K = 3
        self.lookback = [0]*50
        self.LBpenalty = [[0,0]]*50
        self.PLTable = np.zeros((self.nInst,1000))
        
        # A rest occurs when there is significant disagreement with market trend
        # and a instrument's observed trend
        
        # When a rest occurs how far in time should it look back to
        self.lookbackResetAmnt = 5
        
        #accuracy: how often should the market trend correlate with the instrument trend
        self.acc_target = 0.1
        
        # Correlated instrument grouping
        corr_pickle = "market_condition/correlated_groups.pkl"
    
        with open(corr_pickle, 'rb') as f:
            data = pickle.load(f)
        corr_matrix, grouped_instruments = data['correlation_matrix'], data['grouped_instruments']
        self.groups = grouped_instruments
    
    @export
    def position(self,prcSoFar: np.ndarray):
        nInst,t = prcSoFar.shape

        lPrice = prcSoFar[:,-1]
        posMax = 10_000/lPrice

        if self.first:
            self.first = False
            return posMax
        
        pPrice = prcSoFar[:,-2]

        marketTrend = self.getMarketTrend(prcSoFar)
        position = np.full(50,marketTrend)

        self.PLTable[:,t] = lPrice-pPrice 
        Ws,Ls = self.rankedCumulativeWinLoss(t)
        for win in Ws:
            accuracy = self.updatePenaltyTable(win,t,position[win],1)
            if accuracy > self.acc_target:
                position[win] = 1

        for loss in Ls:
            accuracy = self.updatePenaltyTable(loss,t,position[loss],-1)
            if accuracy > self.acc_target:
                position[loss] = -1

        # The G.O.A.T group
        # group = self.groups[3] timeframe = 7, cutoff = 1
        group = [3,29,7]
        g_matrix = prcSoFar[group,:]
        group_trend_index = self.getMarketTrend(g_matrix)
        position[group] = group_trend_index
        

        return posMax*position

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    
    def updatePenaltyTable(self,instrument:int,t:int, current_pos:int, intended_pos:int):
        if current_pos != intended_pos:
            self.LBpenalty[instrument][0] += 1
        self.LBpenalty[instrument][1] += 1
        
        # Rest table if threshold exceeded
        accuracy = 1 - self.LBpenalty[instrument][0] / self.LBpenalty[instrument][1]
        runCount = self.LBpenalty[instrument][1]
        if accuracy < self.acc_target and  runCount > 0 :
            self.lookback[instrument] = t-self.lookbackResetAmnt
            self.LBpenalty[instrument] = [0,0]

        return accuracy
    
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

        for i in range(nInst):
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
