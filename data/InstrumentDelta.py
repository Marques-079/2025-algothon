import numpy as np

class Analyzer_inst_delta:
    def __init__(self,K,nInst=50,T=750):
        self.K = K
        self.T = T
        self.nInst = nInst
        self.netPos = np.zeros((nInst, T))
        self.netNeg = np.zeros((nInst, T))
    
    def getPDiff(self,t:int,prices:np.ndarray):
        pdiff = np.zeros((self.nInst,t))
        for i in range(self.nInst):
            current = prices[i,:t]
            previous = np.zeros((t,))
            previous[0] = 1e-8
            previous[1:] = current[:-1]
            delta = current-previous
            percent_delta = 100* delta / previous
            percent_delta[0] = 0
            pdiff[i,:] = percent_delta
        return pdiff
    
    def updateNPNN(self,t:int, prices:np.ndarray):
        pdiff = self.getPDiff(t,prices)
        posDelta = np.where(pdiff > 0,pdiff,0)
        negDelta = np.where(pdiff < 0,-pdiff,0)

        self.netPos[:,t-1] = posDelta.sum(axis=1)
        self.netNeg[:,t-1] = negDelta.sum(axis=1)
    
    def getTopBottomK(self):
        gdList = []
        for i in range(self.nInst):
            diff = self.netPos[i]-self.netNeg[i]
            delta = np.where(diff  > 0,1,0 ).sum()
            gdList.append((i,delta))
        gdList.sort(key=lambda x:x[1],reverse=True)
        Bs = [i[0] for i in gdList[:self.K] ]
        Ts = [ i[0] for i in gdList[-self.K:] ]
        return (Bs,Ts)

    
    