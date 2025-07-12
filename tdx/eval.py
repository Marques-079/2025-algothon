import numpy as np

commRate = 0.0005
dlrPosLimit = 10000

class Evaluator():
    def __init__(self,prcHist,getPosition):
        self.prcHist = prcHist
        self.getPos = getPosition

    def calcPL(self, startDay, endDay):
        (nInst,nt) = self.prcHist.shape
        cash = 0
        curPos = np.zeros(nInst)
        totDVolume = 0

        value = 0
        todayPLL = []
        for t in range(startDay+1, endDay+1):
            prcHistSoFar = self.prcHist[:,:t]
            curPrices = prcHistSoFar[:,-1]
            if (t < nt):
                # Trading, do not do it on the very last day of the test
                newPosOrig = self.getPos(prcHistSoFar)
                posLimits = np.array([int(x) for x in dlrPosLimit / curPrices])
                newPos = np.clip(newPosOrig, -posLimits, posLimits)
                deltaPos = newPos - curPos
                dvolumes = curPrices * np.abs(deltaPos)
                dvolume = np.sum(dvolumes)
                totDVolume += dvolume
                comm = dvolume * commRate
                cash -= curPrices.dot(deltaPos) + comm
            else:
                newPos = np.array(curPos)
            curPos = np.array(newPos)
            posValue = curPos.dot(curPrices)
            todayPL = cash + posValue - value
            value = cash + posValue
            ret = 0.0
            if (totDVolume > 0):
                ret = value / totDVolume
            if (t > startDay):
                print ("Day %d value: %.2lf todayPL: $%.2lf $-traded: %.0lf return: %.5lf" % (t,value, todayPL, totDVolume, ret))
                todayPLL.append(todayPL)
        pll = np.array(todayPLL)
        (plmu,plstd) = (np.mean(pll), np.std(pll))
        annSharpe = 0.0
        if (plstd > 0):
            annSharpe = np.sqrt(249) * plmu / plstd
        return (plmu, ret, plstd, annSharpe, totDVolume)