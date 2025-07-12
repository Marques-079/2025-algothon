from matplotlib.lines import lineStyles
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

commRate = 0.0005
dlrPosLimit = 10000

class Evaluator():
    def __init__(self,prcHist,benchmarked,refernce):
        """
        benchmarked: Trader that is being tested
        refernce: Another trader to compare with the benchmarked trader (This should just be the insider)
        """
        self.prcHist = prcHist
        self.getPos = benchmarked
        self.positionsHeld = []
        self.referncePositions = []
        if refernce:
            self.calcPL(0,self.prcHist.shape[1],refernce)
            self.referncePositions = np.stack(self.positionsHeld).T
            self.positionsHeld = []

    def calcPL(self, startDay, endDay,getPos):
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
                newPosOrig = getPos(prcHistSoFar)
                self.positionsHeld.append(newPosOrig)
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
                # print ("Day %d value: %.2lf todayPL: $%.2lf $-traded: %.0lf return: %.5lf" % (t,value, todayPL, totDVolume, ret))
                todayPLL.append(todayPL)
        pll = np.array(todayPLL)
        (plmu,plstd) = (np.mean(pll), np.std(pll))
        annSharpe = 0.0
        if (plstd > 0):
            annSharpe = np.sqrt(249) * plmu / plstd
        
        return (plmu, ret, plstd, annSharpe, totDVolume)
    
    def evaluate(self,startDay,endDay):
        (meanpl, ret, plstd, sharpe, dvol) = self.calcPL(startDay,endDay,self.getPos)
        score = meanpl - 0.1*plstd
        print ("=====")
        print ("mean(PL): %.1lf" % meanpl)
        print ("return: %.5lf" % ret)
        print ("StdDev(PL): %.2lf" % plstd)
        print ("annSharpe(PL): %.2lf " % sharpe)
        print ("totDvolume: %.0lf " % dvol)
        print ("Score: %.2lf" % score)
        self.comparePositions()

    def comparePositions(self):
        self.positionsHeld = np.stack(self.positionsHeld).T
        diff = self.positionsHeld-self.referncePositions
        diff = np.sum(np.abs(diff),axis=1)
        diff /= np.linalg.norm(diff)
        ranking = np.argsort(diff)
        score = 1-diff[ranking]
        print("Instrument:",ranking)
        print("Score:",score)