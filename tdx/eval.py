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
        self.positionsHeld = []
        (meanpl, ret, plstd, sharpe, dvol) = self.calcPL(startDay,endDay,self.getPos)
        score = meanpl - 0.1*plstd
        print ("=====")
        print ("mean(PL): %.1lf" % meanpl)
        print ("return: %.5lf" % ret)
        print ("StdDev(PL): %.2lf" % plstd)
        print ("annSharpe(PL): %.2lf " % sharpe)
        print ("totDvolume: %.0lf " % dvol)
        print ("Score: %.2lf" % score)
        self.comparePositions(startDay,endDay)

    def comparePositions(self,startDay,endDay):
        self.positionsHeld = np.stack(self.positionsHeld).T
        diff = self.positionsHeld-self.referncePositions[:,startDay:endDay]
        eqTrades = np.sum(np.where(diff == 0,1,0),axis=1)
        pdiff = np.where(diff > 0, diff , 0)
        ndiff = np.where(diff < 0, diff , 0)
        pscore = np.sum(pdiff,axis=1)
        nscore = np.sum(ndiff,axis=1)
        score = np.sum(np.abs(diff),axis=1)
        print("Mean score:",score.mean(),pscore.mean(),nscore.mean())
        print("Mean equal trades:",eqTrades.mean())
        # score /= np.linalg.norm(score)
        ranking = np.argsort(score)
        df = pd.DataFrame({
            "Instruments":ranking,
            "Score":score[ranking],
            "pScore":pscore[ranking],
            "nScore":nscore[ranking],
            "EqualTrades":eqTrades[ranking]
        })
        print(df.describe())
        # print(df)

        # self.showPositions(range(50),startDay,endDay)
        
    def showPositions(self,p,startDay,endDay):
        n = self.referncePositions.shape[0]
        colors = np.random.rand(n, 3)
        cols = 10
        rows = (n) // cols 
        fig, axes = plt.subplots(rows, cols)
        fig.subplots_adjust(hspace=0.6, wspace=0.7)
        axes = axes.flatten()  # flatten to 1D list of axes

        for i in p:
            diff = self.referncePositions[i, startDay:endDay] - self.positionsHeld[i, :]
            diff = np.abs(diff)
            axes[i].plot(np.linspace(startDay,endDay,endDay-startDay-1),diff, color=colors[i], marker='o', linestyle="", alpha=0.2)
            axes[i].set_title(f"Instrument {i}")
        plt.show()