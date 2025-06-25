from operator import mul
from templates.helper import loadPrices
from templates.StandardTemplate import Trader,export
    
import numpy as np

from scipy.stats import gaussian_kde
from multiprocessing import Pool, cpu_count


class KDEAnalyzer:
    def __init__(self,window_size=20):
        self.window_size = window_size
        self.KDE = None
        self.Grid = None
        self.lDensity = []
    
    def kde(self, past_prices:np.ndarray,t: int):
        past_prices = past_prices[max(0,t-self.window_size):t+1]
        _grid_size = 500
        pmin, pmax = np.min(past_prices), np.max(past_prices)
        padding = (pmax - pmin) * 0.2
        _price_grid = np.linspace(pmin - padding, pmax + padding, _grid_size)

        kde = gaussian_kde(past_prices)
        self.KDE = kde
        self.Grid = _price_grid
        density = kde(_price_grid)
        density /= np.sum(density)
        self.lDensity = density
        return density
    
    def mean(self):
        self.m = np.trapz(self.lDensity*self.Grid)
        return self.m
    
    def std(self):
        self.Var = np.trapz(np.pow( self.Grid-self.mean(),2 )*self.lDensity)
        return np.sqrt(self.Var)


class mmx(Trader):
    def __init__(self):
        super(Trader).__init__()
        self.first = True
    
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    @export
    def minimax(self, prcSoFar: np.ndarray):
        nInst, t = prcSoFar.shape
        lp = prcSoFar[:, -1]
        lpmax = 10_000 / lp

        if self.first:
            self.first = False
            return lpmax
        
class ZScoreReversion(Trader):
    def __init__(self, lookback=20):
        super().__init__()
        self.lookback = lookback
        self.first = True
        self.analyzers: list[KDEAnalyzer] = []
        for _ in range(50):
            self.analyzers.append(KDEAnalyzer())

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def market_trend(self,prcSoFar: np.ndarray):
        nInst, t = prcSoFar.shape
        lp = prcSoFar[:,-1]
        lpmax = 10_000/lp

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
        index_delta = d_dist.sum()/len(d_dist)
        index_delta/=abs(index_delta)

        r = index_delta
        return r
    @export
    def trade(self, prcSoFar: np.ndarray):
        nInst, t = prcSoFar.shape
        lp = prcSoFar[:, -1]
        lpmax = 10_000 / lp

        if self.first:
            self.first = False
            return lpmax

        R = np.full(50,0)
        anoms = []
        anomDiff = []

        for i in range(50):
            priceArray = prcSoFar[i,:]
            a = self.analyzers[i]
            density = np.array( a.kde(priceArray,t) )
            std = a.std()
            mean = a.m
            if std > 2:
                anoms.append(i)
                diff = lp[i]-mean
                diff /= abs(diff)
                anomDiff.append(diff)
                R[i] = -diff
        print(t,anoms,np.array(anomDiff))

            
        return R*lpmax
