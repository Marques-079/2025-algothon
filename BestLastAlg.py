import numpy as np
import pandas as pd
from templates.StandardTemplate import Trader, export

def loadPrices(fn):
    df = pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    return df

class ChooseBestLastBar(Trader):
    def __init__(self):
        super(Trader).__init__()
        self.last_positions = np.full(50, 0)
    
    @export
    def Alg(self,prcSoFar: np.ndarray):
        nInst, t = prcSoFar.shape

        # If first two bars exit!
        if t < 2:
            return np.full(50, 0) 
        
        # Compute last bar returns
        last_returns = prcSoFar[:, -1] - prcSoFar[:, -2] if t >= 2 else np.zeros(nInst)

        # Choose top N by absolute return
        N = 5  # You can tune this
        top_indices = np.argsort(np.abs(last_returns))[-N:]

        # Create position array
        position_values = np.zeros(nInst, dtype=int)
        for idx in top_indices:
            position_values[idx] = 10000 if last_returns[idx] > 0 else -10000
        positions = np.floor(position_values / prcSoFar[:, -1])
        print(positions)

        return positions
    
        
