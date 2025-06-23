import pandas as pd

def loadPrices():
    fn="prices.txt"
    global nt, nInst
    df=pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    (nt,nInst) = df.shape
    return nt, nInst, (df.values).T