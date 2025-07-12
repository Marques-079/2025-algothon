from gen import R3Generator
from eval import Evaluator
import sys
import pandas as pd

import os
# Add the parent directory of main_dir/ to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Model.Dev import Insider

def loadPrices(fn):
    df = pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    return df

RealData = ( loadPrices("prices.txt").values ).T
TD = RealData
ref=  Insider.InsideTrader(TD)

e = Evaluator(TD,ref.get_exported())

t = TD.shape[1]
(meanpl, ret, plstd, sharpe, dvol) = e.calcPL(0,t)
score = meanpl - 0.1*plstd
print ("=====")
print ("mean(PL): %.1lf" % meanpl)
print ("return: %.5lf" % ret)
print ("StdDev(PL): %.2lf" % plstd)
print ("annSharpe(PL): %.2lf " % sharpe)
print ("totDvolume: %.0lf " % dvol)
print ("Score: %.2lf" % score)

