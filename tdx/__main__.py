import matplotlib.pyplot as plt
import sys
import os
from gen import *
from eval import Evaluator
# Add the parent directory of main_dir/ to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Model.Dev import Insider
from Model.Dev import *
import pandas as pd

def loadPrices(fn):
    df = pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    return df

RealData = ( loadPrices("prices.txt").values ).T
TD = RealData

T = ActiveModel.trader
T.export_trader()

ref=  Insider.InsideTrader(TD)

e = Evaluator(TD,getMyPosition,ref.get_exported())

t = TD.shape[1]
e.evaluate(750,t)