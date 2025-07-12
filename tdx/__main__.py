from gen import *
import matplotlib.pyplot as plt

import pandas as pd
def loadPrices(fn):
    df = pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    return df

RealData = ( loadPrices("prices.txt").values ).T

from eval import Evaluator
import sys

import os
# Add the parent directory of main_dir/ to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Model.Dev import Insider


TD = R1Icorr()
ref=  Insider.InsideTrader(TD)

e = Evaluator(TD,ref.get_exported())

t = TD.shape[1]
e.evaluate(0,t)