from templates.helper import loadPrices
from templates.StandardTemplate import Trader,export
    
import numpy as np

class mmx(Trader):
    def __init__(self):
        super(Trader).__init__()
    
    @export
    def minimax(self,prcSoFar: np.ndarray):
        pass