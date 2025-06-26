from templates.StandardTemplate import Trader, export
import numpy as np

class superiorBaseline(Trader):
    def __init__(self):
        super(Trader).__init__()
    
    @export
    def position(self,prcSoFar: np.ndarray):
        pass