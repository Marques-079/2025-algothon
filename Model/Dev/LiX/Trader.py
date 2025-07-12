from Model.standard_template import Trader,export
import numpy as np

class LiX(Trader):
    def __init__(self):
        super(Trader).__init__()
    
    @export
    def make_pos(self,prcSoFar: np.ndarray):
        return 10_000/prcSoFar[:,-1]

