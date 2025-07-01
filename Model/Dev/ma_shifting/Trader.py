from Model.standard_template import Trader,export
import numpy as np

class ma_shifting(Trader):
    @export
    def GodTierTrading(self, _):
        return np.random.random_integers(-10_000,10_000,50)