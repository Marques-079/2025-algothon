import numpy as np
from Model.standard_template import Trader,export
from sklearn.linear_model import LinearRegression

class CalciumV(Trader):
    def __init__(self):
        super(Trader).__init__()
        self.lpmax = np.zeros(50)
        self.first = True
        self.window_length = 35
        self.score_table = np.array([[]]*50)
        self.mean_start = np.zeros(50)
        self.scaled_z_thresh = 0.4
        self.cur_pos = np.zeros(50)
    
    @export
    def make_pos(self,prcSoFar: np.ndarray):
        _, t = prcSoFar.shape
        current_price = prcSoFar[:,-1]
        lpmax = 10_000/current_price
        if self.first:
            self.first = False
            self.hot_start_prep(prcSoFar)
            
        price_window = prcSoFar[:,t-self.window_length-1:t]
        scaled_z,score= self.do_one_update(price_window)
        score /= np.abs(score)
        
        self.cur_pos = np.where(scaled_z > self.scaled_z_thresh,score*lpmax,self.cur_pos)
        return self.cur_pos

    def hot_start_prep(self,prcSoFar: np.ndarray):
        _, t = prcSoFar.shape
        for i in range(self.window_length+1,t):
            price_window = prcSoFar[:,i-self.window_length-1:i]
            scaled_z,score= self.do_one_update(price_window)
    
    def do_one_update(self,price_window: np.ndarray):
        model = LinearRegression(fit_intercept=True)
        i = 0
        window_length = self.window_length
        model.fit(np.linspace(i-window_length-1,i,window_length+1).reshape(-1,1),price_window.T)

        score = model.coef_.flatten()
        self.score_table = np.append(self.score_table,score[:,np.newaxis],axis=1)
        std = np.zeros(50)
        mean = np.zeros(50)
        for i in range(50):
            start = int( self.mean_start[i] )
            pruned_table = self.score_table[i,max(0,start-10):]
            std[i] = pruned_table.std()
            mean[i] = pruned_table.mean()
        z = ( score-mean ) 
        z[std != 0] /= std[std != 0]

        scaled_z = np.abs(z/3)
        scaled_z = np.where(scaled_z > 1, 1, scaled_z)
        self.mean_start = np.where(scaled_z > self.scaled_z_thresh, i, self.mean_start)
        return scaled_z,score