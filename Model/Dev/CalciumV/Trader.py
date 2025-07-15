import numpy as np
from Model.standard_template import Trader,export

class CalciumV(Trader):
    def __init__(self):
        super(Trader).__init__()
        self.lpmax = np.zeros(50)
        self.first = True
        self.window_length = 30
        self.score_table = np.array([[]]*50)
        self.mean_start = np.zeros(50)
        self.scaled_z_thresh = 0.3
    
    @export
    def make_pos(self,prcSoFar: np.ndarray):
        _, t = prcSoFar.shape
        current_price = prcSoFar[:,-1]
        lpmax = 10_000/current_price
        if self.first:
            self.first = False
            self.hot_start_prep(prcSoFar)
            
        price_window = prcSoFar[:,t-self.window_length-1:t]
        scaled_z, z,dir = self.do_one_update(price_window)
        self.mean_start = np.where(scaled_z > self.scaled_z_thresh, t, self.mean_start)
        
        ups = ( dir==1 ).sum(1)
        downs = ( dir==-1 ).sum(1)
        stags = ( dir==0 ).sum(1)
        score = np.abs( ups-downs )+stags
        pscore = score/self.window_length
        pscore = np.where(pscore > 0.7, scaled_z , 0)
        new_var = dir.sum(1)
        direction = np.where( new_var != 0, new_var/np.abs(new_var), 0 )
        direction *=pscore

        # means = np.zeros(50)
        # for i in range(50):
        #     start = int(self.mean_start[i])
        #     relevant_range = prcSoFar[i,start-2:]
        #     means[i] = np.mean(relevant_range)
        
        # direction = current_price - means
        # direction = np.where(direction != 0, direction/np.abs(direction),0)

        return direction * lpmax

    def hot_start_prep(self,prcSoFar: np.ndarray):
        _, t = prcSoFar.shape
        for i in range(self.window_length+1,t):
            price_window = prcSoFar[:,i-self.window_length-1:i]
            scaled_z,_,_ = self.do_one_update(price_window)
            self.mean_start = np.where(scaled_z > self.scaled_z_thresh, i, self.mean_start)
    
    def do_one_update(self,price_window: np.ndarray):
        diff_window = np.diff(price_window,axis=1)
        directions = np.where(diff_window != 0, diff_window/np.abs(diff_window),0)

        ups = np.where(diff_window > 0, diff_window, 0).sum(axis=1)
        downs = np.where(diff_window < 0, diff_window, 0).sum(axis=1)
        score = ups+downs
        self.score_table = np.append(self.score_table,score[:,np.newaxis],axis=1)
        std = self.score_table.std(axis=1)
        mean = self.score_table.mean(axis=1)
        z = ( score-mean ) 
        z[std != 0] /= std[std != 0]

        scaled_z = np.abs(z/3)
        scaled_z = np.where(scaled_z > 1, 1, scaled_z)
        scaled_z *= scaled_z
        return scaled_z,z,directions