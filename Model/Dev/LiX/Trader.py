from Model.standard_template import Trader,export
import numpy as np
import matplotlib.pyplot as plt

class LiX(Trader):
    def __init__(self):
        super(Trader).__init__()
        self.lpmax = np.zeros(50)
        self.emaRange = 50
        self.emaTable = []
        self.first = True
    
    @export
    def make_pos(self,prcSoFar: np.ndarray):
        _, t = prcSoFar.shape
        self.lpmax = 10_000/prcSoFar[:,-1]
        N = 50
        if self.first:
            self.first = False
            for i in range(0,N):
                self.emaTable.append(self.pregen_ema(prcSoFar,span=2*( i+1 )))

        curr_emas = []
        for i in range(0,N):
            emas = self.emaTable[i]
            new_ema = self.get_next_ema(prcSoFar,span=2*(i+1))
            curr_emas.append(new_ema)
            new_ema = new_ema[:,np.newaxis]
            self.emaTable[i] = np.concat(( emas,new_ema ),axis=1)
        curr_emas = np.array(curr_emas)
        cm_mean = np.mean(curr_emas,axis=0)
        cm_std = np.std(curr_emas,axis=0)
        print(curr_emas.shape,cm_mean.shape,cm_std.shape,cm_std[15],t)

        if t == 999:
            pemas = np.stack(self.emaTable)
            print(pemas.shape)
            ipPema = pemas[:,15,:].T
            emas = ipPema
            convergence_measure = np.mean(emas,axis=1)
            stdT = np.std(emas,axis=1) 
            stdB = -stdT
            std = np.stack([stdT+convergence_measure,stdB+convergence_measure])
            print(emas[-10:],stdT[-10:],convergence_measure[-10:])
            plt.figure(figsize=(10, 5))
            plt.plot(prcSoFar[15,:], label='Noisy Measurements', alpha=0.5)
            plt.plot(emas,alpha=0.1,color="red")
            plt.plot(convergence_measure,label="convergence measure")
            plt.plot(std.T,color="yellow")
            plt.show()

        return self.lpmax 

    def pregen_ema(self,mat,span):
        alpha = 2 / (span + 1)
        nInst,T = mat.shape
        T -= 1
        ema = np.zeros(shape=(nInst,T))
        ema[:,0] = mat[:,0]  # Initialize with first value
        for t in range(1, T):
            ema[:,t] = alpha * mat[:,t] + (1 - alpha) * ema[:,t-1]
        return ema
    
    def get_next_ema(self,mat,span):
        alpha = 2 / (span + 1)
        spandex = int(span/2 - 1)
        ema = self.emaTable[spandex]
        new_ema = alpha * mat[:,-1] + (1 - alpha) * ema[:,-1]
        return new_ema
