import os
import numpy as np
import pandas as pd
from model_initialiser import predict_latest

#EXAMPLE USAGE OF THE PIPELINE
def main():

    base_dir = os.path.dirname(os.path.abspath(__file__))
    prices_path = os.path.join(base_dir, 'prices.txt')
    prices_full = pd.read_csv(prices_path, sep=r'\s+', header=None).values.T
    n_inst, T_full = prices_full.shape

    t_use = 170

    """
    READ ME: 
    This is for simulating the actual inference calls, T is essentially a cutoff value for the 2D array which gets passed into the predict_latest function.
    Inference buidls up a cache of 20 features, this is the accumluaiton of 100 sliding windows of the 50 instruments.
    To keep inference to 0.1 seconds 

    Because our caching logic lives in module‐level variables inside the running Python process, 
    you must call predict_latest repeatedly within the same interpreter session if you want to reuse that cache.
    Else the model will repopulate caches every call which wil lead to consistet inference of > 2 seconds.

    """

    seq_len = 20
    win_len = 100

    model_path = os.path.join(base_dir, 'bilstm_self2.pth')

    prices_slice = prices_full[:, :t_use+1]
    preds = predict_latest(
        prices_slice,
        model_path=model_path,
        seq_len=seq_len,
        win_len=win_len,
        device='cpu'
    )  

    print(f"Regime predictions at t={t_use}:")
    for inst, r in enumerate(preds):
        label = 'Bear' if r == 0 else ('Bull' if r == 2 else 'Neutral')
        print(f"  Instrument {inst:2d}: regime={r} ({label})")

if __name__ == '__main__':
    main()

#Actual usage of the function you will want to use in intergration:

# simple_infer.py
# ----------------
# A minimal wrapper that takes a 2D NumPy price array and returns regime predictions.

from model_initialiser import predict_latest


def infer_regimes(prices, model_path,
                   seq_len=20, win_len=100,
                   device='cpu'):
    """
    prices      : np.ndarray of shape (n_instruments, T)
    model_path  : string path to .pth checkpoint
    seq_len     : number of feature snapshots (LSTM sequence length)
    win_len     : number of bars per feature snapshot
    device      : 'cpu' or 'cuda'

    returns     : np.ndarray of shape (n_instruments,), regime labels

    I suggest against changing `seq_len` or `win_len`
    'prices' is the 2D array of historical prices, by nature the model and pipeline will take the most recent 100 prices per instrument.
    and predict the regime for the most recent bar.
    """
    return predict_latest(
        prices,
        model_path=model_path,
        seq_len=seq_len,
        win_len=win_len,
        device=device
    )

# Example usage:
# import numpy as np
# from simple_infer import infer_regimes
# prices = np.load('prices.npy')  # shape (50, T)
# preds  = infer_regimes(prices, 'bilstm_self2.pth')

