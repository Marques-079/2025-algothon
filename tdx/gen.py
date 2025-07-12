import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

def gen_signals(granularity, instruments=50, T=1000,std=1,base=False,m=1):
    n = granularity
    d = instruments
    times = np.linspace(0., T, n)
    dt = times[1] - times[0]
    dB = np.sqrt(dt) * np.random.normal(scale=std,size=(n - 1, d)) * m
    B0 = np.zeros(shape=(1, d))

    B = np.concatenate((B0, np.cumsum(dB, axis=0)), axis=0)

    if base:
        minvals = np.min(B,axis=0) 
        B -= minvals
        B += np.random.randint(20,50,d)

    # interpolate to equalize time steps
    interp = [interp1d(times, B[:, i], kind='linear', fill_value="extrapolate") for i in range(d)]
    times = np.linspace(0., T, T)
    B_interp = np.stack([f(times) for f in interp], axis=1)

    return times, B_interp



def R1Generator():
    I = 50
    # Generate signals of varying granularity
    signals = [
        gen_signals(15, instruments=I, base=True, std=0.9)[1],
        gen_signals(1000, instruments=I, std=0.45, m=0.9)[1],
    ]
    
    combined = np.sum(signals,axis=0)
    return combined.T
