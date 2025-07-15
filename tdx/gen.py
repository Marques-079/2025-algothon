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



def R1Generator(I=50):
    # Generate signals of varying granularity
    signals = [
        gen_signals(15, instruments=I, base=True, std=0.9)[1],
        gen_signals(500, instruments=I, std=0.7, m=0.3)[1],
        gen_signals(1000, instruments=I, std=0.7, m=0.9)[1],
    ]
    minvals = np.min(signals,axis=0) 
    minvals = np.where(minvals < 0,minvals,0)
    signals -= minvals
    
    combined = np.sum(signals,axis=0)
    return combined.T

def R1Icorr(RealI=25,rev = True):
    Reals = R1Generator(RealI).T
    Generated = []
    for i in range(50-RealI):
        sample_idx = np.random.choice(np.arange(RealI),3,replace=False)
        samples = Reals[:,sample_idx]
        m = np.mean(samples,axis=1)
        if rev and np.random.choice([0,1],1) == 0:
            m = m[::-1]
        Generated.append(m)
    
    stacked = np.stack(Generated).T
    combined = np.concatenate(( stacked,Reals ),axis=1)
    return combined.T

def R2Generator(I=50):
    # Generate signals of varying granularity
    signals = [
        gen_signals(15, instruments=I, base=True, std=1)[1],
        gen_signals(300, instruments=I, std=0.8)[1],
        gen_signals(500, instruments=I, std=0.8, m=0.3)[1],
        gen_signals(1000, instruments=I, std=0.8, m=0.9)[1],
    ]
    minvals = np.min(signals,axis=0) 
    minvals = np.where(minvals < 0,minvals,0)
    signals -= minvals
    
    combined = np.sum(signals,axis=0)
    return combined.T

def RxIcorr(source_gen,RealI=25,rev = True):
    Reals = source_gen(RealI).T
    Generated = []
    for i in range(50-RealI):
        sample_idx = np.random.choice(np.arange(RealI),3,replace=False)
        samples = Reals[:,sample_idx]
        m = np.mean(samples,axis=1)
        if rev and np.random.choice([0,1],1) == 0:
            m = m[::-1]
        Generated.append(m)
    
    stacked = np.stack(Generated).T
    combined = np.concatenate(( stacked,Reals ),axis=1)
    return combined.T