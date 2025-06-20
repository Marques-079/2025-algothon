from math import fabs
from pandas import DataFrame
from requests import ReadTimeout
from scipy.special import softmax
import numpy as np

##### TODO #########################################
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

nInst = 50
currentPos = np.zeros(nInst)

# getMyPosition = lambda x: chatgpt(x)
getMyPosition = lambda x: random(x)

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
  """
  Computes the sigmoid (logistic) activation function.

  Args:
    x (np.ndarray or scalar): The input value(s). Can be a single number
                              or a NumPy array.

  Returns:
    np.ndarray or scalar: The sigmoid of x, with values between 0 and 1.
  """
  return 1 / (1 + np.exp(-x))

first = True
lpos = None
def random(prcSoFar: np.ndarray):
    global first
    global lpos
    nInst, t = prcSoFar.shape
    lp = prcSoFar[:,-1]
    lpmax = 10_000/lp

    if first:
        first = False
        lpos = lpmax
        return lpmax
    kernel = np.array([-1,1])  #profit kernel
    convolved_array = np.zeros((nInst,t))
    for i in range(50):
        convolved_array[i,:] = np.convolve(prcSoFar[i,:], kernel, mode='same')  # or 'same', 'full'
    
    d_dist = []
    n = 40
    for i in range(1,n+1):
        diff = convolved_array[:,-i:].sum(axis=1)/i
        d_dist.append(diff)

    d_dist = np.array(d_dist)
    m = t*sigmoid( np.arange(1,n+1) )
    m = m[:,np.newaxis]
    d_dist *= m
    index_delta = d_dist.sum()/n

    r = -index_delta*n
    lpos = r
    return r


#CHATGPT CODE
def chatgpt(prcSoFar):
    nInst, t = prcSoFar.shape
    # need at least 21 days to compute metrics
    if t < 21:
        return np.zeros(nInst, dtype=int)

    # 1) compute log-returns over the last 20 days
    logp = np.log(prcSoFar)
    rets20 = logp[:, -1] - logp[:, -21]        # 20-day log return
    # normalize to per-day
    mom20 = rets20 / 20.0

    # 2) compute realized vol over last 20 days
    daily_rets = logp[:, -20:] - logp[:, -21:-1]
    vol20 = np.std(daily_rets, axis=1, ddof=1)
    # avoid div by zero
    vol20[vol20 == 0] = 1e-8

    # 3) mean-reversion: deviation from 5-day moving average
    sma5 = np.mean(prcSoFar[:, -5:], axis=1)
    dev = (prcSoFar[:, -1] - sma5) / sma5   # positive => current > avg

    # 4) build a composite signal
    #    + momentum, - small mean-reversion
    signal = mom20 - 0.5 * dev

    # 5) risk‐weight the signals by 1/vol
    risk_signal = signal / vol20

    # 6) pick the top/bottom 5 names
    longs  = np.argsort(risk_signal)[-5:]
    shorts = np.argsort(risk_signal)[:5]

    # 7) allocate equal risk budget: target $10k per side
    #    so $2k per position roughly (10k / 5)
    target_notional_per = 10000.0 / 5.0

    pos = np.zeros(nInst, dtype=int)
    prices = prcSoFar[:, -1]
    for i in longs:
        pos[i] = int(target_notional_per / prices[i])
    for i in shorts:
        pos[i] = -int(target_notional_per / prices[i])

    return pos

def template(prcSoFar):
    global currentPos
    (nins, nt) = prcSoFar.shape
    if (nt < 2):
        return np.zeros(nins)
    lastRet = np.log(prcSoFar[:, -1] / prcSoFar[:, -2])
    lNorm = np.sqrt(lastRet.dot(lastRet))
    lastRet /= lNorm
    rpos = np.array([int(x) for x in 5000 * lastRet / prcSoFar[:, -1]])
    currentPos = np.array([int(x) for x in currentPos+rpos])
    return currentPos