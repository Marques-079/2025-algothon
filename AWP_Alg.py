# AWP_Alg.py

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import pickle

from templates.StandardTemplate import Trader, export
from model_AWP.AWP_inference import load_model, predict_regimes

# ─────────── feature‐computation helpers ────────────
def _ols_slope(y: np.ndarray) -> float:
    t = np.arange(len(y))
    X = np.vstack([t, np.ones_like(t)]).T
    m, _ = np.linalg.lstsq(X, y, rcond=None)[0]
    return m

def _slope_vol_reg(close: np.ndarray, idx: int,
                   slope_win: int = 30, vol_win: int = 100) -> float | int:
    logp = np.log(close)
    slope_ser = (
        pd.Series(logp)
          .rolling(slope_win, min_periods=slope_win)
          .apply(lambda arr: _ols_slope(arr), raw=True)
    )
    rtn       = pd.Series(logp).diff()
    vol_ser   = rtn.rolling(vol_win, min_periods=vol_win).std()

    slope = slope_ser.iat[idx]
    vol   = vol_ser.iat[idx]
    if np.isnan(slope) or np.isnan(vol):
        return np.nan

    med = vol_ser.rolling(100, min_periods=100).median().iat[idx]
    return 2 if (slope > 0 and vol < med) else 0

def compute_regime_features_window(prices_window: np.ndarray) -> np.ndarray:
    """
    prices_window: np.ndarray, shape (n_inst, win_len)
    returns:       np.ndarray, shape (n_inst, 9)
    order: [ma, ema, slope_vol, macd, kalman, fib, psar, zscore, wret]
    """
    n_inst, win_len = prices_window.shape
    idx = win_len - 1
    out = np.full((n_inst, 9), np.nan)
    # weights for weighted return
    sqrt_w = np.arange(1, 46, dtype=float) ** 0.5
    sqrt_w /= sqrt_w.sum()

    for i in range(n_inst):
        close = prices_window[i]
        logp  = np.log(close)

        # 1) MA
        ma_s = pd.Series(logp).rolling(5).mean().iat[idx]
        ma_l = pd.Series(logp).rolling(70).mean().iat[idx]
        ma_reg = 0 if ma_l > ma_s else 2

        # 2) EMA
        ema_s = pd.Series(logp).ewm(span=5, adjust=False).mean().iat[idx]
        ema_l = pd.Series(logp).ewm(span=50,adjust=False).mean().iat[idx]
        ema_reg = 2 if ema_s > ema_l else 0

        # 3) Slope/Vol
        sv_reg = _slope_vol_reg(close, idx)

        # 4) MACD
        macd_line = (pd.Series(logp).ewm(span=50,adjust=False).mean()
                     - pd.Series(logp).ewm(span=90,adjust=False).mean())
        sig_line  = macd_line.ewm(span=40,adjust=False).mean()
        macd_reg  = 2 if macd_line.iat[idx] > sig_line.iat[idx] else 0

        # 5) Kalman filter
        proc_var, meas_var = 0.01, 10.0
        x_est = np.zeros(win_len); P = np.zeros(win_len)
        x_est[0], P[0] = logp[0], 1.0
        for t in range(1, win_len):
            Pp = P[t-1] + proc_var
            K  = Pp / (Pp + meas_var)
            x_est[t] = x_est[t-1] + K*(logp[t] - x_est[t-1])
            P[t]     = (1-K)*Pp
        kalman_reg = 2 if logp[idx] > x_est[idx] else 0

        # 6) Fibonacci
        if idx >= 50:
            win50 = close[idx-49:idx+1]
            hi, lo = win50.max(), win50.min()
            rng = hi - lo
            upper, lower = lo + 0.786*rng, lo + 0.618*rng
            fib_reg = 2 if close[idx]>upper else 0 if close[idx]<lower else 1
        else:
            fib_reg = np.nan

        # 7) PSAR
        psar = np.empty(win_len)
        up, af, max_af = True, 0.01, 0.10
        ep = close[0]; psar[0] = ep
        for t in range(1, win_len):
            psar[t] = psar[t-1] + af*(ep - psar[t-1])
            if up:
                if close[t] < psar[t]:
                    up, psar[t], ep, af = False, ep, close[t], 0.01
                elif close[t] > ep:
                    ep, af = close[t], min(af+0.01, max_af)
            else:
                if close[t] > psar[t]:
                    up, psar[t], ep, af = True, ep, close[t], 0.01
                elif close[t] < ep:
                    ep, af = close[t], min(af+0.01, max_af)
        psar_reg = 2 if close[idx] > psar[idx] else 0

        # 8) Z-score
        ma90 = pd.Series(close).rolling(90).mean().iat[idx]
        sd90 = pd.Series(close).rolling(90).std().iat[idx]
        if np.isnan(ma90) or np.isnan(sd90):
            z_reg = np.nan
        else:
            z = (close[idx] - ma90)/sd90
            z_reg = 2 if z>0.5 else 0 if z<-0.5 else 1

        # 9) Weighted return
        if idx >= 45:
            r      = pd.Series(close).pct_change().iloc[idx-44:idx+1].values
            wret_reg = 2 if np.dot(r, sqrt_w) > 0 else 0 if np.dot(r, sqrt_w) <0 else 1
        else:
            wret_reg = np.nan

        out[i] = [ma_reg, ema_reg, sv_reg, macd_reg, kalman_reg,
                  fib_reg, psar_reg, z_reg, wret_reg]

    return out

# ─────────── internal engine ───────────
HERE      = Path(__file__).parent
REPO_ROOT = HERE.parent

class _RegimeEngine:
    def __init__(self,
                 model_path:     str   = "bilstm_regime_model.pth",
                 col_means_path: str   = "col_means.npy",
                 notional:       float = 10_000,
                 win_len:        int   = 100):
        # try three locations for each file
        def _locate(fname):
            p0 = Path(fname)
            if p0.exists(): return p0
            p1 = HERE/"model_AWP"/fname
            if p1.exists(): return p1
            p2 = REPO_ROOT/fname
            if p2.exists(): return p2
            raise FileNotFoundError(f"Couldn’t locate {fname!r} in:\n  {p0}\n  {p1}\n  {p2}")

        # checkpoint
        ckpt = _locate(model_path)
        # col_means
        cmf  = _locate(col_means_path)

        # load BiLSTM
        self.model, self.device = load_model(ckpt.name)
        # load col_means and sanity-check
        cm = np.load(cmf)
        if cm.ndim != 1 or cm.shape[0] < 9:
            raise ValueError(f"col_means.npy must have at least 9 entries, got {cm.shape}")
        self.col_means = cm[:9]

        self.notional = notional
        self.win_len  = win_len
        self.last_pos = None

    def get_raw(self, prcSoFar: np.ndarray) -> np.ndarray:
        nInst, t = prcSoFar.shape
        if t < self.win_len:
            return np.zeros(nInst, dtype=int)

        window = prcSoFar[:, -self.win_len:]
        feats  = compute_regime_features_window(window)
        # impute
        feats  = np.where(np.isnan(feats), self.col_means, feats)
        return predict_regimes(self.model, self.device, feats)

    def getPosition(self, prcSoFar: np.ndarray) -> np.ndarray:
        regs = self.get_raw(prcSoFar)
        nInst, _ = prcSoFar.shape
        # lazy init
        if self.last_pos is None:
            self.last_pos = np.zeros(nInst, dtype=int)

        signs      = np.where(regs==2, +1, -1)
        last_price = prcSoFar[:,-1]
        sizes      = np.floor(self.notional/last_price).astype(int)
        target     = signs * sizes

        new_pos = self.last_pos.copy()
        prev_sign = np.sign(self.last_pos)
        for i in range(nInst):
            if signs[i] != prev_sign[i]:
                new_pos[i] = target[i]

        self.last_pos = new_pos
        return new_pos

# ─────────── Trader wrapper ───────────
class RegimeTrader(Trader):
    def __init__(self,
                 model_path:     str   = "bilstm_regime_model.pth",
                 col_means_path: str   = "col_means.npy",
                 notional:       float = 10_000,
                 win_len:        int   = 100):
        super().__init__()
        self.engine = _RegimeEngine(model_path, col_means_path, notional, win_len)

    @export
    def getPosition(self, prcSoFar: np.ndarray) -> np.ndarray:
        return self.engine.getPosition(prcSoFar)

# ───────── module‐level instance ─────────
AWP = RegimeTrader()
