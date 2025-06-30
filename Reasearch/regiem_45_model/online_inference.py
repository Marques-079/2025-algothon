"""
online_inference
================
Real-time engine for regime prediction.

Usage
-----
>>> import numpy as np
>>> from regiem_450_model.online_inference import OnlineModel
>>>
>>> prc_hist = np.loadtxt("prices.txt")       # shape (n_inst, nt)
>>> model    = OnlineModel(n_inst=prc_hist.shape[0])
>>>
>>> model.warmup(prc_hist, till_t=159)        # fill both caches
>>> for t in range(160, prc_hist.shape[1]):
...     preds = model.update(prc_hist[:, t])
...     if preds is not None:
...         print(f"t={t}  probs={preds:.4f}")
"""
from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import torch

# ––– internal imports ––––––––––––––––––––––––––––––––––––––––––––––––––––– #
from utils.ring_buffer import RingBuffer
from regiem_45_model.model import LSTMRegime          # your network
from regiem_45_model.four_pipeline import _build_feature_dict


# ------------------------------------------------------------------------- #
# 1) small functional wrapper around your heavy pipeline
# ------------------------------------------------------------------------- #
FEATURE_ORDER = list(_build_feature_dict(np.ones(100)).keys())  # 59 names

def compute_features(price_window: np.ndarray) -> np.ndarray:
    """Return the *latest-step* feature vector (len = 59)."""
    fd   = _build_feature_dict(price_window)

    vals = []
    for name in FEATURE_ORDER:
        feat = fd[name]
        # accept scalar, 1-D, or even masked arrays
        if np.ndim(feat) == 0:
            vals.append(feat)
        else:
            vals.append(feat[-1])
    vec = np.asarray(vals, dtype=np.float32)


    vec  = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)  # safety
    return vec                                                   # shape (59,)


# ------------------------------------------------------------------------- #
# 2) live-model class
# ------------------------------------------------------------------------- #
ARTIFACT_DIR = Path(__file__).with_suffix("").parent / "artifacts"
MODEL_FILE   = ARTIFACT_DIR / "regime_lstm_state_dict.pth"
SCALER_FILE  = ARTIFACT_DIR / "scaler.pkl"

PRICE_WINDOW = 100
FEAT_WINDOW  = 60
F_DIM        = len(FEATURE_ORDER)


class OnlineModel:
    """Keeps per-instrument rolling caches and produces predictions."""

    # .......................... ctor .................................... #
    def __init__(self, n_inst: int, device: str | torch.device = "cpu"):
        self.n_inst = n_inst
        self.device = torch.device(device)

        # rolling caches -------------------------------------------------- #
        self._price_buf = [RingBuffer(PRICE_WINDOW)           for _ in range(n_inst)]
        self._feat_buf  = [RingBuffer(FEAT_WINDOW, (F_DIM,))  for _ in range(n_inst)]

        # artefacts ------------------------------------------------------- #
        self.scaler = joblib.load(SCALER_FILE)
        self.model  = self._load_net().to(self.device).eval()

    # .......................... public API .............................. #
    def warmup(self, prices_hist: np.ndarray, till_t: int) -> None:
        """
        Replay history up to *till_t* (inclusive) so that the next call
        to `.update()` will yield a prediction immediately.

        *prices_hist* shape = (n_inst, nt).
        """
        start_t = max(0, till_t - PRICE_WINDOW - FEAT_WINDOW + 1)
        for t in range(start_t, till_t + 1):
            self.update(prices_hist[:, t])

    def update(self, new_prices: np.ndarray) -> np.ndarray | None:
        """
        Push one timestep of close prices (shape = n_inst,).

        Returns
        -------
        • `np.ndarray` of shape (n_inst,) with inference probabilities
          *if* both caches are full.
        • `None` during the warm-up period.
        """
        # 1) --- update caches ------------------------------------------- #
        for i in range(self.n_inst):
            self._price_buf[i].push(new_prices[i])

            if self._price_buf[i].is_full:
                window = self._price_buf[i].view()
                if window.shape[0] == PRICE_WINDOW:
                    feats = compute_features(self._price_buf[i].view())
                    self._feat_buf[i].push(feats)

        # 2) --- are we ready to infer? ---------------------------------- #
        if not all(buf.is_full for buf in self._feat_buf):
            return None

        # 3) --- build model input tensor -------------------------------- #
        X = np.stack([buf.view() for buf in self._feat_buf])      # (n_inst, 60, 59)

        # model was trained on *scaled* features ------------------------- #
        batch, seq, feat = X.shape

        n_inst, seq, feat = X.shape              # feat == 59
        X_rows  = X.reshape(-1, feat)            # (n_inst*60 , 59)
        X_rows  = self.scaler.transform(X_rows)  # scale each row
        X_scl   = X_rows.reshape(n_inst, seq, feat)

        inp     = torch.from_numpy(X_scl).float().to(self.device)
        with torch.no_grad():
            logits = self.model(inp)                              # (n_inst, 1) or (n_inst,)
            probs  = torch.sigmoid(logits).squeeze(-1).cpu().numpy()

        return probs                                              # (n_inst,)

    # .......................... helpers ................................. #
    @staticmethod
    def _load_net() -> LSTMRegime:
        net = LSTMRegime(
            n_features=F_DIM, hidden_dim=64, num_layers=1, dropout=0.2
        )                   
        state = torch.load(MODEL_FILE, map_location="cpu")
        net.load_state_dict(state)
        return net
