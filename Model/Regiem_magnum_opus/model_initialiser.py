import time
import numpy as np
import torch
import torch.nn as nn
from pipeline import infer_from_array   # your unchanged 100-bar feature extractor

# ─── Module-level cache/state ────────────────────────────────────────────────
_last_prices   = None         # np.ndarray shape (n_inst, last_T)
_last_T        = None         # int
_feature_cache = None         # list of np.ndarray shape (n_inst,9)
# ─────────────────────────────────────────────────────────────────────────────

class RegimeBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=True)
        self.fc   = nn.Linear(hidden_size*2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out     = out[:, -1, :]
        return self.fc(out)

def predict_latest(prices: np.ndarray,
                   model_path: str,
                   seq_len:    int = 20,
                   win_len:    int = 100,
                   device:     str = "cpu") -> np.ndarray:
    """
    prices:  (n_inst, T) historic closes
    Returns: (n_inst,) regimes at the most recent bar,
             caching the last `seq_len` feature‐snapshots
             so only new bars trigger infer_from_array calls.
    """
    global _last_prices, _last_T, _feature_cache

    t_start = time.perf_counter()
    device  = torch.device(device)
    n_inst, T = prices.shape
    assert T >= win_len + seq_len - 1, "need at least win_len+seq_len−1 bars"

    # ─── 1) Load (or reuse) your trained model ────────────────────────────────
    t0 = time.perf_counter()
    if not hasattr(predict_latest, "_model"):
        m = RegimeBiLSTM(9, 64, 2, 3, 0.2).to(device)
        sd = torch.load(model_path, map_location=device)
        m.load_state_dict(sd)
        m.eval()
        predict_latest._model = m
    model = predict_latest._model
    t1 = time.perf_counter()
    print(f"[load model]           {t1-t0:.4f}s")

    # ─── 2) Build or update the seq_len-long feature cache ────────────────────
    t_cache_start = time.perf_counter()
    fresh = (_last_prices is None
             or _last_T is None
             or T < _last_T
             or not np.array_equal(prices[:, :_last_T], _last_prices))

    if fresh:
        # First call or non-contiguous history: rebuild entire cache
        _feature_cache = []
        for i, t in enumerate(range(T - seq_len, T), start=1):
            ti = time.perf_counter()
            feats = infer_from_array(prices, timestep=t, win_len=win_len)
            tf = time.perf_counter()
            _feature_cache.append(feats)
            print(f"  [cache {i:2d}/{seq_len}] infer_from_array: {tf-ti:.4f}s")
    else:
        # Continuation: only compute for newly appended bars
        for t in range(_last_T, T):
            ti = time.perf_counter()
            feats = infer_from_array(prices, timestep=t, win_len=win_len)
            tf = time.perf_counter()
            _feature_cache.pop(0)
            _feature_cache.append(feats)
            print(f"  [cache update] t={t}: infer {tf-ti:.4f}s")
    t_cache_end = time.perf_counter()
    print(f"[build cache]          {t_cache_end-t_cache_start:.4f}s total")

    # Update stored history
    _last_prices = prices.copy()
    _last_T      = T

    # ─── 3) Stack cache and move to device ────────────────────────────────────
    t2 = time.perf_counter()
    stacked = np.stack(_feature_cache, axis=0)    # (seq_len, n_inst, 9)
    seqs    = np.transpose(stacked, (1, 0, 2))    # (n_inst, seq_len, 9)
    Xb      = torch.from_numpy(seqs).float().to(device)
    t3 = time.perf_counter()
    print(f"[stack & to device]    {t3-t2:.4f}s")

    # ─── 4) Forward pass through BiLSTM ───────────────────────────────────────
    t4 = time.perf_counter()
    with torch.no_grad():
        logits = model(Xb)                        # (n_inst, 3)
        preds  = logits.argmax(dim=1).cpu().numpy()
    t5 = time.perf_counter()
    print(f"[inference]            {t5-t4:.4f}s")

    # ─── Total ────────────────────────────────────────────────────────────────
    print(f"[TOTAL predict_latest] {t5-t_start:.4f}s\n")
    return preds
