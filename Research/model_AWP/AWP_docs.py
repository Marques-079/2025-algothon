import numpy as np
import torch
import torch.nn as nn

# load weights into a fresh model
def load_model(feat_dim, hidden_dim, num_layers, num_tags, dropout, checkpoint_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = BiLSTMTagger(feat_dim, hidden_dim, num_layers, num_tags, dropout)
    m.load_state_dict(torch.load(checkpoint_path, map_location=device))
    m.to(device).eval()
    return m, device

# your inference function
def predict_regimes(model, device, features: np.ndarray) -> np.ndarray:
    """
    Run inference on a single instrument's full feature sequence.

    Args:
      model: the BiLSTMTagger in eval() mode
      device: torch.device
      features: np.ndarray of shape (T, D) where
        - T is the number of time-steps (e.g. SEQ_LEN)
        - D is the number of feature columns (len(feat_cols))
      (must match the same preprocessing you used during training)

    Returns:
      preds: np.ndarray of shape (T,) containing regime labels {0,1,2}
    """
    # 1) numpy → torch, add batch dim
    x = torch.from_numpy(features).float().to(device)
    x = x.unsqueeze(0)              # shape → (1, T, D)

    # 2) forward pass (no grad)
    with torch.no_grad():
        logits = model(x)           # (1, T, num_tags)
        preds  = logits.argmax(dim=2)  # (1, T)

    return preds.cpu().numpy().squeeze(0)  # → (T,)



# ── C) Example usage ──────────────────────────────────────────────────────────
# (run this anywhere after training)
model, dev = load_model("bilstm_tagger.pth")
# suppose you have a new instrument's cleaned feature array `new_feats` shaped (SEQ_LEN, D):
#     new_feats = <your np.ndarray>
new_feats = np.random.rand(1000, 10)  # Example feature array with 1000 time-steps and 10 features
preds = predict_regimes(model, dev, new_feats)
print("Predicted regimes shape:", preds.shape)
print("First 10 predictions:", preds[:10])

