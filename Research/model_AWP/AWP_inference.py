import numpy as np
import torch
import torch.nn as nn
import os

FEATURE_DIM = 9
HIDDEN_SIZE = 64
NUM_LAYERS  = 2
NUM_TAGS    = 3
DROPOUT     = 0.2

class BiLSTMTagger(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_layers, num_tags, dropout):
        super().__init__()
        self.lstm = nn.LSTM(feat_dim, hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=dropout)
        self.fc = nn.Linear(hidden_dim*2, num_tags)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)

def load_model(checkpoint_name: str, device=None):
    # 1) resolve script‐dir
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path  = os.path.join(script_dir, checkpoint_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Couldn’t find {ckpt_path!r}")

    # 2) device setup
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3) instantiate
    model = BiLSTMTagger(FEATURE_DIM, HIDDEN_SIZE, NUM_LAYERS, NUM_TAGS, DROPOUT)

    # 4) **use the full path** here**
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)

    # 5) move to device & eval
    model.to(device).eval()
    return model, device


def predict_regimes(model, device, features: np.ndarray):
    x = torch.from_numpy(features).float().unsqueeze(0).to(device)
    with torch.no_grad():
        preds = model(x).argmax(dim=-1)
    return preds.cpu().numpy().squeeze(0)

# if __name__=="__main__":
#     model, dev = load_model("bilstm_tagger.pth")
#     dummy = np.random.randn(640, FEATURE_DIM).astype(np.float32)
#     out   = predict_regimes(model, dev, dummy)
#     print(out.shape, out[:10])
