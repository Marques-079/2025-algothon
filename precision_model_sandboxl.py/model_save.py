import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split

# ── 1) Hyperparameters ─────────────────────────────────────────────────────────
SEQ_LEN     = 740
TRAIN_LEN   = 500
BATCH_SIZE  = 10
HIDDEN_SIZE = 64
NUM_LAYERS  = 2
DROPOUT     = 0.2
LR          = 1e-3
NUM_EPOCHS  = 15
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── 2) Load & clean ─────────────────────────────────────────────────────────────
df = pd.read_csv("features_all_models4.csv")
df = (
    df
    .groupby("inst", group_keys=False)
    .apply(lambda g: g.iloc[100:])   # drop first 100 warm-ups
    .reset_index(drop=True)
)

# recompute SEQ_LEN from the cleaned data:
seq_lens = df.groupby("inst").size()
SEQ_LEN  = int(seq_lens.max())        # should be 640 in your case
print("Detected sequence length per instrument:", SEQ_LEN)

# now build X and Y arrays with the correct shape
n_inst    = df["inst"].nunique()
feat_cols = [c for c in df.columns if c not in ("inst","time","true_regime")]

X = np.zeros((n_inst, SEQ_LEN, len(feat_cols)), dtype=np.float32)
Y = np.zeros((n_inst, SEQ_LEN),               dtype=np.int64)

for inst in range(n_inst):
    sub = df[df["inst"]==inst].reset_index(drop=True)
    assert len(sub)==SEQ_LEN              # sanity check
    X[inst,:,:] = sub[feat_cols].values
    Y[inst,:]   = sub["true_regime"].values

NUM_TAGS = int(Y.max())+1


# ── 3) Train/test split along time axis ────────────────────────────────────────
X_train = torch.tensor(X[:, :TRAIN_LEN, :])
Y_train = torch.tensor(Y[:, :TRAIN_LEN])
X_full  = torch.tensor(X)      # for full‐sequence inference
Y_full  = Y                   # numpy for plotting

# ── 4) Dataset & DataLoader ───────────────────────────────────────────────────
class SeqTagDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return self.X.size(0)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

train_ds     = SeqTagDataset(X_train, Y_train)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# ── 5) BiLSTM tagger ────────────────────────────────────────────────────────────
class BiLSTMTagger(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_layers, num_tags, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            feat_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim*2, num_tags)

    def forward(self, x):
        out, _ = self.lstm(x)     # (B, T, 2*H)
        return self.fc(out)       # (B, T, num_tags)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = BiLSTMTagger(
    feat_dim   = X.shape[2],
    hidden_dim = HIDDEN_SIZE,
    num_layers = NUM_LAYERS,
    num_tags   = NUM_TAGS,
    dropout    = DROPOUT
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ── 6) Train ───────────────────────────────────────────────────────────────────
for epoch in range(1, NUM_EPOCHS+1):
    model.train()
    total_loss = 0.0
    for feats, tags in train_loader:
        feats, tags = feats.to(device), tags.to(device)
        logits      = model(feats)              # (B, T, C)
        loss        = criterion(
            logits.view(-1, NUM_TAGS),         # (B*T, C)
            tags.view(-1)                      # (B*T,)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch:02d} — Avg Loss: {total_loss/len(train_loader):.4f}")

# ── 7) Inference & two-panel plotting ──────────────────────────────────────────
def get_segments(reg):
    changes = np.flatnonzero(reg[1:] != reg[:-1])
    starts  = np.concatenate(([0], changes+1))
    ends    = np.concatenate((changes, [len(reg)-1]))
    return list(zip(starts, ends, reg[starts]))

true_cmap = ListedColormap(["#ff0000","#808080","#00ff00"])
pred_cmap = ListedColormap(["#cc0000","#444444","#00cc00"])

model.eval()
with torch.no_grad():
    logits_full = model(X_full.to(device))         # (50,740,C)
    preds_full  = logits_full.argmax(dim=2).cpu().numpy()

for inst in range(n_inst):
    true_seq = Y_full[inst]
    pred_seq = preds_full[inst]
    price    = price_df.iloc[100:100+SEQ_LEN, inst].values

    fig, (ax1, ax2) = plt.subplots(2,1,
                                  sharex=True,
                                  figsize=(12,6))

    # TRUE regimes
    for s,e,lbl in get_segments(true_seq):
        ax1.axvspan(s, e, color=true_cmap(lbl), alpha=0.5, linewidth=0)
    ax1.plot(price, 'k-', label='Price')
    ax1.set_title(f"Inst {inst} — TRUE regimes")
    ax1.legend(loc='upper right')

    # PREDICTED regimes
    for s,e,lbl in get_segments(pred_seq):
        ax2.axvspan(s, e, color=pred_cmap(lbl), alpha=0.5, linewidth=0)
    ax2.plot(price, 'k-', label='Price')
    ax2.set_title(f"Inst {inst} — PREDICTED regimes")
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()
