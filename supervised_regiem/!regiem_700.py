import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.pipeline import make_pipeline
from pre_proc_pipeline import pipeline_regiem
from pre_proc_labelling_long import plot_all_regimes_long

'''

 ▗▄▄▖▗▖ ▗▖▗▄▄▖ ▗▄▄▖ ▗▄▄▄▖▗▖  ▗▖▗▄▄▄▖▗▖ ▗▖  ▗▖    ▗▄▄▖ ▗▄▄▖  ▗▄▖ ▗▖ ▗▖▗▄▄▄▖▗▖  ▗▖
▐▌   ▐▌ ▐▌▐▌ ▐▌▐▌ ▐▌▐▌   ▐▛▚▖▐▌  █  ▐▌  ▝▚▞▘     ▐▌ ▐▌▐▌ ▐▌▐▌ ▐▌▐▌▗▞▘▐▌   ▐▛▚▖▐▌
▐▌   ▐▌ ▐▌▐▛▀▚▖▐▛▀▚▖▐▛▀▀▘▐▌ ▝▜▌  █  ▐▌   ▐▌      ▐▛▀▚▖▐▛▀▚▖▐▌ ▐▌▐▛▚▖ ▐▛▀▀▘▐▌ ▝▜▌
▝▚▄▄▖▝▚▄▞▘▐▌ ▐▌▐▌ ▐▌▐▙▄▄▖▐▌  ▐▌  █  ▐▙▄▄▖▐▌      ▐▙▄▞▘▐▌ ▐▌▝▚▄▞▘▐▌ ▐▌▐▙▄▄▖▐▌  ▐▌
                                                                                
                                                                                
Meme model can wait LOL                                                                      
'''

df = pipeline_regiem()            

label_frames = []
for inst, inst_df in df.groupby(level="inst", sort=False):

    labels = plot_all_regimes_long(len(inst_df), False, inst)

    valid_idx = inst_df.index[: len(labels) ]

    s = pd.Series(labels, index=valid_idx, name="regime")

    label_frames.append(s)

regimes = pd.concat(label_frames) 

df2 = df.copy()
df2["regime"] = regimes         
df2["target"] = (df2.groupby(level="inst")["regime"].shift(-1))

df2 = df2.dropna(subset=["target"])

X = df2.drop(columns=["regime","target"])
y = df2["target"].astype(int)

print("X:", X.shape, "y:", y.shape)

train_X_parts, train_y_parts = [], []
test_X_parts,  test_y_parts  = [], []

for inst, X_inst in X.groupby(level='inst', sort=False):
    idx = X_inst.index
    train_idx, test_idx = idx[:500], idx[500:]
    train_X_parts.append(X.loc[train_idx])
    train_y_parts.append(y.loc[train_idx])
    test_X_parts .append(X.loc[test_idx])
    test_y_parts .append(y.loc[test_idx])

X_train = pd.concat(train_X_parts)
y_train = pd.concat(train_y_parts)
X_test  = pd.concat(test_X_parts)
y_test  = pd.concat(test_y_parts)

#Mapping 2.0 -> 1.0 for clarity
y_train = (y_train == 2).astype(int)
y_test  = (y_test  == 2).astype(int)

X_train, y_train = shuffle(X_train, y_train, random_state=42)

print("Before scaling →", "X_train:", X_train.shape, "X_test:", X_test.shape)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled  = scaler.transform(X_test)      


'''
Dataset Context:

X_train: (25000, 59) X_test: (6950, 59)
59 features,
Labels: 0 -> Bear, 1 -> Bull
Derived from only close price data over 50 instruments, we train on 500 timesteps of data per instrument
Models Aim: We want to maximise consistency in regiem identifcation, regiems often last around 150 time steps long, but can range from 30 to 500.

Acess data by (all data is shuffled, segregated by bound 500):
X_train_scaled : Training set first 500 timesteps of each instrument
y_train : Labelled data for X_train_scaled already aligned


X_test_scaled : Test set last 250 timesteps of each instrument
y_test : Labelled data for X_test_scaled already aligned

'''
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ─── 0) Assume:
#     X: DataFrame with MultiIndex (inst, time) and 59 feature columns
#     y: Series  with same MultiIndex, values in {1,2}

# ─── 1) Build per‐instrument arrays (750 timesteps each) ───────────────────
raw_X_parts = []
raw_y_parts = []
for inst, grp in X.groupby(level='inst', sort=False):
    # take exactly 750 rows
    arr  = grp.iloc[:750].values                 # shape (750, 59)
    labs = (y.loc[grp.index[:750]] == 2).astype(np.float32).values  # map 1→0,2→1
    raw_X_parts.append(arr)
    raw_y_parts.append(labs)

X_all = np.stack(raw_X_parts).astype(np.float32)  # (n_inst, 750, 59)
y_all = np.stack(raw_y_parts).astype(np.float32)  # (n_inst, 750)

n_inst, T, F = X_all.shape

# ─── Set split index ────────────────────────────────────────────────────────
train_steps = 700   # use first 700 for training, last 50 for testing

# ─── 2) Fit scaler on train portion (first train_steps) and transform all ──
scaler = StandardScaler()
flat_train = X_all[:, :train_steps, :].reshape(-1, F)    # (n_inst*700, 59)
scaler.fit(flat_train)

X_all_scaled = scaler.transform(X_all.reshape(-1, F)) \
                   .reshape(n_inst, T, F) \
                   .astype(np.float32)

# ─── 3) Create sliding windows (L=60) and split into train/test ───────────
L = 60
Xw_train, yw_train = [], []
Xw_test,  yw_test  = [], []

for inst_idx in range(n_inst):
    series = X_all_scaled[inst_idx]
    labels = y_all[inst_idx]
    for i in range(T - L + 1):
        window = series[i : i + L]
        lab    = labels[i + L - 1]
        end_t  = i + L - 1

        if end_t < train_steps:
            Xw_train.append(window); yw_train.append(lab)
        else:
            Xw_test.append(window);  yw_test.append(lab)

X_seq_train = np.stack(Xw_train)             # (n_train_windows, 60, 59)
y_seq_train = np.array(yw_train, dtype=np.float32)
X_seq_test  = np.stack(Xw_test)              # (n_test_windows, 60, 59)
y_seq_test  = np.array(yw_test,  dtype=np.float32)

print("Train windows:", X_seq_train.shape, y_seq_train.shape)
print("Test  windows:", X_seq_test.shape,  y_seq_test.shape)


# ─── 4) Dataset & DataLoader ───────────────────────────────────────────────
class RegimeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)       # FloatTensor
        self.y = torch.from_numpy(y)       # FloatTensor
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

batch_size = 256
train_loader = DataLoader(RegimeDataset(X_seq_train, y_seq_train),
                          batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(RegimeDataset(X_seq_test,  y_seq_test),
                          batch_size=batch_size, shuffle=False)

# ─── 5) Define LSTM model ───────────────────────────────────────────────────
class LSTMRegime(nn.Module):
    def __init__(self, n_features, hidden_dim=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size   = n_features,
            hidden_size  = hidden_dim,
            num_layers   = num_layers,
            batch_first  = True,
            dropout      = (dropout if num_layers>1 else 0.0)
        )
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        h_last        = h_n[-1]            # (batch, hidden_dim)
        h_last        = self.drop(h_last)
        return torch.sigmoid(self.fc(h_last)).squeeze(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = LSTMRegime(n_features=F).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
criterion = nn.BCELoss()

# ─── 6) Train on first 600, then test on next 150 ─────────────────────────
num_epochs = 20
for epoch in range(1, num_epochs+1):
    model.train()
    total_loss = 0.0
    for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]"):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss  = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch {epoch:02d} — Train Loss: {avg_loss:.4f}")

# ─── 7) Evaluate on test set ───────────────────────────────────────────────
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb     = xb.to(device), yb.to(device)
        pred_labels = (model(xb) > 0.5).float()
        correct    += (pred_labels == yb).sum().item()
        total      += yb.numel()
test_acc = correct / total
print(f"\nTest Accuracy: {test_acc:.4f}")

# ─── 8) Ablation at inference ──────────────────────────────────────────────
negative_feats = [
    'donch_pct_50','macd_line','streak_down','velocity','adx_14',
    'log_price','close','price_minus_sma_50','sma_12_26_diff',
    'lr_std_100','bb_width_30','percent_b_100','std_20','std_10','lr_std_30'
]
all_feats = list(X.columns)
neg_idx   = [all_feats.index(f) for f in negative_feats]

X_seq_test_ablate = X_seq_test.copy()
X_seq_test_ablate[..., neg_idx] = 0.0

ablate_loader = DataLoader(
    RegimeDataset(X_seq_test_ablate, y_seq_test),
    batch_size=batch_size, shuffle=False
)

model.eval()
correct_ab, total_ab = 0, 0
with torch.no_grad():
    for xb, yb in ablate_loader:
        xb, yb     = xb.to(device), yb.to(device)
        pred_labels = (model(xb) > 0.5).float()
        correct_ab += (pred_labels == yb).sum().item()
        total_ab   += yb.numel()
ablate_acc = correct_ab / total_ab

print(f"Ablated Test Accuracy: {ablate_acc:.4f}")
print(f"Δ Accuracy: {ablate_acc - test_acc:+.4f}")
