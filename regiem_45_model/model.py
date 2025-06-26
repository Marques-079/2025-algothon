import torch
import torch.nn as nn

class LSTMRegime(nn.Module):
    """
    Simple 1-layer (or multi-layer) LSTM followed by a sigmoid head.
    """

    def __init__(self, n_features: int,
                 hidden_dim: int = 64,
                 num_layers: int = 1,
                 dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0)
        )
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch, seq_len, n_features)
        _, (h_n, _) = self.lstm(x)       # h_n : (num_layers, batch, hidden)
        h_last = self.drop(h_n[-1])      # take last layer’s hidden
        logits = self.fc(h_last)
        return torch.sigmoid(logits).squeeze(1)  # (batch, )
