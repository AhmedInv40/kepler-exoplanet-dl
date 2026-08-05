import torch
import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        self.norm = nn.BatchNorm1d(dim)
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.norm(x)
        out = self.linear1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.linear2(out)
        return out + x


class ResNetTabular(nn.Module):

    def __init__(self, in_features: int, n_classes: int = 3, d_model: int = 128,
                 hidden_dim: int = 256, n_blocks: int = 4, dropout: float = 0.3):
        super().__init__()
        self.input_projection = nn.Linear(in_features, d_model)
        self.blocks = nn.ModuleList([
            ResidualBlock(d_model, hidden_dim, dropout) for _ in range(n_blocks)
        ])
        self.head = nn.Sequential(
            nn.BatchNorm1d(d_model), nn.GELU(), nn.Linear(d_model, n_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)
