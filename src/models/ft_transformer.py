import torch
import torch.nn as nn
from rtdl_revisiting_models import FTTransformer as RTDL_FTTransformer


class FTTransformer(nn.Module):

    def __init__(self, in_features: int, n_classes: int = 3, d_block: int = 192,
                 n_blocks: int = 3, attention_n_heads: int = 8, attention_dropout: float = 0.2,
                 ffn_d_hidden_multiplier: float = 4/3, ffn_dropout: float = 0.1,
                 residual_dropout: float = 0.0):
        super().__init__()
        self.model = RTDL_FTTransformer(
            n_cont_features=in_features,
            cat_cardinalities=[],
            d_out=n_classes,
            n_blocks=n_blocks,
            d_block=d_block,
            attention_n_heads=attention_n_heads,
            attention_dropout=attention_dropout,
            ffn_d_hidden_multiplier=ffn_d_hidden_multiplier,
            ffn_dropout=ffn_dropout,
            residual_dropout=residual_dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x_cont=x, x_cat=None)
