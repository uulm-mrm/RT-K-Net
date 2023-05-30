from torch import nn

__all__ = ["FFN"]


class FFN(nn.Module):
    """Implementation of Feedforward model with skip connection"""

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt
