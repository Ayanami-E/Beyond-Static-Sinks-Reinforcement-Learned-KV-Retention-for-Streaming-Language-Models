# kv_policy/policy_net.py
import torch
import torch.nn as nn


class GatedResidualBlock(nn.Module):
    """
    Residual block with LayerNorm and gated activation:
        x → LN → Linear(2 * expansion * d) → split into (v, g)
        output = GELU(v) * sigmoid(g) → Linear(d) → Dropout → + x
    Used to enhance model capacity while keeping computation stable.
    """
    def __init__(self, d: int, expansion: int = 2, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d * expansion * 2)
        self.fc2 = nn.Linear(d * expansion, d)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        h = self.fc1(h)
        v, g = h.chunk(2, dim=-1)
        h = self.act(v) * torch.sigmoid(g)
        h = self.fc2(h)
        h = self.dropout(h)
        return x + h


class RetentionPolicyMLP(nn.Module):
    """
    MLP policy network:
      - Input: feature vector (hidden state, position features, age, etc.)
      - Middle: LayerNorm + several gated residual blocks
      - Output: single logit representing retention likelihood
    """
    def __init__(
        self,
        d_in: int,
        d_hidden: int = 256,
        num_blocks: int = 3,
        dropout: float = 0.1,
        expansion: int = 2,
    ):
        super().__init__()
        self.fc_in = nn.Linear(d_in, d_hidden)
        self.ln_in = nn.LayerNorm(d_hidden)
        self.blocks = nn.ModuleList(
            [
                GatedResidualBlock(d_hidden, expansion=expansion, dropout=dropout)
                for _ in range(num_blocks)
            ]
        )
        self.ln_out = nn.LayerNorm(d_hidden)
        self.fc_out = nn.Linear(d_hidden, 1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc_in(x)
        h = self.act(self.ln_in(h))
        for block in self.blocks:
            h = block(h)
        h = self.ln_out(h)
        out = self.fc_out(h)
        return out.squeeze(-1)  # [N]
