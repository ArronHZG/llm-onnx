import torch.nn as nn
import torch.nn.functional as F
import torch

class SwiGLU(nn.Module):
    """SwiGLU激活函数 (SiLU/Swish + GLU)"""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, hidden_size]
        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        # SwiGLU: gate * silu(up)
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
