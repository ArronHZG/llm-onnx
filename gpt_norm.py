import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """RMSNorm (Root Mean Square Layer Normalization)"""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, hidden_size]
        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        # 计算RMS
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # 归一化并乘以权重
        return x * rms * self.weight


class ZeroCenteredRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 计算RMS
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # 归一化并乘以权重
        return x * rms * (1.0 + self.weight)


class GatedRMSNorm(nn.Module):
    """GatedRMSNorm - 支持门控归一化的RMSNorm

    支持两种调用方式:
        - forward(x): 标准RMSNorm
        - forward(x, z): 门控RMSNorm, 输出 = norm(x) * sigmoid(z)

    公式:
        - 标准: y = x / sqrt(E[x^2] + epsilon) * weight
        - 门控: y = x / sqrt(E[x^2] + epsilon) * weight * sigmoid(z)
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor, z: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: 输入张量 [..., hidden_size]
            z: 可选的门控张量 [..., hidden_size]
        Returns:
            output: 归一化后的张量 [..., hidden_size]
        """
        # 计算RMS
        input_shape = x.shape
        x = x.reshape(-1, x.shape[-1])
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # 归一化并乘以权重
        x = x * rms * self.weight

        # 如果提供了门控z，应用门控
        if z is not None:
            z = z.reshape(-1, z.shape[-1])
            x = x * torch.sigmoid(z)

        return x.reshape(input_shape)



