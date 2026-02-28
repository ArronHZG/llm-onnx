"""
注意力机制实现：自注意力和多头注意力
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    自注意力机制 (Self-Attention)

    公式: Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
    """

    def __init__(self, d_model: int, num_heads: int = 8):
        super(SelfAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度

        # Q, K, V 的线性变换
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # 输出线性变换
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入 tensor, shape (batch_size, seq_len, d_model)
            mask: 注意力掩码, shape (batch_size, seq_len, seq_len)

        Returns:
            输出 tensor, shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # 线性变换得到 Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 将 Q, K, V 分割成多个头
        # shape: (batch_size, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 计算注意力分数: Q * K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

        # 应用掩码（如果有）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)

        # 应用注意力到 V
        attn_output = torch.matmul(attn_weights, V)

        # 合并多个头的结果
        # shape: (batch_size, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # 输出线性变换
        output = self.W_o(attn_output)

        return output


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制的完整实现（包含残差连接和层归一化）
    """

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()

        self.attention = SelfAttention(d_model, num_heads)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 自注意力
        output = self.attention(x, mask)
        # 残差连接和层归一化
        output = self.norm(x + self.dropout(output))
        return output


if __name__ == "__main__":
    # 测试代码
    d_model = 512
    num_heads = 8
    batch_size = 2
    seq_len = 10

    model = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(batch_size, seq_len, d_model)

    with torch.no_grad():
        output = model(x)

    print(f"输入 shape: {x.shape}")
    print(f"输出 shape: {output.shape}")
    print("注意力机制测试通过！")

