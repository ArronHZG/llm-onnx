"""
GPT Transformer模型实现（带绝对位置嵌入）
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GPTMultiHeadAttention(nn.Module):
    """GPT风格的多头注意力（兼容ONNX）"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # 线性投影层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, 1, seq_len, seq_len] (因果掩码)
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # 线性投影并拆分多头: [batch_size, seq_len, n_heads, d_k]
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 注意力计算: [batch_size, n_heads, seq_len, seq_len]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 应用掩码（兼容ONNX）
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # 注意力权重和输出
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # [batch_size, n_heads, seq_len, d_k] -> [batch_size, seq_len, d_model]
        attn_output = torch.matmul(attn_weights, v).transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)

        # 最终投影
        output = self.w_o(attn_output)
        return output


class FeedForward(nn.Module):
    """GPT前馈网络（使用ReLU激活，实际GPT使用GELU）"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activate = nn.ReLU()  # 注意：GPT使用GELU激活

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        x = self.linear1(x)
        x = self.activate(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class GPTTransformerBlock(nn.Module):
    """GPT Transformer块（Pre-LN结构）"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # Pre-LN结构（GPT的特点）
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = GPTMultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        创建因果掩码（兼容ONNX）
        Args:
            seq_len: 序列长度
            device: 设备
        Returns:
            mask: [1, 1, seq_len, seq_len]
        """
        # 使用triu构造上三角掩码，避免动态条件判断
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        # 转换为0/1掩码（ONNX更兼容）
        mask = ~mask  # True表示可见，False表示不可见
        mask = mask.unsqueeze(0).unsqueeze(0)  # 扩展维度
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        # 创建因果掩码
        causal_mask = self._create_causal_mask(seq_len, x.device)

        # 自注意力子层（残差连接）
        x_residual = x
        x = self.ln1(x)
        x = self.attn(x, causal_mask)
        x = self.dropout(x)
        x = x + x_residual

        # 前馈子层（残差连接）
        x_residual = x
        x = self.ln2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x + x_residual

        return x


class GPTTransformer(nn.Module):
    """完整的GPT Transformer模型"""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        d_ff: int = 3072,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 位置嵌入（GPT使用绝对位置嵌入）
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        # 词嵌入dropout
        self.drop_emb = nn.Dropout(dropout)
        # Transformer层堆叠
        self.layers = nn.ModuleList([
            GPTTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        # 最终层归一化
        self.ln_f = nn.LayerNorm(d_model)
        # 输出层
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # 保存配置参数
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        完整GPT Transformer前向传播
        Args:
            input_ids: [batch_size, seq_len]
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.size()

        # 位置编码
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)

        # 嵌入层
        x = self.embedding(input_ids) + self.pos_embedding(position_ids)
        x = self.drop_emb(x)

        # 遍历Transformer层
        for layer in self.layers:
            x = layer(x)

        # 最终层归一化和输出
        x = self.ln_f(x)
        logits = self.head(x)

        return logits


if __name__ == "__main__":
    # 测试代码
    vocab_size = 50257  # GPT-2词汇表大小
    batch_size = 2
    seq_len = 10

    model = GPTTransformer(
        vocab_size=vocab_size,
        d_model=768,
        n_heads=12,
        n_layers=1,  # 简化版
        d_ff=3072,
        max_seq_len=1024,
    )

    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(dummy_input)

    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {logits.shape}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print("GPT Transformer测试通过！")

