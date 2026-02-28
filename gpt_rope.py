"""
GPT Transformer with Rotary Position Embedding (RoPE) 实现
"""
import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from gpt import FeedForward
from utils.onnx_utils import export_and_simplify, validate_onnx


def precompute_freqs_cis(dim: int, end: int, rope_base: float = 10000.0):
    """
    预计算RoPE的cos和sin值
    Args:
        dim: 每个头的维度（d_model // n_heads）
        end: 最大序列长度
        rope_base: RoPE的基数，默认为10000
    Returns:
        freqs_cos: [end, dim] 预计算的cos值
        freqs_sin: [end, dim] 预计算的sin值
    """
    inv_freq = 1.0 / (rope_base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end, device=inv_freq.device)
    freqs = torch.outer(t, inv_freq).float()
    # 将freqs复制一份，以便与交错后的维度匹配
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    # 交错重复以匹配原始维度（因为每个复数维度对应两个实数维度）
    freqs_cos = freqs_cos.repeat_interleave(2, dim=-1)  # shape: [end, dim]
    freqs_sin = freqs_sin.repeat_interleave(2, dim=-1)  # shape: [end, dim]
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, freqs_cos, freqs_sin):
    """
    将旋转位置编码应用于查询和键
    Args:
        q: [batch_size, n_heads, seq_len, head_dim]
        k: [batch_size, n_heads, seq_len, head_dim]
        freqs_cos: [seq_len, head_dim]
        freqs_sin: [seq_len, head_dim]
    Returns:
        q_rot: 旋转后的查询
        k_rot: 旋转后的键
    """
    # 将cos和sin扩展到与q、k相同的形状
    cos = freqs_cos.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, head_dim]
    sin = freqs_sin.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, head_dim]

    # 旋转公式: q_rot = q * cos + rotate_half(q) * sin
    # rotate_half: 将后半部分维度与前半部分交换并取反
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


class MultiHeadAttentionWithRoPE(nn.Module):
    """带RoPE的多头注意力"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, rope_base: float = 10000.0, max_seq_len=1024):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.rope_base = rope_base

        # 线性投影层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # 预计算的位置编码缓冲区
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=self.d_k,
            end=max_seq_len,
            rope_base=self.rope_base
        )

        self.register_buffer('freqs_cos', freqs_cos, persistent=False)
        self.register_buffer('freqs_sin', freqs_sin, persistent=False)

        # 注册因果掩码缓冲区（上三角矩阵，对角线以上为1，表示需要mask的位置）
        # mask形状: [seq_len, seq_len]
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1)
        self.register_buffer('causal_mask', mask, persistent=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, 1, seq_len, seq_len] (因果掩码)
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # 初始化RoPE位置编码

        # 线性投影并拆分多头: [batch_size, seq_len, n_heads, d_k]
        b, seq_len, _ = x.shape  # b = batch_size, n_tokens = seq_len

        # 线性投影并拆分多头: [batch_size, seq_len, n_heads, d_k]
        q = self.w_q(x).view(b, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(b, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(b, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 应用旋转位置编码
        q_rot, k_rot = apply_rotary_pos_emb(q, k,
                                           self.freqs_cos[:seq_len],
                                           self.freqs_sin[:seq_len])

        # 注意力计算: [batch_size, n_heads, seq_len, seq_len]
        attn_scores = torch.matmul(q_rot, k_rot.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 应用因果掩码（上三角矩阵）
        # 扩展mask维度以匹配多头注意力: [1, 1, seq_len, seq_len]
        mask_expanded = self.causal_mask[:seq_len, :seq_len].view(1, 1, seq_len, seq_len)
        attn_scores.masked_fill_(mask_expanded.bool(), -torch.inf)

        # 注意力权重和输出
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # [batch_size, n_heads, seq_len, d_k] -> [batch_size, seq_len, d_model]
        attn_output = torch.matmul(attn_weights, v).transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)

        # 最终投影
        output = self.w_o(attn_output)
        return output


class GPTTransformerBlockWithRoPE(nn.Module):
    """带RoPE的GPT Transformer块"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # Pre-LN结构（GPT的特点）
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttentionWithRoPE(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # 自注意力子层（残差连接）
        x_residual = x
        x = self.ln1(x)
        x = self.attn(x)  # 注意：不再传递mask参数
        x = self.dropout(x)
        x = x + x_residual

        # 前馈子层（残差连接）
        x_residual = x
        x = self.ln2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x + x_residual

        return x


class GPTTransformerWithRoPE(nn.Module):
    """完整的带RoPE的GPT Transformer模型"""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        d_ff: int = 3072,
        dropout: float = 0.1,
        rope_base: float = 10000.0,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 词嵌入dropout
        self.drop_emb = nn.Dropout(dropout)
        # Transformer层堆叠
        self.layers = nn.ModuleList([
            GPTTransformerBlockWithRoPE(d_model, n_heads, d_ff, dropout)
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
        self.rope_base = rope_base

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        完整GPT Transformer前向传播
        Args:
            input_ids: [batch_size, seq_len]
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        # 嵌入层
        x = self.embedding(input_ids)
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

    vocab_size = 1234  # 50257  # GPT-2词汇表大小
    batch_size = 2
    seq_len = 10
    max_seq_len = 1024

    model = GPTTransformerWithRoPE(
        vocab_size=vocab_size,
        d_model=768,
        n_heads=12,
        n_layers=1,  # 简化版
        d_ff=3072,
        max_seq_len=max_seq_len,
    )

    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(dummy_input)

    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {logits.shape}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 测试不同序列长度
    seq_len2 = 5
    dummy_input2 = torch.randint(0, vocab_size, (batch_size, seq_len2))
    logits2 = model(dummy_input2)
    print(f"不同序列长度输出形状: {logits2.shape}")

    print("GPT Transformer with RoPE测试通过！")

    # 导出为ONNX
    os.makedirs("onnx_data", exist_ok=True)
    onnx_path = "onnx_data/gpt_transformer_rope.onnx"

    # 导出并简化
    final_path = export_and_simplify(
        model=model,
        dummy_input=dummy_input,
        onnx_path=onnx_path,
        simplify=True,
        input_names=['input_ids'],
        output_names=['logits'],
        skipped_optimizers=['FuseMatMul'],
    )

    # 验证ONNX模型
    if validate_onnx(final_path):
        print("ONNX模型验证通过!")
    else:
        print("ONNX模型验证失败!")

    print(f"模型已导出到: {final_path}")
