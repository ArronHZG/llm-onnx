"""
Qwen3 Dense 模型实现
基于sglang的qwen3.py简化而来

主要特点:
- RMSNorm (而非LayerNorm)
- RoPE旋转位置编码
- GQA (Grouped Query Attention)
- SwiGLU激活函数
- QK归一化
"""
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

from gpt_activate import SwiGLU
from gpt_rope import precompute_freqs_cis, apply_rotary_pos_emb
from utils.onnx_utils import export_and_simplify, validate_onnx

# 配置参数
vocab_size = 1234
batch_size = 2
seq_len = 10
max_seq_len = 1024


class Qwen3Attention(nn.Module):
    """Qwen3多头注意力 (支持GQA)"""

    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            num_kv_heads: int,
            dropout: float = 0.0,
            rope_base: float = 1000000.0,
            max_seq_len: int = 1024,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        assert num_kv_heads <= num_heads

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.rope_base = rope_base

        # QKV投影 (合并为一个线性层，模拟QKVParallelLinear)
        self.qkv_proj = nn.Linear(hidden_size, (num_heads + 2 * num_kv_heads) * self.head_dim, bias=False)

        # O投影
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

        self.dropout = nn.Dropout(dropout)

        # QK归一化 (Qwen3特点)
        self.q_norm = LayerNorm(self.head_dim)
        self.k_norm = LayerNorm(self.head_dim)

        # RoPE
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=self.head_dim,
            end=max_seq_len,
            rope_base=rope_base
        )
        self.register_buffer('freqs_cos', freqs_cos, persistent=False)
        self.register_buffer('freqs_sin', freqs_sin, persistent=False)

        # 因果掩码
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1)
        self.register_buffer('causal_mask', mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, hidden_size]
        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = x.shape
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim

        # QKV投影并分割
        qkv = self.qkv_proj(x)
        q, kv = qkv.split([q_size, 2 * kv_size], dim=-1)

        # 分割K和V
        k, v = kv.split([kv_size, kv_size], dim=-1)

        # Reshape: [batch_size, seq_len, num_heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # QK归一化
        q = self.q_norm(q)
        k = self.k_norm(k)

        # 应用RoPE
        q, k = apply_rotary_pos_emb(
            q, k,
            self.freqs_cos[:seq_len],
            self.freqs_sin[:seq_len]
        )

        # 如果KV头数少于Q头，需要复制K和V以匹配Q头数
        if self.num_kv_heads < self.num_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        # 注意力计算
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 因果掩码
        mask_expanded = self.causal_mask[:seq_len, :seq_len].view(1, 1, seq_len, seq_len)
        attn_scores.masked_fill_(mask_expanded.bool(), -torch.inf)

        # Softmax和dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 注意力输出
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.head_dim)

        # O投影
        output = self.o_proj(attn_output)
        return output


class Qwen3DecoderLayer(nn.Module):
    """Qwen3 Decoder层"""

    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            num_kv_heads: int,
            intermediate_size: int,
            dropout: float = 0.0,
            rope_base: float = 1000000.0,
            max_seq_len: int = 1024,
    ):
        super().__init__()
        # Pre-LN结构
        self.input_layernorm = LayerNorm(hidden_size)
        self.self_attn = Qwen3Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
            rope_base=rope_base,
            max_seq_len=max_seq_len,
        )
        self.post_attention_layernorm = LayerNorm(hidden_size)
        self.mlp = SwiGLU(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, hidden_size]
        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        # Self Attention with Pre-LN
        x_residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x)
        x = x + x_residual

        # MLP with Pre-LN
        x_residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = x + x_residual

        return x


class Qwen3DenseModel(nn.Module):
    """Qwen3 Dense模型"""

    def __init__(
            self,
            vocab_size: int,
            hidden_size: int = 768,
            num_attention_heads: int = 12,
            num_key_value_heads: int = 2,  # GQA: KV头数可以少于Q头数
            num_hidden_layers: int = 1,
            intermediate_size: int = 2048,
            dropout: float = 0.0,
            rope_base: float = 1000000.0,
            max_position_embeddings: int = 1024,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

        # 词嵌入
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        # Transformer层
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_attention_heads,
                num_kv_heads=num_key_value_heads,
                intermediate_size=intermediate_size,
                dropout=dropout,
                rope_base=rope_base,
                max_seq_len=max_position_embeddings,
            )
            for _ in range(num_hidden_layers)
        ])

        # 最终归一化
        self.norm = torch.nn.LayerNorm(hidden_size)

        # LM Head
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len]
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        # 词嵌入
        hidden_states = self.embed_tokens(input_ids)

        # Transformer层
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        # 最终归一化
        hidden_states = self.norm(hidden_states)

        # LM Head
        logits = self.lm_head(hidden_states)

        return logits


if __name__ == "__main__":
    # 测试代码
    model = Qwen3DenseModel(
        vocab_size=vocab_size,
        hidden_size=768,
        num_attention_heads=12,
        num_key_value_heads=2,  # GQA
        num_hidden_layers=1,
        intermediate_size=2048,
        max_position_embeddings=max_seq_len,
    )

    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(dummy_input)

    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {logits.shape}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print("Qwen3 Dense模型测试通过！")

    # 导出为ONNX
    os.makedirs("onnx_data", exist_ok=True)
    onnx_path = "onnx_data/qwen3_dense.onnx"

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
