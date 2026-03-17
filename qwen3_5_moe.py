"""
Qwen3.5 MoE (Mixture of Experts) 模型实现
基于qwen3_moe.py简化而来，添加了Qwen3.5的GatedDeltaNet线性注意力

主要特点:
- RMSNorm: 均方根归一化
- RoPE: 旋转位置编码
- GQA (Grouped Query Attention): 分组查询注意力
- QK归一化: 对Query和Key进行归一化
- GatedDeltaNet: Qwen3.5特色的线性注意力机制
- MoE (Mixture of Experts): 专家混合模型
- SwiGLU: 激活函数 (SiLU/Swish + GLU)
- TopK专家路由: 每个token选择top-k个专家
- Pre-LN结构
"""
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from gpt_rope import precompute_freqs_cis, apply_rotary_pos_emb
from qwen3_5_dense import RMSNorm, SwiGLU
from utils.onnx_utils import export_and_simplify, validate_onnx

# 配置参数
vocab_size = 1234
batch_size = 2
seq_len = 10
max_seq_len = 1024


class Qwen3_5MoEDecoderLayer(nn.Module):
    """Qwen3.5 MoE Decoder层 (支持GatedDeltaNet线性注意力)"""

    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            num_kv_heads: int,
            head_dim: int = 128,
            num_experts: int = 8,
            top_k: int = 2,
            intermediate_size: int = 2048,
            use_linear_attention: bool = True,  # Qwen3.5特色
            dropout: float = 0.0,
            rope_base: float = 1000000.0,
            max_seq_len: int = 1024,
            conv_kernel_size: int = 4,
    ):
        super().__init__()

        # Pre-LN
        self.input_layernorm = RMSNorm(hidden_size)

        # 注意力层
        if use_linear_attention:
            self.self_attn = Qwen3_5GatedDeltaNet(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                conv_kernel_size=conv_kernel_size,
                rope_base=rope_base,
                max_seq_len=max_seq_len,
            )
        else:
            self.self_attn = Qwen3_5Attention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                dropout=dropout,
                rope_base=rope_base,
                max_seq_len=max_seq_len,
            )

        self.post_attention_layernorm = RMSNorm(hidden_size)

        # MoE前馈网络
        self.moe = Qwen3_5SimpleMoE(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            intermediate_size=intermediate_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """MoE Decoder forward"""
        # Self Attention with Pre-LN
        x_residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x)
        x = x + x_residual

        # MoE with Pre-LN
        x_residual = x
        x = self.post_attention_layernorm(x)
        x = self.moe(x)
        x = x + x_residual

        return x


class Qwen3_5MoEModel(nn.Module):
    """Qwen3.5 MoE 完整模型 (支持GatedDeltaNet)"""

    def __init__(
            self,
            vocab_size: int,
            hidden_size: int = 768,
            num_attention_heads: int = 12,
            num_key_value_heads: int = 2,  # GQA
            head_dim: int = 128,
            num_hidden_layers: int = 1,
            num_experts: int = 8,  # MoE专家总数
            top_k: int = 2,  # 每个token激活的专家数
            intermediate_size: int = 2048,
            use_linear_attention: bool = True,  # Qwen3.5特色
            dropout: float = 0.0,
            rope_base: float = 1000000.0,
            max_position_embeddings: int = 1024,
            conv_kernel_size: int = 4,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

        # 词嵌入
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        # MoE Transformer层
        self.layers = nn.ModuleList([
            Qwen3_5MoEDecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_attention_heads,
                num_kv_heads=num_key_value_heads,
                head_dim=head_dim,
                num_experts=num_experts,
                top_k=top_k,
                intermediate_size=intermediate_size,
                use_linear_attention=use_linear_attention,
                dropout=dropout,
                rope_base=rope_base,
                max_seq_len=max_position_embeddings,
                conv_kernel_size=conv_kernel_size,
            )
            for _ in range(num_hidden_layers)
        ])

        # 最终归一化
        self.norm = RMSNorm(hidden_size)

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
    # 测试代码 - GatedDeltaNet版本
    print("=" * 50)
    print("测试 Qwen3.5 MoE (GatedDeltaNet)")
    print("=" * 50)

    model = Qwen3_5MoEModel(
        vocab_size=vocab_size,
        hidden_size=768,
        num_attention_heads=6,
        num_key_value_heads=2,
        head_dim=128,
        num_hidden_layers=1,
        num_experts=8,  # 8个专家
        top_k=2,  # 激活2个专家
        intermediate_size=2048,
        use_linear_attention=True,  # GatedDeltaNet
        max_position_embeddings=max_seq_len,
    )

    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(dummy_input)

    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {logits.shape}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print("Qwen3.5 MoE (GatedDeltaNet) 测试通过！")

    # 测试标准注意力版本
    print("\n" + "=" * 50)
    print("测试 Qwen3.5 MoE (标准Attention)")
    print("=" * 50)

    model_standard = Qwen3_5MoEModel(
        vocab_size=vocab_size,
        hidden_size=768,
        num_attention_heads=6,
        num_key_value_heads=2,
        head_dim=128,
        num_hidden_layers=1,
        num_experts=8,
        top_k=2,
        intermediate_size=2048,
        use_linear_attention=False,  # 标准Attention
        max_position_embeddings=max_seq_len,
    )

    logits_standard = model_standard(dummy_input)

    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {logits_standard.shape}")
    print(f"模型参数量: {sum(p.numel() for p in model_standard.parameters()):,}")
    print("Qwen3.5 MoE (标准Attention) 测试通过！")

    # 导出为ONNX
    os.makedirs("onnx_data", exist_ok=True)
    onnx_path = "onnx_data/qwen3_5_moe.onnx"

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
        print("\nONNX模型验证通过!")
    else:
        print("\nONNX模型验证失败!")

    print(f"模型已导出到: {final_path}")
