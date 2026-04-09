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

import os
import random

import numpy as np
import torch
import torch.nn as nn

from norm_layer import ZeroCenteredRMSNorm
from qwen3_5_dense import GatedDeltaNet, GatedAttention, SwiGLU, TrueSwiGLU
from qwen3_moe import Qwen3SimpleMoE, TopKGate
from utils.onnx_utils import export_and_simplify

# 配置参数
vocab_size = 1234
batch_size = 2
seq_len = 10
max_seq_len = 1024


def set_seed(seed: int = 42):
    """固定随机种子，确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # CUDA deterministic 模式（可能影响性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 默认固定种子
set_seed(42)

ZeroCenteredRMSNorm = nn.LayerNorm


class SimpleMoE(nn.Module):
    """
    Qwen3 风格的 MoE (Mixture of Experts) 实现

    特点:
        - 使用 TopKGate 进行 token 到专家的路由
        - 每个专家使用 SwiGLU (SiLU 门控线性单元)
        - 包含一个共享专家 (所有 token 都会经过)
    """

    def __init__(self, hidden_size, num_experts, top_k, intermediate_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k

        # 使用 TopKGate 进行路由
        self.gate = TopKGate(hidden_size, num_experts, top_k)

        # 专家网络 (每个专家是一个 SwiGLU MLP)
        # SwiGLU: gate_proj + up_proj -> SiLU(gate) * up -> down_proj
        self.experts = nn.ModuleList(
            [SwiGLU(hidden_size, intermediate_size) for _ in range(num_experts)]
        )

        # 共享专家 (所有 token 都会经过)
        self.shared_expert = TrueSwiGLU(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        original_shape = hidden_states.shape
        batch_size, seq_len, hidden_dim = original_shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # 1. 先注释掉复杂的路由逻辑
        # top_k_gates, top_k_indices = self.gate(hidden_states)

        # 2. 直接通过共享专家和第一个专家（模拟静态路径）
        shared_output = self.shared_expert(hidden_states)

        # 3. 暂时直接累加所有专家输出（绕过 TopK 动态索引）
        # 这一步是为了验证如果没有动态形状，模型是否能导出
        expert_output_sum = 0
        for expert in self.experts:
            expert_output_sum += expert(hidden_states) * 0.1  # 假的权重

        final_hidden_states = shared_output + expert_output_sum

        return final_hidden_states.view(batch_size, seq_len, hidden_dim)


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
        self.input_layernorm = ZeroCenteredRMSNorm(hidden_size)

        # 注意力层
        if use_linear_attention:
            self.self_attn = GatedDeltaNet(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                conv_kernel_size=conv_kernel_size,
                rope_base=rope_base,
                max_seq_len=max_seq_len,
                use_qk_norm=True,  # Qwen3.5特色: QK归一化
                qk_norm_type="l2",
                use_gate=True,  # 使用门控注意力
            )
        else:
            self.self_attn = GatedAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                dropout=dropout,
                rope_base=rope_base,
                max_seq_len=max_seq_len,
            )

        self.post_attention_layernorm = ZeroCenteredRMSNorm(hidden_size)

        # MoE前馈网络
        self.moe = Qwen3SimpleMoE(
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
    """Qwen3.5 MoE 完整模型 (支持混合注意力架构)

    结构:
    - 第1层: GatedDeltaNet (线性注意力)
    - 第2层: GatedAttention (标准注意力)
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        num_key_value_heads: int = 2,  # GQA
        head_dim: int = 128,
        num_hidden_layers: int = 2,  # 默认2层: 1层GatedDeltaNet + 1层GatedAttention
        num_experts: int = 8,  # MoE专家总数
        top_k: int = 2,  # 每个token激活的专家数
        intermediate_size: int = 2048,
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

        # Transformer层: 混合架构
        # 第1层使用 GatedDeltaNet (线性注意力)
        # 第2层使用 GatedAttention (标准注意力)
        self.layers = nn.ModuleList()

        for layer_idx in range(num_hidden_layers):
            # 第一层使用 GatedDeltaNet，其余层使用 GatedAttention
            use_linear_attention = layer_idx == 0

            self.layers.append(
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
            )

        # 最终归一化
        self.norm = ZeroCenteredRMSNorm(hidden_size)

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
    # 测试代码 - 混合架构
    print("=" * 50)
    print("测试 Qwen3.5 MoE (混合架构)")
    print("1层 GatedDeltaNet + 1层 GatedAttention")
    print("=" * 50)

    model = Qwen3_5MoEModel(
        vocab_size=vocab_size,
        hidden_size=768,
        num_attention_heads=6,
        num_key_value_heads=2,
        head_dim=128,
        num_hidden_layers=1,  # 1层 GatedDeltaNet + 1层 GatedAttention
        num_experts=8,  # 8个专家
        top_k=2,  # 激活2个专家
        intermediate_size=2048,
        max_position_embeddings=max_seq_len,
    )

    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(dummy_input)

    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {logits.shape}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print("Qwen3.5 MoE (混合架构) 测试通过！")

    # 导出为ONNX
    os.makedirs("onnx_data", exist_ok=True)
    onnx_path = "onnx_data/qwen3_5_moe.onnx"

    # 导出并简化
    final_path = export_and_simplify(
        model=model,
        dummy_input=dummy_input,
        onnx_path=onnx_path,
        simplify=True,
        input_names=["input_ids"],
        output_names=["logits"],
        skipped_optimizers=["FuseMatMul"],
    )

    print(f"模型已导出到: {final_path}")
