"""
Qwen3 MoE (Mixture of Experts) 模型实现
基于sglang的qwen3_moe.py简化而来

主要特点:
- RMSNorm: 均方根归一化
- RoPE: 旋转位置编码
- GQA (Grouped Query Attention): 分组查询注意力
- QK归一化: 对Query和Key进行归一化
- MoE (Mixture of Experts): 专家混合模型
- SwiGLU: 激活函数 (SiLU/Swish + GLU)
- TopK专家路由: 每个token选择top-k个专家
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from gpt import FeedForward
from qwen3_dense import Qwen3Attention, SwiGLU
from utils.onnx_utils import export_and_simplify, validate_onnx

# 配置参数
vocab_size = 1234
batch_size = 2
seq_len = 10
max_seq_len = 1024

# 简化模型结构
TrueSwiGLU = SwiGLU
SwiGLU = FeedForward


class TopKGate(nn.Module):
    """
    Top-K 门控机制 (Top-K Gating)

    作用: 决定每个token应该被路由到哪些专家
    工作原理:
        1. 通过一个线性层计算每个token对每个专家的得分(logits)
        2. 使用softmax将得分转换为概率分布
        3. 选择概率最高的top-k个专家
        4. 对top-k个专家的概率进行归一化

    优点:
        - 稀疏激活: 只有少数专家被激活，减少计算量
        - 可扩展性: 可以轻松增加专家数量而不显著增加推理成本
    """

    def __init__(self, hidden_size: int, num_experts: int, top_k: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts  # 专家总数
        self.top_k = top_k  # 每个token选择的专家数量

        # 门控权重: [hidden_size, num_experts]
        # 输入: hidden_size维的向量
        # 输出: num_experts维的logits
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: [batch_size * seq_len, hidden_size] - 展平后的隐藏状态
        Returns:
            topk_gates: [batch_size * seq_len, top_k] - 归一化后的专家权重
            topk_indices: [batch_size * seq_len, top_k] - 选中的专家索引
        """
        # 计算路由logits: [num_tokens, num_experts]
        # 每个token对每个专家的得分
        logits = self.gate(x)

        # 获取 top-k 专家
        # topk_gates: 每个token对其top-k专家的得分
        # topk_indices: 每个token的top-k专家的索引
        topk_gates, topk_indices = torch.topk(logits, self.top_k, dim=-1)

        # 对top-k专家的得分进行softmax归一化
        # 使得所有被选中专家的权重之和为1
        topk_gates = F.softmax(topk_gates, dim=-1)

        return topk_gates, topk_indices


class Qwen3SimpleMoE(nn.Module):
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
        self.experts = nn.ModuleList([
            SwiGLU(hidden_size, intermediate_size)
            for _ in range(num_experts)
        ])

        # 共享专家 (所有 token 都会经过)
        self.shared_expert = TrueSwiGLU(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_dim)
        Returns:
            output: (batch_size, seq_len, hidden_dim)
        """
        # 保存原始形状，支持3D或2D输入
        original_shape = hidden_states.shape
        batch_size, seq_len, hidden_dim = original_shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        num_tokens = hidden_states.shape[0]

        # ===== 1. 路由计算 (使用 TopKGate) =====
        top_k_gates, top_k_indices = self.gate(hidden_states)

        # ===== 2. 共享专家计算 =====
        shared_output = self.shared_expert(hidden_states)

        # ===== 3. 专家计算 (按专家分组批量处理) =====
        # 展平: 每个token的每个top-k选择变成一个条目
        # top_k_indices: (num_tokens, top_k) -> flat_idx: (num_tokens * top_k,)
        flat_idx = top_k_indices.view(-1)
        # top_k_gates: (num_tokens, top_k) -> flat_weights: (num_tokens * top_k,)
        flat_weights = top_k_gates.view(-1)

        # 初始化输出
        final_hidden_states = torch.zeros_like(hidden_states)

        # 按专家索引排序，将同一专家的token连续存放
        sorted_idx = flat_idx.argsort()

        # 统计每个专家的token数量（bincount会自动为未激活的专家返回0）
        expert_counts = flat_idx.bincount(minlength=self.num_experts)

        # 计算每个专家的起始位置（前缀和）
        expert_offsets = torch.zeros(self.num_experts, dtype=torch.long, device=flat_idx.device)
        expert_offsets[1:] = expert_counts[:-1].cumsum(0)

        # 按专家分组批量处理
        for expert_id in range(self.num_experts):
            start = expert_offsets[expert_id].item()
            count = expert_counts[expert_id].item()

            if count == 0:
                continue

            # 获取该专家负责的token在排序后的位置
            sorted_positions = sorted_idx[start:start + count]
            # 获取对应的原始token索引
            token_indices = sorted_positions // self.top_k

            # 获取对应的权重
            weights = flat_weights[sorted_positions]

            # 批量计算
            expert_tokens = hidden_states[token_indices]
            expert_output = self.experts[expert_id](expert_tokens)

            # 加权累加到对应位置
            weighted_output = expert_output * weights.unsqueeze(-1)
            final_hidden_states.index_add_(0, token_indices, weighted_output)

        # ===== 4. 合并共享专家输出 =====
        final_hidden_states = final_hidden_states + shared_output

        # 恢复原始形状
        return final_hidden_states.view(batch_size, seq_len, hidden_dim)


class VectorizedMoE(nn.Module):
    """高效的向量化 MoE 实现"""

    def __init__(self, hidden_size, num_experts, num_experts_per_tok, intermediate_size):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = num_experts_per_tok

        # 所有专家打包成一个大模块
        # w1: gate_proj, w3: up_proj -> 合并为 w13 (gate + up)
        # w2: down_proj
        self.w13 = nn.Linear(hidden_size, intermediate_size * 2, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)

        # 路由网络
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

        # 共享专家
        self.shared_expert = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size, bias=False),
            nn.SiLU(),
            nn.Linear(intermediate_size, hidden_size, bias=False),
        )
        self.shared_expert_gate = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_dim) 或 (num_tokens, hidden_dim)
        """
        # 保存原始形状
        original_shape = hidden_states.shape
        if len(original_shape) == 3:
            batch_size, seq_len, hidden_dim = original_shape
            num_tokens = batch_size * seq_len
            hidden_states = hidden_states.view(-1, hidden_dim)
        else:
            num_tokens, hidden_dim = hidden_states.shape

        # ==================== 1. 路由计算 ====================
        # router_logits: (num_tokens, num_experts)
        router_logits = self.gate(hidden_states)

        # Top-k 选择 + softmax 归一化
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        # ==================== 2. 专家分发 (核心向量化) ====================
        # 将 token 按专家分组，同一专家的 token 一起计算
        # 方法: 使用 scatter_reduce 或 gather + bmm

        # flat_expert_indices: (num_tokens * top_k,) 展平后的专家索引
        flat_expert_indices = top_k_indices.view(-1)  # (num_tokens * top_k,)

        # flat_tokens: (num_tokens * top_k, hidden_dim) - 每个 token-topk 对的输入
        # 通过 expand + view 实现 token 复制
        flat_tokens = hidden_states.unsqueeze(1).expand(-1, self.top_k, -1).contiguous()
        flat_tokens = flat_tokens.view(-1, hidden_dim)  # (num_tokens * top_k, hidden_dim)

        # flat_weights: (num_tokens * top_k,) - 对应权重
        flat_weights = top_k_weights.view(-1)  # (num_tokens * top_k,)

        # ==================== 3. 专家计算 (分组批量处理) ====================
        # 计算 gate + up
        flat_gate_up = self.w13(flat_tokens)  # (num_tokens * top_k, intermediate_size * 2)
        gate_up = flat_gate_up.chunk(2, dim=-1)  # 拆分成 gate 和 up

        # SiLU 激活
        activated = F.silu(gate_up[0]) * gate_up[1]  # (num_tokens * top_k, intermediate_size)

        # 计算 down_proj (专家输出)
        expert_outputs = self.w2(activated)  # (num_tokens * top_k, hidden_dim)

        # ==================== 4. 加权合并 ====================
        # 用 weights * output 的方式，需要先按 token 重新排列
        # 每个 token 有 top_k 个专家的输出，需要加权求和

        # 方法: 创建输出 tensor，然后用 scatter_add
        final_hidden_states = torch.zeros_like(hidden_states)

        # 把结果加回到对应位置
        # top_k_indices: (num_tokens, top_k)
        # flat_weights: (num_tokens * top_k,)
        # expert_outputs: (num_tokens * top_k, hidden_dim)

        # 遍历每个 top_k 位置 (top_k 通常很小，如 2-8)
        for k in range(self.top_k):
            expert_idx = top_k_indices[:, k]  # (num_tokens,)
            weight = top_k_weights[:, k]  # (num_tokens,)
            output = expert_outputs.view(num_tokens, self.top_k, -1)[:, k, :]  # (num_tokens, hidden_dim)

            # 按专家索引分组累加
            # 真正的生产代码会用更复杂的分组逻辑或自定义 CUDA kernel
            for expert_id in range(self.num_experts):
                mask = (expert_idx == expert_id)  # (num_tokens,)
                if mask.any():
                    final_hidden_states[mask] += output[mask] * weight[mask].unsqueeze(-1)

        # ==================== 5. 共享专家 ====================
        shared_output = self.shared_expert(hidden_states)
        shared_gate = torch.sigmoid(self.shared_expert_gate(hidden_states))
        shared_output = shared_gate * shared_output
        final_hidden_states += shared_output

        # 恢复原始形状
        if len(original_shape) == 3:
            final_hidden_states = final_hidden_states.view(batch_size, seq_len, hidden_dim)

        return final_hidden_states


# ==================== 更极致的优化版本 (概念) ====================
#
# 实际生产代码会这样做:
#
# 1. Token 分组: 把选中同一专家的 token 聚在一起
#    tokens_for_expert_0 = [token_3, token_7, token_15, ...]  # 都选专家0
#    tokens_for_expert_1 = [token_1, token_8, token_22, ...]  # 都选专家1
#    ...
#
# 2. 批量矩阵乘法: 同一专家处理多个 token
#    expert_0(torch.cat(tokens_for_expert_0, dim=0))  # 一次计算多个 token
#
# 3. 结果重排: 把专家输出放回原 token 位置
#
# 4. CUDA/Triton Kernel: 高度优化的自定义 kernel
#    - 内存访问模式优化
#    - 共享内存利用
#    - 异步执行

class Qwen3MoEDecoderLayer(nn.Module):
    """
    Qwen3 MoE Decoder层

    结构 (Pre-LN):
        1. input_layernorm: 输入层归一化
        2. self_attn: 多头自注意力
        3. post_attention_layernorm: 注意力后归一化
        4. moe: MoE前馈网络

    Pre-LN vs Post-LN:
        - Pre-LN: LayerNorm在残差连接之前
        - Post-LN: LayerNorm在残差连接之后
        - Qwen3使用Pre-LN，训练更稳定
    """

    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            num_kv_heads: int,
            num_experts: int = 8,
            top_k: int = 2,
            intermediate_size: int = 2048,
            dropout: float = 0.0,
            rope_base: float = 1000000.0,
            max_seq_len: int = 1024,
    ):
        super().__init__()

        # Pre-LN: 归一化在前面
        self.input_layernorm = nn.LayerNorm(hidden_size)

        # 自注意力层
        self.self_attn = Qwen3Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
            rope_base=rope_base,
            max_seq_len=max_seq_len,
        )

        # Pre-LN: 注意力后归一化
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)

        # MoE前馈网络 (替代标准FFN)
        self.moe = Qwen3SimpleMoE(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            intermediate_size=intermediate_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, hidden_size]
        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        # ===== Self Attention with Pre-LN =====
        # Pre-LN: 先归一化，再计算注意力，最后残差连接
        x_residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x)
        x = x + x_residual  # 残差连接

        # ===== MoE FFN with Pre-LN =====
        x_residual = x
        x = self.post_attention_layernorm(x)
        x = self.moe(x)
        x = x + x_residual  # 残差连接

        return x


class Qwen3MoEModel(nn.Module):
    """
    Qwen3 MoE 完整模型

    结构:
        1. 词嵌入层 (embed_tokens): token_id -> hidden_state
        2. N层Transformer Decoder (layers)
        3. 最终归一化 (norm)
        4. LM Head (lm_head): hidden_state -> logits

    与Dense模型的区别:
        - FFN层被替换为MoE层
        - 每个token只激活部分专家计算
        - 大幅增加模型容量同时保持推理成本可控
    """

    def __init__(
            self,
            vocab_size: int,
            hidden_size: int = 768,
            num_attention_heads: int = 12,
            num_key_value_heads: int = 2,  # GQA: KV头数少于Q头数
            num_hidden_layers: int = 1,
            num_experts: int = 8,  # MoE专家总数
            top_k: int = 2,  # 每个token激活的专家数
            intermediate_size: int = 2048,
            dropout: float = 0.0,
            rope_base: float = 1000000.0,
            max_position_embeddings: int = 1024,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

        # 词嵌入: [vocab_size, hidden_size]
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        # Transformer层堆叠
        self.layers = nn.ModuleList([
            Qwen3MoEDecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_attention_heads,
                num_kv_heads=num_key_value_heads,
                num_experts=num_experts,
                top_k=top_k,
                intermediate_size=intermediate_size,
                dropout=dropout,
                rope_base=rope_base,
                max_seq_len=max_position_embeddings,
            )
            for _ in range(num_hidden_layers)
        ])

        # 最终归一化层
        self.norm = nn.LayerNorm(hidden_size)

        # LM Head: 预测下一个token的logits
        # 权重通常与embed_tokens共享 (tie_word_embeddings)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len] - token IDs
        Returns:
            logits: [batch_size, seq_len, vocab_size] - 每个位置的预测logits
        """
        # 1. 词嵌入
        hidden_states = self.embed_tokens(input_ids)

        # 2. 依次通过所有Transformer层
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        # 3. 最终归一化
        hidden_states = self.norm(hidden_states)

        # 4. LM Head得到logits
        logits = self.lm_head(hidden_states)

        return logits


if __name__ == "__main__":
    # 创建模型实例
    model = Qwen3MoEModel(
        vocab_size=vocab_size,
        hidden_size=768,
        num_attention_heads=12,
        num_key_value_heads=2,  # GQA: 12个Q头，2个KV头
        num_hidden_layers=1,
        num_experts=8,  # 8个专家
        top_k=2,  # 激活2个专家
        intermediate_size=2048,
        max_position_embeddings=max_seq_len,
    )

    # 创建随机输入
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))

    # 前向传播测试
    logits = model(dummy_input)

    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {logits.shape}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print("Qwen3 MoE模型测试通过！")

    # 导出为ONNX
    os.makedirs("onnx_data", exist_ok=True)
    onnx_path = "onnx_data/qwen3_moe.onnx"

    # 导出并简化ONNX模型
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
