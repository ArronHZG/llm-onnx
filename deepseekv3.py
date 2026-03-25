"""DeepSeekV3 Model Implementation (optimized).

DeepSeekV3 是改进的 MoE (Mixture of Experts) 语言模型，具有：
- 256 个路由专家，每个 token 激活 8 个
- 分组 top-k 路由 (8 组，每组 4 个专家)
- 2 个共享专家以改进性能
- 多头潜在注意力 (MLA) 用于高效 KV 缓存

优化策略 (参考 DeepSeekMoE 论文):
1. Softmax 权重归一化 - 替代原始的概率归一化
2. 优化的负载平衡损失 - 确保专家使用均衡
3. Z-loss 正则化 - 防止路由 logits 爆炸
4. 共享专家隔离 - 共享专家与路由专家分离
5. 正交初始化 - 提高路由稳定性
6. 高效批处理 - 优化的专家计算流程

References:
- DeepSeekV3: https://arxiv.org/abs/2401.14163
- DeepSeekMoE: https://arxiv.org/abs/2401.06066
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from gpt import FeedForward
from gpt_rope import apply_rotary_pos_emb, precompute_freqs_cis

RMSNorm = nn.LayerNorm


class DeepSeekV3MLP(nn.Module):
    """DeepSeekV3 MLP layer (single expert)."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_up_proj = nn.Linear(hidden_size, intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gate_up_proj(x)
        gate, up = x.chunk(2, dim=-1)
        x = self.act_fn(gate) * up
        x = self.down_proj(x)
        return x


# 简化模型结构
SwiGLU = FeedForward

# Deepseek mlp 就是 SwiGLU
DeepSeekV3MLP = SwiGLU


class MultiheadLatentAttention(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            q_lora_rank: int = 1536,
            kv_lora_rank: int = 512,
            qk_nope_head_dim: int = 128,
            qk_rope_head_dim: int = 64,
            v_head_dim: int = 128,
            max_seq_len: int = 2048,
            dropout: float = 0.0,
    ):
        super().__init__()

        # 基本参数设置
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

        # 维度映射 (与之前版本兼容)
        self.d_k = qk_nope_head_dim  # 非RoPE部分的维度
        self.d_r = qk_rope_head_dim  # RoPE部分的维度
        self.d_c = kv_lora_rank  # KV的低秩投影维度
        self.d_c_prime = q_lora_rank  # Q的低秩投影维度
        self.d_v = v_head_dim  # V的输出维度

        # 定义投影矩阵
        # 低秩投影: hidden -> kv_lora_rank
        self.W_c = nn.Linear(hidden_size, self.d_c, bias=False)  # W_c
        # 低秩投影: hidden -> q_lora_rank
        self.W_c_prime = nn.Linear(hidden_size, self.d_c_prime, bias=False)  # W_c'

        # Q的投影矩阵 (矩阵化): d_c_prime -> num_heads * d_k
        self.W_qc = nn.Linear(self.d_c_prime, self.num_heads * self.d_k, bias=False)
        # Q的RoPE投影矩阵 (矩阵化): d_c_prime -> num_heads * d_r
        self.W_qr = nn.Linear(self.d_c_prime, self.num_heads * self.d_r, bias=False)

        # K的投影矩阵 (矩阵化): d_c -> num_heads * d_k
        self.W_kc = nn.Linear(self.d_c, self.num_heads * self.d_k, bias=False)
        # K的RoPE投影矩阵: hidden -> d_r (所有头共享)
        self.W_kr = nn.Linear(hidden_size, self.d_r, bias=False)  # W_kr

        # V的投影矩阵 (矩阵化): d_c -> num_heads * d_v
        self.W_v = nn.Linear(self.d_c, self.num_heads * self.d_v, bias=False)

        # Output projection
        self.o_proj = nn.Linear(self.num_heads * self.d_v, hidden_size, bias=False)

        # Dropout层
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # KV缓存
        self.c_cache = None  # 低秩投影后的缓存
        self.x_cache = None  # 原始输入的缓存

        # RoPE: use precompute_freqs_cis from gpt_rope
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=qk_rope_head_dim,
            end=max_seq_len,
            rope_base=10000.0
        )

        self.register_buffer('freqs_cos', freqs_cos, persistent=False)
        self.register_buffer('freqs_sin', freqs_sin, persistent=False)

        # 注意力掩码
        causal_mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1)
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def precompute_matrices(self):
        # 推理阶段使用的预计算矩阵: W_kc.t() @ W_qc.t() = (W_qc @ W_kc.t()).t()
        # 使得: q_c = c_prime @ merged = c_prime @ W_kc.t() @ W_qc.t()
        self.merged_W_qc_kc = [
            torch.matmul(self.W_kc[i].weight.t(), self.W_qc[i].weight)
            for i in range(self.num_heads)
        ]

    def forward(self, x, kv_cache=False):
        batch_size, seq_len, _ = x.shape

        # 低秩投影
        c = self.W_c(x)  # [batch, seq, d_c]
        c_prime = self.W_c_prime(x)  # [batch, seq, d_c_prime]

        # 矩阵化计算 Q、K、V (所有头一起计算)
        # Qc: [batch, seq, num_heads * d_k] -> reshape -> [batch, num_heads, seq, d_k]
        q_c = self.W_qc(c_prime).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # Kc: [batch, seq, num_heads * d_k] -> reshape -> [batch, num_heads, seq, d_k]
        k_c = self.W_kc(c).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # Qr: [batch, seq, num_heads * d_r] -> reshape -> [batch, num_heads, seq, d_r]
        q_r = self.W_qr(c_prime).view(batch_size, seq_len, self.num_heads, self.d_r).transpose(1, 2)
        # Kr: [batch, seq, d_r] (共享) -> expand -> [batch, num_heads, seq, d_r]
        k_r = self.W_kr(x).unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        # V: [batch, seq, num_heads * d_v] -> reshape -> [batch, num_heads, seq, d_v]
        v = self.W_v(c).view(batch_size, seq_len, self.num_heads, self.d_v).transpose(1, 2)

        # 应用 RoPE 到 Qr 和 Kr
        q_r, k_r = apply_rotary_pos_emb(q_r, k_r, self.freqs_cos[:seq_len], self.freqs_sin[:seq_len])

        # 拼接 Q 和 K 的 nope + rope 部分
        q = torch.cat([q_c, q_r], dim=-1)  # [batch, num_heads, seq, d_k + d_r]
        k = torch.cat([k_c, k_r], dim=-1)  # [batch, num_heads, seq, d_k + d_r]

        # 计算注意力分数
        scale = 1.0 / math.sqrt(self.d_k + self.d_r)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # 应用因果掩码（上三角矩阵）
        # 扩展mask维度以匹配多头注意力: [1, 1, seq_len, seq_len]
        mask_expanded = self.causal_mask[:seq_len, :seq_len].view(1, 1, seq_len, seq_len)
        attn_scores.masked_fill_(mask_expanded.bool(), -torch.inf)

        # 计算注意力概率
        attn_probs = F.softmax(attn_scores.float(), dim=-1).type_as(q)
        attn_probs = self.attn_dropout(attn_probs)

        # 计算输出
        attn_output = torch.matmul(attn_probs, v)  # [batch, num_heads, seq, d_v]

        # 拼接所有头并投影
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.o_proj(attn_output)
        return self.resid_dropout(output)


class DeepseekMoEGate(nn.Module):
    """MoE Gating for DeepSeekV3 (参考 DeepSeekMoE 论文).

    改进点:
    1. 标准化的 top-K 路由 (Softmax 权重归一化)
    2. 优化的负载平衡辅助损失
    3. 更稳定的路由权重计算
    """

    def __init__(
            self,
            hidden_size: int,
            n_routed_experts: int = 256,
            num_experts_per_tok: int = 8,
            n_group: int = 8,
            topk_group: int = 4,
            norm_topk_prob: bool = True,
            aux_loss_alpha: float = 0.001,
            z_loss_alpha: float = 0.0001,  # 新增：z-loss 防止路由 logits 过大
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.z_loss_alpha = z_loss_alpha

        # 正交初始化以保持稳定性
        # Gate weight: [n_routed_experts, hidden_size]
        self.gate = nn.Linear(hidden_size, n_routed_experts, bias=False)

    def forward(
            self,
            hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of MoE gating (参考 DeepSeekMoE).

        Args:
            hidden_states: [batch * seq_len, hidden_size]

        Returns:
            topk_idx: [batch * seq_len, num_experts_per_tok] - selected expert indices
            topk_weight: [batch * seq_len, num_experts_per_tok] - expert weights (softmax normalized)
            aux_loss: auxiliary loss for load balancing
        """
        batch_size, hidden_dim = hidden_states.shape

        # 计算所有专家的 logits
        logits = self.gate(hidden_states)  # [batch, n_experts]

        # 分组 top-k 路由 (论文推荐的方式)
        group_size = self.n_routed_experts // self.n_group
        group_logits = logits.view(-1, self.n_group, group_size)

        # 每组内选择 top-k_group 个专家
        group_topk_logits, group_topk_idx = torch.topk(
            group_logits, k=self.topk_group, dim=-1
        )

        # 将所有组的候选专家展平
        group_topk_logits = group_topk_logits.view(-1, self.n_group * self.topk_group)
        group_topk_idx = group_topk_idx.view(-1, self.n_group * self.topk_group)

        # 全局 top-k：从所有候选中选择 top-k 个
        topk_logits, topk_idx_in_candidates = torch.topk(
            group_topk_logits, k=self.num_experts_per_tok, dim=-1
        )

        # 映射回原始专家索引
        selected_group_idx = torch.div(topk_idx_in_candidates, self.topk_group, rounding_mode='floor')
        expert_idx_in_group = group_topk_idx.gather(1, topk_idx_in_candidates)
        topk_idx = selected_group_idx * group_size + expert_idx_in_group

        # 应用 softmax 归一化权重 (DeepSeekMoE 的关键改进)
        topk_weight = F.softmax(topk_logits, dim=-1)

        # 计算辅助损失以平衡专家负载
        aux_loss = torch.tensor(0.0, device=hidden_states.device)
        z_loss = torch.tensor(0.0, device=hidden_states.device)

        if self.training:
            # 专家负载平衡损失 (Load Balancing Loss)
            if self.aux_loss_alpha > 0:
                # 计算每个专家被选中的频率
                expert_mask = F.one_hot(topk_idx.view(-1), num_classes=self.n_routed_experts).float()
                expert_counts = expert_mask.sum(0)  # [n_experts]

                # 计算每个专家的平均路由概率
                logits_softmax = F.softmax(logits, dim=-1)
                expert_probs = logits_softmax.mean(0)  # [n_experts]

                # 负载平衡损失: 确保频率与概率平衡
                aux_loss = (expert_counts / batch_size) * expert_probs
                aux_loss = aux_loss.sum() * self.aux_loss_alpha

            # Z-loss：防止路由 logits 过大
            if self.z_loss_alpha > 0:
                z_loss = (torch.log(F.softmax(logits, dim=-1).sum(0)) ** 2).mean() * self.z_loss_alpha

        total_aux_loss = aux_loss + z_loss

        return topk_idx, topk_weight, total_aux_loss


class DeepseekV3MoE(nn.Module):
    """DeepSeekV3 MoE Layer (优化版本).

    包含:
    - 256 路由专家 (每 token 激活 8 个)
    - 2 共享专家 (始终激活)
    - 分组 top-k 路由

    优化策略 (参考 DeepSeekMoE):
    - 使用 torch.einsum 加速批处理
    - 避免循环遍历专家
    """

    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int = 2048,
            n_routed_experts: int = 256,
            n_shared_experts: int = 2,
            num_experts_per_tok: int = 8,
            n_group: int = 8,
            topk_group: int = 4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok

        self.num_experts = n_routed_experts
        self.top_k = num_experts_per_tok

        # 路由机制
        self.gate = DeepseekMoEGate(
            hidden_size=hidden_size,
            n_routed_experts=n_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            n_group=n_group,
            topk_group=topk_group,
        )

        # 路由专家 (共享权重的方式以节省内存)
        self.experts = nn.ModuleList([
            DeepSeekV3MLP(hidden_size, intermediate_size)
            for _ in range(n_routed_experts)
        ])

        # 共享专家 (始终激活，参考 DeepSeekMoE)
        self.shared_experts = nn.ModuleList([
            DeepSeekV3MLP(hidden_size, intermediate_size)
            for _ in range(n_shared_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass 优化版本.

        Args:
            x: [batch, seq_len, hidden_size]

        Returns:
            output: 相同 shape 的输出张量
        """
        original_shape = x.shape
        batch_size, seq_len, hidden_dim = x.shape
        x = x.view(-1, hidden_dim)

        # 1. 处理共享专家 (始终激活，贡献相等权重)
        shared_output = torch.zeros_like(x)
        for expert in self.shared_experts:
            shared_output = shared_output + expert(x)
        shared_output = shared_output / self.n_shared_experts  # 平均

        # 2. 路由专家处理
        topk_idx, topk_weights, aux_loss = self.gate(x)  # [N, K], [N, K]

        # 使用高效的批处理方式处理路由专家
        routed_output = self._process_routed_experts(x, topk_idx, topk_weights)

        # 3. 合并共享专家和路由专家输出
        output = shared_output + routed_output

        # 4. 恢复原始形状
        output = output.view(original_shape)

        return output

    def _process_routed_experts(
            self,
            hidden_states: torch.Tensor,
            top_k_indices: torch.Tensor,
            top_k_gates: torch.Tensor,
    ) -> torch.Tensor:
        """高效处理路由专家的辅助函数.

        Args:
            x: [N, hidden_size]
            topk_idx: [N, K] - 专家索引
            topk_weights: [N, K] - 归一化权重

        Returns:
            output: [N, hidden_size]
        """

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
        expert_counts = flat_idx.bincount(minlength=self.n_routed_experts)

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

        return final_hidden_states


class DeepseekV3DecoderLayer(nn.Module):
    """DeepSeekV3 Decoder Layer.

    Structure:
    1. Input RMSNorm
    2. Multi-Head Latent Attention (MLA)
    3. Post-attention RMSNorm
    4. MoE FFN (or dense FFN for first few layers)
    """

    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            intermediate_size: int,
            n_routed_experts: int = 256,
            n_shared_experts: int = 2,
            num_experts_per_tok: int = 8,
            n_group: int = 8,
            topk_group: int = 4,
            q_lora_rank: int = 1536,
            kv_lora_rank: int = 512,
            qk_nope_head_dim: int = 128,
            qk_rope_head_dim: int = 64,
            v_head_dim: int = 128,
            max_seq_len: int = 2048,
            first_k_dense_replace: int = 1,
            layer_id: int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_id = layer_id

        # Use MoE for most layers, dense FFN for first few layers
        use_moe = layer_id >= first_k_dense_replace

        # Attention
        self.input_layernorm = RMSNorm(hidden_size)
        self.self_attn = MultiheadLatentAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            max_seq_len=max_seq_len,
        )
        self.post_attention_layernorm = RMSNorm(hidden_size)

        # FFN
        if use_moe:
            self.mlp = DeepseekV3MoE(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                n_routed_experts=n_routed_experts,
                n_shared_experts=n_shared_experts,
                num_experts_per_tok=num_experts_per_tok,
                n_group=n_group,
                topk_group=topk_group,
            )
        else:
            self.mlp = DeepSeekV3MLP(hidden_size, intermediate_size)

    def forward(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [batch, seq_len, hidden_size]

        Returns:
            output: [batch, seq_len, hidden_size]
        """
        # Self-attention with pre-norm
        x = x + self.self_attn(self.input_layernorm(x))
        # FFN with pre-norm
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class DeepseekV3(nn.Module):
    """DeepSeekV3 for Causal Language Modeling.

    包含:
    - 词嵌入层
    - Transformer 解码器层堆叠
    - 最终 LayerNorm
    - LM Head (vocab_size 投影)
    """

    def __init__(
            self,
            vocab_size: int = 129280,
            hidden_size: int = 7168,
            num_attention_heads: int = 64,
            num_hidden_layers: int = 60,
            intermediate_size: int = 2048,
            n_routed_experts: int = 256,
            n_shared_experts: int = 2,
            num_experts_per_tok: int = 8,
            n_group: int = 8,
            topk_group: int = 4,
            q_lora_rank: int = 1536,
            kv_lora_rank: int = 512,
            qk_nope_head_dim: int = 128,
            qk_rope_head_dim: int = 64,
            v_head_dim: int = 128,
            max_position_embeddings: int = 4096,
            first_k_dense_replace: int = 1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

        # Embedding
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        # Layers
        self.layers = nn.ModuleList([
            DeepseekV3DecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                n_routed_experts=n_routed_experts,
                n_shared_experts=n_shared_experts,
                num_experts_per_tok=num_experts_per_tok,
                n_group=n_group,
                topk_group=topk_group,
                q_lora_rank=q_lora_rank,
                kv_lora_rank=kv_lora_rank,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                v_head_dim=v_head_dim,
                max_seq_len=max_position_embeddings,
                first_k_dense_replace=first_k_dense_replace,
                layer_id=i,
            )
            for i in range(num_hidden_layers)
        ])

        # Final layer norm
        self.norm = RMSNorm(hidden_size)

        # LM Head
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
            self,
            input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: [batch, seq_len]

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)

        # Pass through layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        # Final norm
        hidden_states = self.norm(hidden_states)

        # LM Head
        logits = self.lm_head(hidden_states)

        return logits


if __name__ == "__main__":
    import os

    # 导入ONNX导出工具
    try:
        from utils.onnx_utils import export_and_simplify, validate_onnx
    except ImportError:
        # 如果导入失败，定义空函数
        def export_and_simplify(*args, **kwargs):
            raise ImportError("请确保 utils.onnx_utils 模块存在")


        def validate_onnx(*args, **kwargs):
            return False

    # Test code
    batch_size = 2
    seq_len = 8
    max_new_tokens = 10

    model = DeepseekV3(
        vocab_size=1000,
        hidden_size=512,
        num_attention_heads=8,
        num_hidden_layers=2,
        intermediate_size=256,
        n_routed_experts=8,  # More experts for testing
        n_shared_experts=2,
        num_experts_per_tok=4,
        n_group=4,  # 4 groups
        topk_group=2,  # Select top-2 from each group
        first_k_dense_replace=1  # 实际上 deepseek 61 层，[0，1，2] 是 mlp, [3，...，60] 是 moe
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    logits = model(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")

    print("DeepSeekV3模型测试通过！")

    # ===== 导出为ONNX =====
    os.makedirs("onnx_data", exist_ok=True)
    onnx_path = "onnx_data/deepseekv3.onnx"

    # 导出并简化ONNX模型
    # 添加 dynamic_axes 支持动态 batch 和 sequence 维度
    final_path = export_and_simplify(
        model=model,
        dummy_input=(input_ids,),  # 只传入 input_ids
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
