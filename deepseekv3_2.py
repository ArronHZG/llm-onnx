"""DeepSeekV3.2 Model Implementation.

DeepSeekV3.2 是 DeepSeekV3 的改进版本，具有：
- Lightning Indexer (NSA - Native Sparse Attention) 用于高效稀疏注意力
- 256 个路由专家，每个 token 激活 8 个
- 分组 top-k 路由 (8 组，每组 4 个专家)
- 2 个共享专家以改进性能
- 多头潜在注意力 (MLA) 用于高效 KV 缓存

关键特性:
1. Lightning Indexer (NSA): 使用 Indexer 类实现稀疏注意力
   - index_n_heads: 索引头数
   - index_head_dim: 索引头维度 (128)
   - index_topk: top-k 稀疏选择
2. Top-k Selector: 使用 grouped_topk 路由策略
   - n_group: 专家分组数 (8)
   - topk_group: 每组选择数 (4)

References:
- DeepSeekV3: https://arxiv.org/abs/2401.14163
- DeepSeekV2: https://arxiv.org/abs/2405.00041
- NSA: https://arxiv.org/abs/2502.11089
"""

import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepseekv3 import DeepseekMoEGate, DeepseekV3MoE
from gpt import FeedForward
from gpt_rope import apply_rotary_pos_emb, precompute_freqs_cis

# 简化模型结构
SwiGLU = FeedForward
RMSNorm = nn.LayerNorm

# Deepseek mlp 就是 SwiGLU
DeepSeekV3MLP = SwiGLU


class LightningIndexerNSA(nn.Module):
    """Lightning Indexer for NSA (Native Sparse Attention).

    这是 DeepSeekV3.2 的关键特性，用于高效处理长序列的稀疏注意力。
    通过索引选择重要的 token 进行注意力计算，减少计算量。

    参数:
        hidden_size: 隐藏层维度
        index_n_heads: 索引注意力头数
        index_head_dim: 每个索引头的维度 (默认 128)
        index_topk: 稀疏选择的 top-k 数量
    """

    def __init__(
        self,
        hidden_size: int,
        index_n_heads: int = 8,
        index_head_dim: int = 128,
        index_topk: int = 8,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.max_seq_len = max_seq_len

        # 索引投影矩阵
        self.index_proj = nn.Linear(hidden_size, index_n_heads * index_head_dim, bias=False)

        # 索引筛选的门控机制
        self.index_gate = nn.Linear(hidden_size, index_n_heads, bias=False)

        # 预计算位置编码
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=index_head_dim,
            end=max_seq_len,
            rope_base=10000.0
        )
        self.register_buffer('freqs_cos', freqs_cos, persistent=False)
        self.register_buffer('freqs_sin', freqs_sin, persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of Lightning Indexer.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: 可选的注意力掩码

        Returns:
            index_states: 索引选择后的隐藏状态 [batch, index_n_heads * index_topk, index_head_dim]
            index_mask: 索引掩码
        """
        batch_size, seq_len, _ = hidden_states.shape

        # 投影到索引空间
        index_states = self.index_proj(hidden_states)  # [batch, seq, index_n_heads * index_head_dim]
        index_states = index_states.view(batch_size, seq_len, self.index_n_heads, self.index_head_dim)

        # 门控分数
        gate_scores = self.index_gate(hidden_states)  # [batch, seq, index_n_heads]

        # 对每个头选择 top-k 索引
        index_states_list = []
        index_mask_list = []

        for head_idx in range(self.index_n_heads):
            head_gate = gate_scores[:, :, head_idx]  # [batch, seq]
            head_index_states = index_states[:, :, head_idx, :]  # [batch, seq, index_head_dim]

            # 选择 top-k
            topk_values, topk_indices = torch.topk(head_gate, min(self.index_topk, seq_len), dim=-1)

            # 根据索引 gather 对应的 hidden states
            batch_indices = torch.arange(batch_size, device=hidden_states.device).unsqueeze(1).expand(-1, self.index_topk)

            selected_states = head_index_states[batch_indices, topk_indices]  # [batch, index_topk, index_head_dim]

            # 创建掩码
            head_mask = torch.zeros(batch_size, seq_len, device=hidden_states.device)
            head_mask[batch_indices, topk_indices] = 1.0

            index_states_list.append(selected_states)
            index_mask_list.append(head_mask)

        # 拼接所有头的索引状态
        index_states = torch.stack(index_states_list, dim=1)  # [batch, index_n_heads, index_topk, index_head_dim]
        index_states = index_states.view(batch_size, self.index_n_heads * self.index_topk, self.index_head_dim)

        # 合并所有头的掩码
        index_mask = torch.stack(index_mask_list, dim=1)  # [batch, index_n_heads, seq_len]
        index_mask = index_mask.any(dim=1)  # [batch, seq_len]

        return index_states, index_mask


class MultiheadLatentAttentionV3_2(nn.Module):
    """DeepSeekV3.2 Multi-Head Latent Attention (MLA) with NSA support.

    包含 Lightning Indexer (NSA) 用于稀疏注意力加速。
    """

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
            # NSA 相关参数
            use_nsa: bool = True,
            index_n_heads: int = 8,
            index_head_dim: int = 128,
            index_topk: int = 8,
    ):
        super().__init__()
        self.use_nsa = use_nsa

        # 基本参数设置
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

        # NSA 参数
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk

        # 维度映射
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

        # NV 缓存
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

        # Lightning Indexer (NSA) - DeepSeekV3.2 关键特性
        if self.use_nsa:
            self.indexer = LightningIndexerNSA(
                hidden_size=hidden_size,
                index_n_heads=index_n_heads,
                index_head_dim=index_head_dim,
                index_topk=index_topk,
                max_seq_len=max_seq_len,
            )
            # NSA 稀疏注意力的独立投影层
            # 将 index_states 映射到 Q, K, V 空间 (每个头独立处理)
            self.sparse_q_proj = nn.Linear(index_head_dim, index_head_dim, bias=False)
            self.sparse_k_proj = nn.Linear(index_head_dim, index_head_dim, bias=False)
            self.sparse_v_proj = nn.Linear(index_head_dim, index_head_dim, bias=False)
            # 将稀疏注意力输出映射回 hidden_size
            self.index_output_proj = nn.Linear(index_head_dim, hidden_size, bias=False)

    def forward(self, x, kv_cache=False):
        batch_size, seq_len, _ = x.shape

        # ===== 主干注意力计算 =====

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

        # ===== NSA 稀疏注意力补充计算 =====
        if self.use_nsa and self.indexer is not None:
            # 获取索引选择的稀疏 token
            # index_states: [batch, index_n_heads, index_topk, index_head_dim]
            # index_mask: [batch, seq_len]
            index_states, index_mask = self.indexer(x)

            # 对稀疏 token 计算注意力
            # index_states: [batch, index_n_heads, index_topk, index_head_dim]
            sparse_output = self._compute_sparse_attention(
                x, index_states, index_mask, batch_size, seq_len
            )
            # sparse_output: [batch, index_n_heads * index_topk, hidden_size]

            # 将稀疏输出投影到与 attn_output 相同的维度
            # sparse_output: [batch, index_n_heads * index_topk, hidden_size]
            index_output = sparse_output.view(batch_size, self.index_n_heads, self.index_topk, self.hidden_size)
            # index_output: [batch, index_n_heads, index_topk, hidden_size]

            # 对 index_topk 维度取平均
            index_output = index_output.mean(dim=2)  # [batch, index_n_heads, hidden_size]

            # 扩展到序列维度
            index_output = index_output.unsqueeze(2).expand(-1, -1, seq_len, -1)  # [batch, index_n_heads, seq, hidden_size]

            # 截取前 d_v 维度
            index_output = index_output[:, :, :, :self.d_v]  # [batch, index_n_heads, seq, d_v]

            # 扩展 index_n_heads 到 num_heads (如果不同)
            if self.index_n_heads < self.num_heads:
                # 复制 index_n_heads 的输出到 num_heads
                repeat_factor = self.num_heads // self.index_n_heads
                index_output = index_output.repeat_interleave(repeat_factor, dim=1)
            elif self.index_n_heads > self.num_heads:
                # 截断到 num_heads
                index_output = index_output[:, :self.num_heads, :, :]

            # 残差连接
            attn_output = attn_output + index_output

        # 拼接所有头并投影
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.o_proj(attn_output)
        return self.resid_dropout(output)

    def _compute_sparse_attention(
        self,
        hidden_states: torch.Tensor,
        index_states: torch.Tensor,
        index_mask: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        """计算 NSA 稀疏注意力 (完整 PyTorch 实现).

        NSA (Native Sparse Attention) 的核心：
        1. 使用 Lightning Indexer 选择的索引 token 计算 Q, K, V
        2. 对索引 token 内部计算自注意力
        3. 返回稀疏注意力输出

        Args:
            hidden_states: 原始隐藏状态 [batch, seq, hidden_size]
            index_states: 索引选择后的状态 [batch, index_n_heads * index_topk, index_head_dim]
            index_mask: 索引掩码 [batch, seq_len]
            batch_size: batch 大小
            seq_len: 序列长度

        Returns:
            稀疏注意力输出 [batch, index_n_heads * index_topk, hidden_size]
        """
        # 重塑 index_states: [batch, index_n_heads * index_topk, index_head_dim]
        index_states = index_states.view(
            batch_size, self.index_n_heads, self.index_topk, self.index_head_dim
        )
        # index_states: [batch, index_n_heads, index_topk, index_head_dim]

        # 计算稀疏 Q, K, V
        # index_states: [batch, index_n_heads, index_topk, index_head_dim]
        # -> transpose -> [batch, index_topk, index_n_heads, index_head_dim]
        # -> reshape -> [batch * index_topk * index_n_heads, index_head_dim]
        index_states_per_head = index_states.transpose(1, 2).reshape(
            batch_size * self.index_topk * self.index_n_heads, self.index_head_dim
        )

        # 投影 - 每个头独立投影
        # 投影输入是 [batch * index_topk * index_n_heads, index_head_dim]
        # 投影输出是 [batch * index_topk * index_n_heads, index_head_dim]
        sparse_q_per_head = self.sparse_q_proj(index_states_per_head)  # [batch * index_topk * index_n_heads, index_head_dim]
        sparse_k_per_head = self.sparse_k_proj(index_states_per_head)
        sparse_v_per_head = self.sparse_v_proj(index_states_per_head)

        # reshape 回多头格式
        # [batch * index_topk * index_n_heads, index_head_dim] -> [batch, index_topk, index_n_heads, index_head_dim]
        sparse_q = sparse_q_per_head.view(batch_size, self.index_topk, self.index_n_heads, self.index_head_dim)
        sparse_k = sparse_k_per_head.view(batch_size, self.index_topk, self.index_n_heads, self.index_head_dim)
        sparse_v = sparse_v_per_head.view(batch_size, self.index_topk, self.index_n_heads, self.index_head_dim)

        # transpose: [batch, index_topk, index_n_heads, index_head_dim] -> [batch, index_n_heads, index_topk, index_head_dim]
        sparse_q = sparse_q.transpose(1, 2)
        sparse_k = sparse_k.transpose(1, 2)
        sparse_v = sparse_v.transpose(1, 2)

        # 计算稀疏注意力分数
        # Q @ K^T: [batch, index_n_heads, index_topk, index_head_dim] @ [batch, index_n_heads, index_head_dim, index_topk]
        #       = [batch, index_n_heads, index_topk, index_topk]
        scale = 1.0 / math.sqrt(self.index_head_dim)
        sparse_attn_scores = torch.matmul(sparse_q, sparse_k.transpose(-2, -1)) * scale

        # 应用 softmax 获取注意力权重
        sparse_attn_probs = F.softmax(sparse_attn_scores.float(), dim=-1).type_as(sparse_q)

        # 计算稀疏注意力输出
        # attn_probs @ V: [batch, index_n_heads, index_topk, index_topk] @ [batch, index_n_heads, index_topk, index_head_dim]
        #             = [batch, index_n_heads, index_topk, index_head_dim]
        sparse_attn_output = torch.matmul(sparse_attn_probs, sparse_v)

        # transpose 并重塑: [batch, index_n_heads, index_topk, index_head_dim] -> [batch, index_n_heads * index_topk, index_head_dim]
        sparse_attn_output = sparse_attn_output.transpose(1, 2).contiguous()
        sparse_attn_output = sparse_attn_output.view(batch_size, self.index_n_heads * self.index_topk, self.index_head_dim)

        # 投影回 hidden_size 维度
        # 投影: [batch, index_n_heads * index_topk, index_head_dim] -> [batch, index_n_heads * index_topk, hidden_size]
        sparse_output = self.index_output_proj(sparse_attn_output)

        return sparse_output


class DeepseekV3_2DecoderLayer(nn.Module):
    """DeepSeekV3.2 Decoder Layer.

    Structure:
    1. Input RMSNorm
    2. Multi-Head Latent Attention (MLA) with NSA
    3. Post-attention RMSNorm
    4. MoE FFN (or dense FFN for first few layers)

    关键特性:
    - Lightning Indexer (NSA)
    - grouped_topk 路由
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
            # NSA 参数
            use_nsa: bool = True,
            index_n_heads: int = 8,
            index_head_dim: int = 128,
            index_topk: int = 8,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_id = layer_id

        # Use MoE for most layers, dense FFN for first few layers
        use_moe = layer_id >= first_k_dense_replace

        # Attention with NSA
        self.input_layernorm = RMSNorm(hidden_size)
        self.self_attn = MultiheadLatentAttentionV3_2(
            hidden_size=hidden_size,
            num_heads=num_heads,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            max_seq_len=max_seq_len,
            use_nsa=use_nsa,
            index_n_heads=index_n_heads,
            index_head_dim=index_head_dim,
            index_topk=index_topk,
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
        # Self-attention with pre-norm (包含 NSA)
        x = x + self.self_attn(self.input_layernorm(x))
        # FFN with pre-norm
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class DeepseekV3_2(nn.Module):
    """DeepSeekV3.2 for Causal Language Modeling.

    包含:
    - 词嵌入层
    - Transformer 解码器层堆叠 (支持 NSA)
    - 最终 LayerNorm
    - LM Head (vocab_size 投影)

    关键特性:
    1. Lightning Indexer (NSA):
       - index_n_heads: 索引头数
       - index_head_dim: 索引头维度 (128)
       - index_topk: 稀疏选择数
    2. Top-k Selector (grouped_topk):
       - n_group: 专家分组数 (8)
       - topk_group: 每组选择数 (4)
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
            # NSA 参数
            use_nsa: bool = True,
            index_n_heads: int = 8,
            index_head_dim: int = 128,
            index_topk: int = 8,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

        # NSA 配置
        self.use_nsa = use_nsa
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk

        # Top-k Selector 配置
        self.n_group = n_group
        self.topk_group = topk_group

        # Embedding
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        # Layers
        self.layers = nn.ModuleList([
            DeepseekV3_2DecoderLayer(
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
                use_nsa=use_nsa,
                index_n_heads=index_n_heads,
                index_head_dim=index_head_dim,
                index_topk=index_topk,
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

        # Pass through layers (with NSA support)
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

    # DeepSeekV3.2 配置
    # 包含 NSA 关键参数:
    # - index_n_heads: 8
    # - index_head_dim: 128
    # - index_topk: 8
    # 包含 grouped_topk 关键参数:
    # - n_group: 8
    # - topk_group: 4
    model = DeepseekV3_2(
        vocab_size=1000,
        hidden_size=512,
        num_attention_heads=8,
        num_hidden_layers=2,
        intermediate_size=256,
        n_routed_experts=8,
        n_shared_experts=2,
        num_experts_per_tok=4,
        n_group=4,
        topk_group=2,
        first_k_dense_replace=1,
        # NSA 配置
        use_nsa=True,
        index_n_heads=4,
        index_head_dim=128,
        index_topk=4,
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"NSA enabled: {model.use_nsa}")
    print(f"index_n_heads: {model.index_n_heads}")
    print(f"index_head_dim: {model.index_head_dim}")
    print(f"index_topk: {model.index_topk}")
    print(f"n_group (grouped_topk): {model.n_group}")
    print(f"topk_group (grouped_topk): {model.topk_group}")

    # Test forward pass
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    logits = model(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")

    print("DeepSeekV3.2模型测试通过！")

    # ===== 导出为ONNX =====
    os.makedirs("onnx_data", exist_ok=True)
    onnx_path = "onnx_data/deepseekv3_2.onnx"

    # 导出并简化ONNX模型
    final_path = export_and_simplify(
        model=model,
        dummy_input=(input_ids,),
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

