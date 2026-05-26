"""
THD 格式的可变长度多头注意力（VarLen Multi-Head Attention）

核心区别：
  - BSHD Attention: 输入 [batch_size, seq_len, d_model]，所有样本等长（含 padding）
  - THD Attention:   输入 [total_nnz, d_model]，无 padding，通过 cu_seqlens 区分样本边界

THD 注意力的关键：
  1. 不需要传统的方形 attention mask（padding 区域已被移除）
  2. 使用 cu_seqlens 构建因果掩码的稀疏版本
  3. 每个样本内部做因果注意力，不同样本之间完全隔离
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from thd.packing import PackedSeqParams


def build_varlen_causal_mask(
    cu_seqlens: torch.Tensor,
    total_nnz: int,
) -> torch.Tensor:
    """
    为 THD 格式构建因果（causal）attention mask

    在 THD 格式中，不同样本的 token 被拼接在一起。因果 mask 需要保证：
      - 每个样本内部：第 i 个位置只能看到前 i 个位置（含自身）
      - 不同样本之间：完全隔离（不能互相 attention）

    实现方式：为每个 (q_pos, k_pos) 对判断是否属于同一样本且 q_pos >= k_pos

    Args:
        cu_seqlens: 累积序列长度 [batch_size + 1]
        total_nnz: 有效 token 总数

    Returns:
        causal_mask: bool 张量 [total_nnz, total_nnz]
                      True 表示需要 mask 掉的位置（即不允许 attend 的位置）

    Example:
        >>> cu_seqlens = torch.tensor([0, 3, 7])  # 两个样本，长度分别为 3 和 4
        >>> mask = build_varlen_causal_mask(cu_seqlens, 7)
        >>> # 样本0内部: 上三角被mask
        >>> # 样本1内部: 上三角被mask
        >>> # 跨样本区域: 全部被mask
    """
    device = cu_seqlens.device
    batch_size = cu_seqlens.shape[0] - 1

    # 构建 [total_nnz] 的 sample_id：标记每个 token 属于哪个样本
    sample_ids = torch.zeros(total_nnz, dtype=torch.long, device=device)
    for i in range(batch_size):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()
        sample_ids[start:end] = i

    # 构建 [total_nnz] 的 intra_pos：每个 token 在其所属样本内的相对位置
    intra_pos = torch.zeros(total_nnz, dtype=torch.long, device=device)
    for i in range(batch_size):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()
        seqlen = end - start
        intra_pos[start:end] = torch.arange(seqlen, device=device)

    # 因果条件：(q 和 k 属于同一样本) AND (q的相对位置 < k的相对位置) → 需要 mask
    same_sample = sample_ids.unsqueeze(0) == sample_ids.unsqueeze(1)  # [nnz, nnz]
    q_before_k = intra_pos.unsqueeze(0) < intra_pos.unsqueeze(1)       # [nnz, nnz]

    # 需要 mask 的位置 = 不同样本 OR q在k之前（因果违反）
    causal_mask = ~(same_sample & ~q_before_k)  # True = mask out

    return causal_mask


class VarLenMultiHeadAttention(nn.Module):
    """
    THD 格式的可变长度多头注意力

    与标准 BSHD 注意力的区别：
      - 输入是 [total_nnz, d_model] 而非 [batch_size, seq_len, d_model]
      - 通过 PackedSeqParams.cu_seqlens 区分样本边界
      - 使用稀疏因果 mask 替代固定大小的上三角矩阵
      - 无 padding 计算，效率更高
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
    ):
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

    def forward(
        self,
        x: torch.Tensor,
        params: PackedSeqParams,
    ) -> torch.Tensor:
        """
        THD 格式的多头注意力前向传播

        Args:
            x: THD 格式的输入，shape [total_nnz, d_model]
            params: PackedSeqParams 元信息（包含 cu_seqlens 等）

        Returns:
            output: THD 格式的输出，shape [total_nnz, d_model]
        """
        total_nnz = params.total_nnz

        # ---- 1. 线性投影并拆分多头 ----
        # Q, K, V: [total_nnz, n_heads, d_k]
        q = self.w_q(x).view(total_nnz, self.n_heads, self.d_k)
        k = self.w_k(x).view(total_nnz, self.n_heads, self.d_k)
        v = self.w_v(x).view(total_nnz, self.n_heads, self.d_k)

        # 转置为 [n_heads, total_nnz, d_k] 用于 batch matmul
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        # ---- 2. 计算注意力分数 ----
        # [n_heads, total_nnz, total_nnz]
        scale = math.sqrt(self.d_k)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        # ---- 3. 应用 THD 因果 mask ----
        causal_mask = build_varlen_causal_mask(params.cu_seqlens, total_nnz)
        # 扩展到 head 维度: [1, total_nnz, total_nnz]
        attn_scores.masked_fill_(causal_mask.unsqueeze(0), float("-inf"))

        # ---- 4. Softmax + Dropout + Weighted Sum ----
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # [n_heads, total_nnz, d_k]
        attn_output = torch.matmul(attn_weights, v)

        # ---- 5. 合并多头并投影 ----
        # [total_nnz, n_heads, d_k]
        attn_output = attn_output.transpose(0, 1).contiguous()
        # [total_nnz, d_model]
        attn_output = attn_output.view(total_nnz, self.d_model)

        output = self.w_o(attn_output)
        return output


class FeedForwardTHD(nn.Module):
    """
    THD 格式的前馈网络（与 BSHD 版本逻辑相同，只是输入输出维度不同）

    FFN 是逐 token 操作，不涉及跨 token 交互，
    所以 THD 和 BSHD 的计算结果完全一致（仅排列顺序不同）。
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activate = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [total_nnz, d_model]
        Returns:
            output: [total_nnz, d_model]
        """
        x = self.linear1(x)
        x = self.activate(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class GPTTransformerBlockTHD(nn.Module):
    """THD 格式的 GPT Transformer 块（Pre-LN 结构）"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = VarLenMultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForwardTHD(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        params: PackedSeqParams,
    ) -> torch.Tensor:
        """
        Args:
            x: [total_nnz, d_model]
            params: PackedSeqParams 元信息
        Returns:
            output: [total_nnz, d_model]
        """
        # 自注意力子层（残差连接）
        x_residual = x
        x = self.ln1(x)
        x = self.attn(x, params)
        x = self.dropout(x)
        x = x + x_residual

        # 前馈子层（残差连接）
        x_residual = x
        x = self.ln2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x + x_residual

        return x


if __name__ == "__main__":
    print("=" * 60)
    print("THD VarLen 多头注意力测试")
    print("=" * 60)

    from thd.packing import compute_cu_seqlens, unpad_input, pad_input

    # ---- 测试参数 ----
    d_model = 64
    n_heads = 4
    d_ff = 256
    batch_size = 3
    seq_len = 8

    # 模拟变长序列
    attention_mask = torch.tensor([
        [1, 1, 1, 1, 0, 0, 0, 0],   # 样本0: 长度 4
        [1, 1, 1, 1, 1, 1, 0, 0],   # 样本1: 长度 6
        [1, 1, 1, 0, 0, 0, 0, 0],   # 样本2: 长度 3
    ])
    hidden_states_bshd = torch.randn(batch_size, seq_len, d_model)

    print(f"\nBSHD 输入形状: {hidden_states_bshd.shape}")

    # ---- 1. Packing: BSHD → THD ----
    hidden_thd, indices, params = unpad_input(hidden_states_bshd, attention_mask)
    print(f"THD 输入形状: {hidden_thd.shape}")
    print(f"cu_seqlens: {params.cu_seqlens.tolist()}")
    print(f"total_nnz: {params.total_nnz}")

    # ---- 2. THD Attention 前向 ----
    attn = VarLenMultiHeadAttention(d_model, n_heads)
    output_thd = attn(hidden_thd, params)
    print(f"\nTHD Attention 输出形状: {output_thd.shape}")

    assert output_thd.shape == (params.total_nnz, d_model), \
        f"输出形状错误: {output_thd.shape}"

    # ---- 3. Unpacking: THD → BSHD ----
    output_bshd = pad_input(output_thd, indices, batch_size, seq_len)
    print(f"恢复后 BSHD 形状: {output_bshd.shape}")

    # ---- 4. 验证 padding 区域为零 ----
    print("\n--- Padding 区域验证 ---")
    seqlens, _ = compute_cu_seqlens(attention_mask)
    for i in range(batch_size):
        valid_len = int(seqlens[i].item())
        padding_out = output_bshd[i, valid_len:]
        max_val = padding_out.abs().max().item()
        print(f"  样本 {i}: padding 区域最大值 = {max_val:.2e} (应接近 0)")

    # ---- 5. 完整 Transformer Block 测试 ----
    print("\n--- 完整 Transformer Block 测试 ---")
    block = GPTTransformerBlockTHD(d_model, n_heads, d_ff)
    block_output_thd = block(hidden_thd, params)
    block_output_bshd = pad_input(block_output_thd, indices, batch_size, seq_len)
    print(f"Block 输出 BSHD 形状: {block_output_bshd.shape}")
    assert block_output_bshd.shape == (batch_size, seq_len, d_model)

    print("\n✅ 所有 THD Attention 测试通过！")

