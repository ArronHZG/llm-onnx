"""
THD（Time-Head-Dimension）Packing 核心工具函数

核心思想：将多个变长序列拼接成一维连续张量（无 padding），避免 padding token 带来的计算浪费。

数据格式对比：
  - BSHD 格式 (Batch-Sequence-Head-Dimension): 传统格式，所有样本 padding 到相同长度
  - THD 格式 (Time-Head-Dimension): 打包格式，所有有效 token 拼接为一维张量，通过 cu_seqlens 标记边界

数据流示意：
  BSHD: input_ids [[1,2,3,0,0], [4,5,6,7,0]]   (bsz=2, seqlen=5)
  THD:  input_ids [1,2,3,4,5,6,7]                (total_nnz=7)
        cu_seqlens [0, 3, 7]                       (累积序列长度)
"""

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class PackedSeqParams:
    """
    THD 格式的元信息参数

    Attributes:
        cu_seqlens: 累积序列长度，shape [batch_size + 1]
                    cu_seqlens[0]=0, cu_seqlens[i] = 前 i-1 个样本的有效 token 数之和
        max_seqlen: batch 中最大序列长度
        total_nnz: 有效 token 总数（即 THD 张量的 Time 维度）
        batch_size: 原始 batch 大小
    """

    cu_seqlens: torch.Tensor  # [batch_size + 1], int32
    max_seqlen: int
    total_nnz: int
    batch_size: int


def compute_cu_seqlens(
    attention_mask: torch.Tensor,
    dtype: torch.dtype = torch.int32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从 attention mask 计算每个样本的实际序列长度和累积序列长度 (cu_seqlens)

    Args:
        attention_mask: 注意力掩码，shape [batch_size, seq_len]
                        值为 1 表示有效 token，0 表示 padding
        dtype: 输出张量的数据类型，默认 int32

    Returns:
        seqlens_in_batch: 每个样本的有效长度，shape [batch_size]
        cu_seqlens: 累积序列长度，shape [batch_size + 1]

    Example:
        >>> mask = torch.tensor([[1,1,1,0,0], [1,1,1,1,0]])
        >>> seqlens, cu_seqlens = compute_cu_seqlens(mask)
        >>> seqlens    # tensor([3, 4])
        >>> cu_seqlens  # tensor([0, 3, 7])
    """
    batch_size, _ = attention_mask.shape
    device = attention_mask.device

    # 每个样本的有效 token 数
    seqlens_in_batch = attention_mask.sum(dim=-1).to(dtype)

    # 构建累积序列长度: [0, L₁, L₁+L₂, ..., ΣLᵢ]
    cu_seqlens = torch.zeros(batch_size + 1, dtype=dtype, device=device)
    cu_seqlens[1:] = torch.cumsum(seqlens_in_batch, dim=0)

    return seqlens_in_batch, cu_seqlens


def unpad_input(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, PackedSeqParams]:
    """
    将 BSHD 格式转换为 THD 格式（去除 padding token）

    原理：根据 attention_mask 找出所有有效 token 的位置，
          将其索引出来拼接到一维连续张量中。

    Args:
        hidden_states: 输入张量，shape [batch_size, seq_len, *]
                       （支持任意额外的尾随维度）
        attention_mask: 注意力掩码，shape [batch_size, seq_len]

    Returns:
        hidden_states_rmpad: 去除 padding 后的张量，shape [total_nnz, *]
        indices: 每个 THD 位置对应的原始展平索引，shape [total_nnz]
                 用于后续 pad_input 恢复原始形状
        params: PackedSeqParams 元信息

    Example:
        >>> x = torch.tensor([[[1.],[2.],[3.],[0.],[0.]],
        ...                   [[4.],[5.],[6.],[7.],[0.]]])
        >>> mask = torch.tensor([[1,1,1,0,0], [1,1,1,1,0]])
        >>> x_rmpad, indices, params = unpad_input(x, mask)
        >>> x_rmpad.shape  # torch.Size([7, 1])  — 3+4=7 个有效 token
    """
    batch_size, seq_len = attention_mask.shape
    device = hidden_states.device
    original_shape = hidden_states.shape  # 保存原始 shape 用于恢复

    # 计算累积序列长度
    seqlens, cu_seqlens = compute_cu_seqlens(attention_mask)
    max_seqlen = int(seqlens.max().item())
    total_nnz = int(seqlens.sum().item())

    if total_nnz == 0:
        raise ValueError("No non-padding tokens found in the input")

    # 将输入展平到二维: [batch_size * seq_len, ...]
    flat_hidden = hidden_states.reshape(batch_size * seq_len, *original_shape[2:])

    # 获取非零位置的展平索引
    # attention_mask 展平后，非零元素的位置即为有效 token 的索引
    flat_mask = attention_mask.reshape(-1)  # [batch_size * seq_len]
    indices = torch.nonzero(flat_mask, as_tuple=False).squeeze(-1)  # [total_nnz]

    # 按索引取出有效 token → THD 格式
    hidden_states_rmpad = flat_hidden[indices]  # [total_nnz, *]

    # 构建元信息
    params = PackedSeqParams(
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        total_nnz=total_nnz,
        batch_size=batch_size,
    )

    return hidden_states_rmpad, indices, params


def pad_input(
    hidden_states: torch.Tensor,
    indices: torch.Tensor,
    batch_size: int,
    seqlen: int,
    padding_value: float = 0.0,
) -> torch.Tensor:
    """
    将 THD 格式恢复为 BSHD 格式（填充回原始位置）

    Args:
        hidden_states: THD 格式的张量，shape [total_nnz, *]
        indices: unpad_input 返回的索引，shape [total_nnz]
        batch_size: 原始 batch 大小
        seqlen: 原始序列长度（含 padding）
        padding_value: 填充值，默认 0.0

    Returns:
        output: BSHD 格式的张量，shape [batch_size, seqlen, *]

    Example:
        >>> # 接上例 unpad_input 的输出
        >>> restored = pad_input(x_rmpad, indices, batch_size=2, seqlen=5)
        >>> restored.shape  # torch.Size([2, 5, 1])
    """
    total_nnz = hidden_states.shape[0]
    tail_dims = hidden_states.shape[1:]  # 额外维度

    # 创建全零的输出张量: [batch_size * seqlen, *]
    output_flat = torch.zeros(
        batch_size * seqlen, *tail_dims,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    output_flat[:] = padding_value

    # 将 THD 数据放回原始位置
    output_flat[indices.clone()] = hidden_states

    # reshape 回 BSHD: [batch_size, seqlen, *]
    output = output_flat.reshape(batch_size, seqlen, *tail_dims)

    return output


def unpad_input_only_mask(attention_mask: torch.Tensor) -> Tuple[torch.Tensor, PackedSeqParams]:
    """
    仅基于 attention mask 计算 THD 元信息（不处理实际数据）

    用于需要提前获取 packing 参数的场景。

    Args:
        attention_mask: 注意力掩码，shape [batch_size, seq_len]

    Returns:
        indices: 有效 token 的展平索引，shape [total_nnz]
        params: PackedSeqParams 元信息
    """
    _, cu_seqlens = compute_cu_seqlens(attention_mask)
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    max_seqlen = int(seqlens.max().item())
    total_nnz = int(seqlens.sum().item())
    batch_size = attention_mask.shape[0]

    flat_mask = attention_mask.reshape(-1)
    indices = torch.nonzero(flat_mask.bool(), as_tuple=False).squeeze(-1)

    params = PackedSeqParams(
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        total_nnz=total_nnz,
        batch_size=batch_size,
    )
    return indices, params


def build_cu_seqlens_from_indices(
    indices: torch.Tensor,
    batch_size: int,
    seq_len: int,
) -> torch.Tensor:
    """
    从 indices 反推 cu_seqlens（用于已知 THD 索引但缺少原始 mask 的场景）

    通过统计 indices 在每个样本区间内的数量来重建 cu_seqlens。

    Args:
        indices: 展平索引，shape [total_nnz]
        batch_size: 原始 batch 大小
        seq_len: 原始序列长度

    Returns:
        cu_seqlens: 累积序列长度，shape [batch_size + 1]
    """
    device = indices.device
    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)

    for i in range(batch_size):
        start = i * seq_len
        end = (i + 1) * seq_len
        # 统计落在当前样本区间内的 index 数量
        count = ((indices >= start) & (indices < end)).sum().to(torch.int32)
        cu_seqlens[i + 1] = cu_seqlens[i] + count

    return cu_seqlens


# ============================================================
# 便捷包装：对 input_ids 和 position_ids 做 unpad
# ============================================================

def unpack_inputs(
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, PackedSeqParams]:
    """
    对 input_ids 和 position_ids 同时做 unpad（BSHD → THD）

    Args:
        input_ids: [batch_size, seq_len]
        position_ids: [batch_size, seq_len]
        attention_mask: [batch_size, seq_len]

    Returns:
        input_ids_thd: [total_nnz]
        position_ids_thd: [total_nnz]
        params: PackedSeqParams 元信息
    """
    input_ids_expanded = input_ids.unsqueeze(-1).float()  # [bsz, seqlen, 1]
    input_ids_rmpad, indices, params = unpad_input(input_ids_expanded, attention_mask)
    input_ids_thd = input_ids_rmpad.squeeze(-1).long()  # [total_nnz]

    pos_expanded = position_ids.unsqueeze(-1).float()  # [bsz, seqlen, 1]
    pos_rmpad, _, _ = unpad_input(pos_expanded, attention_mask)
    position_ids_thd = pos_rmpad.squeeze(-1).long()  # [total_nnz]

    return input_ids_thd, position_ids_thd, params, indices


if __name__ == "__main__":
    print("=" * 60)
    print("THD Packing 工具函数测试")
    print("=" * 60)

    # ---- 测试数据 ----
    batch_size = 3
    seq_len = 8
    d_model = 16

    # 构造变长序列（模拟真实场景：不同样本有不同有效长度）
    attention_mask = torch.tensor([
        [1, 1, 1, 1, 0, 0, 0, 0],  # 样本0: 有效长度 4
        [1, 1, 1, 1, 1, 1, 0, 0],  # 样本1: 有效长度 6
        [1, 1, 1, 0, 0, 0, 0, 0],  # 样本2: 有效长度 3
    ])
    hidden_states = torch.randn(batch_size, seq_len, d_model)

    print(f"\n原始 BSHD 形状: {hidden_states.shape}")
    print(f"Attention Mask:\n{attention_mask.int()}")

    # ---- 1. compute_cu_seqlens ----
    print("\n--- 1. compute_cu_seqlens ---")
    seqlens, cu_seqlens = compute_cu_seqlens(attention_mask)
    print(f"各样本有效长度 (seqlens): {seqlens.tolist()}")
    print(f"累积序列长度 (cu_seqlens): {cu_seqlens.tolist()}")
    assert cu_seqlens.tolist() == [0, 4, 10, 13], f"cu_seqlens 计算错误: {cu_seqlens.tolist()}"

    # ---- 2. unpad_input (BSHD → THD) ----
    print("\n--- 2. unpad_input (BSHD → THD) ---")
    thd_hidden, indices, params = unpad_input(hidden_states, attention_mask)
    print(f"THD 形状: {thd_hidden.shape}")  # 应为 [13, 16]
    print(f"indices 形状: {indices.shape}")
    print(f"PackedSeqParams:")
    print(f"  cu_seqlens: {params.cu_seqlens.tolist()}")
    print(f"  max_seqlen: {params.max_seqlen}")
    print(f"  total_nnz: {params.total_nnz}")
    print(f"  batch_size: {params.batch_size}")

    assert thd_hidden.shape == (13, d_model), f"THD 形状错误: {thd_hidden.shape}"
    assert params.total_nnz == 13, f"total_nnz 错误: {params.total_nnz}"

    # ---- 3. pad_input (THD → BSHD) ----
    print("\n--- 3. pad_input (THD → BSHD) ---")
    restored = pad_input(thd_hidden, indices, batch_size, seq_len)
    print(f"恢复后 BSHD 形状: {restored.shape}")
    assert restored.shape == (batch_size, seq_len, d_model), \
        f"恢复形状错误: {restored.shape}"

    # ---- 4. 验证数值一致性 ----
    print("\n--- 4. 数值一致性验证 ---")
    for i in range(batch_size):
        valid_len = int(seqlens[i].item())
        original = hidden_states[i, :valid_len]
        recovered = restored[i, :valid_len]
        diff = (original - recovered).abs().max().item()
        print(f"  样本 {i}: 最大误差 = {diff:.2e}")
        assert diff < 1e-6, f"样本 {i} 数值不一致!"

    # 验证 padding 区域为零
    for i in range(batch_size):
        valid_len = int(seqlens[i].item())
        padding_region = restored[i, valid_len:]
        assert padding_region.abs().max().item() < 1e-6, \
            f"样本 {i} padding 区域不为零!"

    print("\n✅ 所有测试通过！")

