"""
THD（Time-Head-Dimension）Packing 工具包

提供将 BSHD 格式（带 padding 的批数据）转换为 THD 格式（无 padding 连续张量）
的完整工具链，包括：
  - packing: 核心转换函数（unpad/pad/cu_seqlens）
  - attention: THD 格式的可变长度多头注意力
  - model: 完整的 THD GPT Transformer 模型
"""

from thd.packing import PackedSeqParams, compute_cu_seqlens, unpad_input, pad_input

__all__ = [
    "PackedSeqParams",
    "compute_cu_seqlens",
    "unpad_input",
    "pad_input",
]

