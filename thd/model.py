"""
THD 格式的完整 GPT Transformer 模型

与标准 BSHD GPT 的对比：
  - BSHD: 输入 [batch_size, seq_len]，内部所有计算基于固定长度的 batch
  - THD:  输入 [batch_size, seq_len] + attention_mask，先 packing 成 [total_nnz, d_model]
          然后在 THD 格式下完成全部 Transformer 计算，最后 pad 回 BSHD

数据流：
  input_ids (BSHD) → embed → unpad (BSHD→THD) → Transformer layers (THD)
  → pad (THD→BSHD) → head → logits (BSHD)
"""

import sys
import os
import math

# 确保能找到项目根目录的模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional

import torch
import torch.nn as nn

from thd.packing import PackedSeqParams, unpad_input, pad_input, compute_cu_seqlens
from thd.attention import VarLenMultiHeadAttention, FeedForwardTHD, build_varlen_causal_mask


class PositionalEncodingTHD(nn.Module):
    """
    THD 格式的位置编码

    与 BSHD 版本的区别：需要根据 cu_seqlens 为每个 token 分配正确的位置编码，
    因为不同样本的 token 被拼接在一起。
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        import numpy as np
        # 预计算位置编码矩阵: [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # [max_len, d_model]

    def forward(self, x: torch.Tensor, params: PackedSeqParams) -> torch.Tensor:
        """
        为 THD 格式的输入添加位置编码

        Args:
            x: THD 嵌入向量，shape [total_nnz, d_model]
            params: PackedSeqParams（包含 cu_seqlens 用于确定每个 token 的绝对位置）

        Returns:
            x_with_pe: 加上位置编码后的张量，shape [total_nnz, d_model]
        """
        total_nnz = x.shape[0]
        cu_seqlens = params.cu_seqlens
        device = x.device

        # 为每个 token 计算其在全局序列中的位置索引
        positions = torch.zeros(total_nnz, dtype=torch.long, device=device)
        batch_size = cu_seqlens.shape[0] - 1
        for i in range(batch_size):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            seqlen = end - start
            # 样本 i 内部的 token 的相对位置为 0, 1, ..., seqlen-1
            positions[start:end] = torch.arange(seqlen, device=device)

        # 取出对应位置的编码并相加
        # self.pe: [max_len, d_model] → 取出 positions 对应的行 → [total_nnz, d_model]
        x = x + self.pe[positions]
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

    def forward(self, x: torch.Tensor, params: PackedSeqParams) -> torch.Tensor:
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


class GPTTransformerTHD(nn.Module):
    """
    完整的 THD 格式 GPT Transformer 模型

    支持两种模式：
      - use_remove_padding=True（默认）：使用 THD packing，避免 padding 计算
      - use_remove_padding=False：退化为标准 BSHD 模式
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        d_ff: int = 3072,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
        use_remove_padding: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.use_remove_padding = use_remove_padding

        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 位置编码
        self.pos_embedding = PositionalEncodingTHD(d_model, max_seq_len)
        self.drop_emb = nn.Dropout(dropout)

        # Transformer 层堆叠
        self.layers = nn.ModuleList(
            [
                GPTTransformerBlockTHD(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ]
        )

        # 最终层归一化
        self.ln_f = nn.LayerNorm(d_model)

        # 输出层
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward_thd(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        THD Packing 模式的前向传播

        数据流:
          input_ids [bsz, seqlen] → embed → unpad → [nnz, d_model]
          → pos_embed → transformer layers (THD) → pad → [bsz, seqlen, d_model]
          → head → logits [bsz, seqlen, vocab_size]

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]，1=有效token, 0=padding

        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape

        # ---- Step 1: 词嵌入 (BSHD) ----
        x = self.embedding(input_ids)  # [bsz, seqlen, d_model]

        # ---- Step 2: Unpack (BSHD → THD) ----
        x_thd, indices, params = unpad_input(x, attention_mask)
        # x_thd: [total_nnz, d_model]

        # ---- Step 3: 位置编码 (THD) ----
        x_thd = self.pos_embedding(x_thd, params)
        x_thd = self.drop_emb(x_thd)

        # ---- Step 4: Transformer 层 (THD) ----
        for layer in self.layers:
            x_thd = layer(x_thd, params)

        # ---- Step 5: 层归一化 (THD) ----
        x_thd = self.ln_f(x_thd)

        # ---- Step 6: Pad 恢复 (THD → BSHD) ----
        x = pad_input(x_thd, indices, batch_size, seq_len)
        # x: [bsz, seqlen, d_model]

        # ---- Step 7: 输出投影 (BSHD) ----
        logits = self.head(x)
        # logits: [bsz, seqlen, vocab_size]

        return logits

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播入口（自动选择 THD 或 BSHD 模式）

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]，可选。
                            如果 use_remove_padding=True 则必须提供；
                            如果未提供且 use_remove_padding=False，则自动生成全1 mask。

        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        if self.use_remove_padding:
            if attention_mask is None:
                # 默认全为有效 token（无 padding）
                attention_mask = torch.ones_like(input_ids)
            return self.forward_thd(input_ids, attention_mask)
        else:
            # BSHD fallback 模式
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            x = self.embedding(input_ids)
            x = self.pos_embedding(x, _params_from_mask(attention_mask))
            x = self.drop_emb(x)
            for layer in self.layers:
                x = layer(x, _params_from_mask(attention_mask))
            x = self.ln_f(x)
            logits = self.head(x)
            return logits


def _params_from_mask(attention_mask: torch.Tensor) -> PackedSeqParams:
    """从 attention mask 快速构建 PackedSeqParams（用于 BSHD fallback）"""
    seqlens, cu_seqlens = compute_cu_seqlens(attention_mask)
    return PackedSeqParams(
        cu_seqlens=cu_seqlens,
        max_seqlen=int(seqlens.max().item()),
        total_nnz=int(seqlens.sum().item()),
        batch_size=attention_mask.shape[0],
    )


if __name__ == "__main__":
    print("=" * 60)
    print("THD GPT Transformer 完整模型测试")
    print("=" * 60)

    # ---- 模型配置 ----
    vocab_size = 1000
    d_model = 128
    n_heads = 4
    n_layers = 2
    d_ff = 512
    batch_size = 3
    seq_len = 10

    # 构造变长输入
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.tensor([
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],   # 样本0: 长度 7
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],   # 样本1: 长度 9
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],   # 样本2: 长度 4
    ])

    print(f"\n输入形状: {input_ids.shape}")
    print(f"有效长度: {attention_mask.sum(dim=-1).tolist()}")
    print(f"总有效 token 数: {attention_mask.sum().item()}")

    # ---- 创建 THD 模型 ----
    model = GPTTransformerTHD(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_len=seq_len,
        use_remove_padding=True,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {total_params:,}")

    # ---- 前向传播 ----
    logits = model(input_ids, attention_mask)
    print(f"输出 logits 形状: {logits.shape}")
    assert logits.shape == (batch_size, seq_len, vocab_size), \
        f"logits 形状错误: {logits.shape}"

    # ---- 验证 padding 区域输出 ----
    print("\n--- Padding 区域输出验证 ---")
    seqlens, _ = compute_cu_seqlens(attention_mask)
    for i in range(batch_size):
        valid_len = int(seqlens[i].item())
        padding_logits = logits[i, valid_len:]
        # padding 位置经过 pad_input 后应该是零向量经过 head 投影的结果
        # 由于 pad_input 填充的是 0，ln_f(0) 非零，所以 padding 区域不一定为零
        # 但应该是有意义的有限值
        is_finite = torch.isfinite(padding_logits).all().item()
        print(f"  样本 {i}: 有效长度={valid_len}, "
              f"padding 区域 finite={is_finite}")

    # ---- 对比：无 padding 场景（全有效 token）----
    print("\n--- 全有效 token 场景测试 ---")
    full_mask = torch.ones_like(input_ids)
    logits_full = model(input_ids, full_mask)
    print(f"全有效 token 输出形状: {logits_full.shape}")
    assert logits_full.shape == (batch_size, seq_len, vocab_size)

    # ---- 效率对比信息 ----
    bsz_total = batch_size * seq_len
    thd_total = int(attention_mask.sum().item())
    savings = (1 - thd_total / bsz_total) * 100
    print(f"\n--- 效率分析 ---")
    print(f"  BSHD 总 token 数: {bsz_total}")
    print(f"  THD 有效 token 数: {thd_total}")
    print(f"  节省计算量: {savings:.1f}%")

    print("\n✅ THD GPT Transformer 所有测试通过！")

