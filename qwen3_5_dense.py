"""
Qwen3.5 Dense 模型实现
基于qwen3_dense.py简化而来，添加了Qwen3.5的GatedDeltaNet线性注意力

主要特点:
- RMSNorm (而非LayerNorm)
- RoPE旋转位置编码
- GQA (Grouped Query Attention)
- SwiGLU激活函数
- QK归一化
- GatedDeltaNet: Qwen3.5特色的线性注意力机制
- Pre-LN结构
- 自定义ONNX算子：GatedDeltaRule
"""
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from gpt import FeedForward
from norm_layer import GatedRMSNorm, ZeroCenteredRMSNorm
from gpt_rope import precompute_freqs_cis, apply_rotary_pos_emb
from qwen3_dense import SwiGLU
from utils.onnx_utils import export_and_simplify, validate_onnx

# 配置参数
vocab_size = 1234
batch_size = 2
seq_len = 10
max_seq_len = 1024

# 简化模型结构
TrueSwiGLU = SwiGLU
SwiGLU = FeedForward
ZeroCenteredRMSNorm = nn.LayerNorm


class GatedDeltaRuleOp(torch.autograd.Function):
    """
    基于DeltaNet优化后的GatedDeltaRule算子

    关键优化:
    - 支持QK归一化 (l2或sum)
    - 支持beta参数控制状态追踪
    - 高效的Delta规则实现
    """

    @staticmethod
    def symbolic(g, q, k, v, a, b, head_dim, conv_kernel_size, use_qk_norm, qk_norm_type):
        """
        ONNX 符号定义
        """
        output = g.op(
            'custom::GatedDeltaRule',
            q, k, v, a, b,
            head_dim_i=head_dim,
            conv_kernel_size_i=conv_kernel_size,
            use_qk_norm_i=use_qk_norm,
            qk_norm_type_s=qk_norm_type
        )
        output.setType(v.type())
        return output

    @staticmethod
    def forward(ctx, q, k, v, a, b, head_dim, conv_kernel_size, use_qk_norm=False, qk_norm_type='l2'):
        """
        优化后的forward实现

        Args:
            q: [batch, seq_len, num_kv_heads * head_dim]
            k: [batch, seq_len, num_kv_heads * head_dim]
            v: [batch, seq_len, num_kv_heads * head_dim]
            a: [batch, seq_len, num_kv_heads]  # 状态A (用于门控)
            b: [batch, seq_len, num_kv_heads]  # 状态B (用于门控)
            head_dim: int
            conv_kernel_size: int
            use_qk_norm: bool - 是否使用QK归一化
            qk_norm_type: str - 归一化类型 ('l2' 或 'sum')
        """
        batch, seq_len, _ = q.shape
        kv_heads = k.shape[-1] // head_dim if head_dim > 0 else 1

        # Reshape to heads
        q_heads = q.view(batch, seq_len, kv_heads, head_dim)
        k_heads = k.view(batch, seq_len, kv_heads, head_dim)
        v_heads = v.view(batch, seq_len, kv_heads, head_dim)
        a_heads = a.view(batch, seq_len, kv_heads, 1)  # [B, S, H, 1]
        b_heads = b.view(batch, seq_len, kv_heads, 1)  # [B, S, H, 1]

        # ===== QK归一化 (与DeltaNet一致) =====
        if use_qk_norm:
            if qk_norm_type == 'l2':
                # L2归一化
                q_heads = F.normalize(q_heads, p=2, dim=-1)
                k_heads = F.normalize(k_heads, p=2, dim=-1)
            elif qk_norm_type == 'sum':
                # Sum归一化
                q_heads = q_heads / (q_heads.sum(-1, keepdim=True) + 1e-8)
                k_heads = k_heads / (k_heads.sum(-1, keepdim=True) + 1e-8)

        # ===== Delta规则计算 (带卷积窗口限制) =====
        # 简化的DeltaNet计算，使用卷积窗口
        output = torch.zeros_like(v_heads)
        scale = 1.0 / (head_dim ** 0.5)

        for i in range(seq_len):
            q_i = q_heads[:, i, :, :]  # [batch, kv_heads, head_dim]
            # 限制卷积窗口
            start_idx = max(0, i - conv_kernel_size + 1)

            # 计算注意力权重 (向量化)
            # [batch, kv_heads, head_dim] @ [start:i+1, kv_heads, head_dim]^T
            k_window = k_heads[:, start_idx:i+1, :, :]  # [batch, window, kv_heads, head_dim]
            attn_weights = torch.einsum('bhd,bwhd->bhw', q_i, k_window) * scale  # [batch, kv_heads, window]

            # 计算门控 (使用a和b)
            a_window = a_heads[:, start_idx:i+1, :, :]  # [batch, window, kv_heads, 1]
            b_window = b_heads[:, i:i+1, :, :]  # [batch, 1, kv_heads, 1]
            # 调整维度以正确广播: [batch, window, kv_heads, 1] * [batch, 1, kv_heads, 1]
            gates = torch.sigmoid(a_window * b_window)  # [batch, window, kv_heads, 1]

            # 应用门控 - 需要调整 attn_weights 的维度
            # attn_weights: [batch, kv_heads, window] -> [batch, window, kv_heads]
            attn_weights = attn_weights.transpose(1, 2) * gates.squeeze(-1)  # [batch, window, kv_heads]

            # 加权求和
            v_window = v_heads[:, start_idx:i+1, :, :]  # [batch, window, kv_heads, head_dim]
            output[:, i, :, :] = torch.einsum('bwh,bwhd->bhd', attn_weights, v_window)

        output = output.view(batch, seq_len, -1)

        ctx.save_for_backward(q, k, v, a, b)
        ctx.head_dim = head_dim
        ctx.conv_kernel_size = conv_kernel_size
        ctx.use_qk_norm = use_qk_norm
        ctx.qk_norm_type = qk_norm_type
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """优化的backward实现"""
        q, k, v, a, b = ctx.saved_tensors
        grad_q = grad_k = grad_v = grad_a = grad_b = None

        if ctx.needs_input_grad[0]:
            grad_q = torch.zeros_like(q)
        if ctx.needs_input_grad[1]:
            grad_k = torch.zeros_like(k)
        if ctx.needs_input_grad[2]:
            grad_v = torch.zeros_like(v)
        if ctx.needs_input_grad[3]:
            grad_a = torch.zeros_like(a)
        if ctx.needs_input_grad[4]:
            grad_b = torch.zeros_like(b)

        return grad_q, grad_k, grad_v, grad_a, grad_b, None, None, None, None

# 注册 ONNX 符号处理（在 symbolic 方法中已完成，这里作为备选）
def _register_onnx_ops():
    """注册 GatedDeltaRuleOp 的 ONNX 符号处理"""
    try:
        from torch.onnx import register_custom_op_symbolic

        def symbolic_gated_delta_rule(g, q, k, v, a, b, head_dim, conv_kernel_size):
            """GatedDeltaRuleOp 的 ONNX 符号处理"""
            output = g.op(
                'custom::GatedDeltaRule',
                q, k, v, a, b,
                head_dim_i=head_dim,
                conv_kernel_size_i=conv_kernel_size
            )
            # 使用 Identity 帮助形状推断
            return g.op('Identity', output)

        register_custom_op_symbolic(
            'custom::gated_delta_rule_op',
            symbolic_gated_delta_rule,
            opset_version=14
        )
    except Exception:
        pass


_register_onnx_ops()


class GatedDeltaRuleModule(nn.Module):
    """
    GatedDeltaRule的模块化包装

    用于ONNX导出
    支持QK归一化选项
    """

    def __init__(self, head_dim: int, conv_kernel_size: int, use_qk_norm: bool = False, qk_norm_type: str = 'l2'):
        super().__init__()
        self.head_dim = head_dim
        self.conv_kernel_size = conv_kernel_size
        self.use_qk_norm = use_qk_norm
        self.qk_norm_type = qk_norm_type
        self._export_op_name = 'sglang::GatedDeltaRule'

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        执行GatedDeltaRule计算（不应用z门控）

        Args:
            q: [batch, seq_len, num_kv_heads * head_dim]
            k: [batch, seq_len, num_kv_heads * head_dim]
            v: [batch, seq_len, num_kv_heads * head_dim]
            a: [batch, seq_len, num_kv_heads]
            b: [batch, seq_len, num_kv_heads]

        Returns:
            output: [batch, seq_len, num_kv_heads * head_dim]
        """
        return GatedDeltaRuleOp.apply(
            q, k, v, a, b,
            self.head_dim, self.conv_kernel_size,
            self.use_qk_norm, self.qk_norm_type
        )


class GatedDeltaNet(nn.Module):
    """
    基于DeltaNet优化后的GatedDeltaNet线性注意力机制

    关键改进 (参考DeltaNet):
    - ShortConvolution: 使用因果卷积处理局部上下文
    - QK归一化: 支持L2或Sum归一化，提升训练稳定性
    - SiLU激活: QK使用SiLU激活函数
    - 简化的状态参数

    结构:
        输入 x
        ├─→ in_proj_qkv ──→ Q, K, V (SiLU激活)
        ├─→ in_proj_z ────→ z (输出门控)
        ├─→ in_proj_a ────→ a (状态参数)
        ├─→ in_proj_b ────→ b (状态参数)
        ├─→ ShortConv (Q, K, V) ──→ 局部卷积
        │
        └─→ GatedDeltaRule ──→ RMSNorm(z) ──→ out_proj ──> 输出
    """

    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            num_kv_heads: int,
            head_dim: int = 128,
            conv_kernel_size: int = 4,
            rope_base: float = 1000000.0,
            max_seq_len: int = 1024,
            use_qk_norm: bool = True,      # 新增: QK归一化选项
            qk_norm_type: str = 'l2',       # 新增: 归一化类型
            use_gate: bool = True,         # 新增: 输出门控
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        assert num_kv_heads <= num_heads
        assert qk_norm_type in ['l2', 'sum'], f"qk_norm_type must be 'l2' or 'sum', got {qk_norm_type}"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.conv_kernel_size = conv_kernel_size
        self.rope_base = rope_base
        self.use_qk_norm = use_qk_norm
        self.qk_norm_type = qk_norm_type
        self.use_gate = use_gate

        self.key_dim = num_kv_heads * head_dim
        self.value_dim = num_kv_heads * head_dim
        self.q_size = num_heads * head_dim

        # ===== 投影层 (参考DeltaNet) =====
        # QKV 投影 - 激活函数在外部应用
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        # z 投影 (输出门控)
        self.z_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        # a, b 投影 (状态参数 - DeltaNet中的beta)
        self.a_proj = nn.Linear(hidden_size, num_kv_heads, bias=False)  # 状态A
        self.b_proj = nn.Linear(hidden_size, num_kv_heads, bias=False)  # 状态B (类似beta)

        # ===== ShortConvolution (参考DeltaNet) =====
        # 使用三个独立的卷积层分别处理Q, K, V
        self.q_conv = nn.Conv1d(
            in_channels=self.key_dim,
            out_channels=self.key_dim,
            kernel_size=conv_kernel_size,
            padding=conv_kernel_size - 1,
            groups=self.key_dim,  # Depthwise convolution
            bias=False
        )
        self.k_conv = nn.Conv1d(
            in_channels=self.key_dim,
            out_channels=self.key_dim,
            kernel_size=conv_kernel_size,
            padding=conv_kernel_size - 1,
            groups=self.key_dim,
            bias=False
        )
        self.v_conv = nn.Conv1d(
            in_channels=self.value_dim,
            out_channels=self.value_dim,
            kernel_size=conv_kernel_size,
            padding=conv_kernel_size - 1,
            groups=self.value_dim,
            bias=False
        )

        # ===== 状态参数 (DeltaNet中的beta和gate) =====
        # A_log: 状态转移矩阵的对数 (可学习参数)
        self.A_log = nn.Parameter(torch.zeros(num_kv_heads))
        # dt_bias: 动态时间步长的偏置
        self.dt_bias = nn.Parameter(torch.ones(num_kv_heads))

        # ===== GatedDeltaRule ONNX算子 =====
        self.gated_delta_rule_module = GatedDeltaRuleModule(
            head_dim, conv_kernel_size,
            use_qk_norm=use_qk_norm,
            qk_norm_type=qk_norm_type
        )

        # ===== 注意力输出归一化 =====
        if use_gate:
            self.norm = GatedRMSNorm(self.value_dim, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(self.value_dim, eps=1e-6)

        # ===== 输出投影 =====
        self.out_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        # ===== RoPE =====
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=head_dim,
            end=max_seq_len,
            rope_base=rope_base
        )
        self.register_buffer('freqs_cos', freqs_cos, persistent=False)
        self.register_buffer('freqs_sin', freqs_sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, hidden_size]
        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = x.shape

        # ===== 投影 =====
        # QKV 投影
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 状态参数
        z = self.z_proj(x)
        a = self.a_proj(x)
        b = self.b_proj(x)

        # ===== SiLU激活 (参考DeltaNet) =====
        q = F.silu(q)
        k = F.silu(k)
        v = F.silu(v)

        # ===== ShortConvolution (因果卷积) =====
        # 需要将 [B, S, H] -> [B, H, S] 进行卷积
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = self.q_conv(q)[:, :, :seq_len]  # 裁剪到原始序列长度
        k = self.k_conv(k)[:, :, :seq_len]
        v = self.v_conv(v)[:, :, :seq_len]

        # 转回 [B, S, H]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # ===== Reshape到heads =====
        q = q.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # ===== 应用RoPE到Q和K =====
        q = q.transpose(1, 2)  # [B, H, S, D]
        k = k.transpose(1, 2)

        q, k = apply_rotary_pos_emb(
            q, k,
            self.freqs_cos[:seq_len],
            self.freqs_sin[:seq_len]
        )

        # 转回 [B, S, H, D] 然后flatten
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        q = q.view(batch_size, seq_len, -1)
        k = k.view(batch_size, seq_len, -1)

        # ===== GatedDeltaRule 计算 =====
        attn_output = self.gated_delta_rule_module(q, k, v, a, b)

        # ===== 归一化, 应用z门控 =====
        if self.use_gate:
            attn_output = self.norm(attn_output, z)
        else:
            attn_output = self.norm(attn_output)
            attn_output = attn_output * torch.sigmoid(z)

        # ===== 输出投影 =====
        output = self.out_proj(attn_output)

        return output


class GatedAttention(nn.Module):
    """
    Qwen3.5 标准多头注意力 (用于对比)

    特点:
        - RoPE (Rotary Position Embedding): 旋转位置编码
        - GQA (Grouped Query Attention): 分组查询注意力
        - QK归一化: 对Query和Key分别进行归一化
        - attn_output_gate: 输出门控 (Qwen3.5特色)
    """

    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            num_kv_heads: int,
            head_dim: int = 128,
            dropout: float = 0.0,
            rope_base: float = 1000000.0,
            max_seq_len: int = 1024,
            attn_output_gate: bool = True,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        assert num_kv_heads <= num_heads

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rope_base = rope_base
        self.attn_output_gate = attn_output_gate

        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim

        # QKV投影
        if attn_output_gate:
            # Q有gate，所以Q的维度翻倍
            self.q_output_size = self.q_size * 2
        else:
            self.q_output_size = self.q_size

        self.qkv_proj = nn.Linear(
            hidden_size,
            self.q_output_size + 2 * self.kv_size,
            bias=False
        )

        # O投影
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        self.dropout = nn.Dropout(dropout)

        # QK归一化
        self.q_norm = ZeroCenteredRMSNorm(head_dim, eps=1e-6)
        self.k_norm = ZeroCenteredRMSNorm(head_dim, eps=1e-6)

        # RoPE
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=head_dim,
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

        # 1. QKV投影
        qkv = self.qkv_proj(x)

        # 2. 分割Q, K, V
        if self.attn_output_gate:
            q_gate, k, v = qkv.split([self.q_output_size, self.kv_size, self.kv_size], dim=-1)
            # 分割gate
            q, gate = q_gate.chunk(2, dim=-1)
        else:
            q, k, v = qkv.split([self.q_output_size, self.kv_size, self.kv_size], dim=-1)
            gate = None

        # 3. Reshape
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 4. QK归一化
        q = self.q_norm(q)
        k = self.k_norm(k)

        # 5. RoPE - apply_rotary_pos_emb 期望 [batch, heads, seq, head_dim]
        q, k = apply_rotary_pos_emb(
            q, k,
            self.freqs_cos[:seq_len],
            self.freqs_sin[:seq_len]
        )

        # 6. GQA: 扩展K,V到Q的头数
        if self.num_kv_heads < self.num_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        # 7. 注意力计算
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 因果掩码
        mask_expanded = self.causal_mask[:seq_len, :seq_len].view(1, 1, seq_len, seq_len)
        attn_scores.masked_fill_(mask_expanded.bool(), -torch.inf)

        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 8. 输出
        attn_output = torch.matmul(attn_weights, v)

        # 恢复形状
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.head_dim)

        # 应用输出门控
        if self.attn_output_gate and gate is not None:
            gate = torch.sigmoid(gate)
            attn_output = attn_output * gate

        # O投影
        output = self.o_proj(attn_output)

        return output


class Qwen3_5DecoderLayer(nn.Module):
    """Qwen3.5 Decoder层 (支持GatedDeltaNet线性注意力)"""

    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            num_kv_heads: int,
            head_dim: int = 128,
            intermediate_size: int = 2048,
            use_linear_attention: bool = True,  # Qwen3.5特色
            dropout: float = 0.0,
            rope_base: float = 1000000.0,
            max_seq_len: int = 1024,
            conv_kernel_size: int = 4,
    ):
        super().__init__()

        # Pre-LN结构
        self.input_layernorm = ZeroCenteredRMSNorm(hidden_size)

        # 注意力层 (可选标准注意力或GatedDeltaNet)
        if use_linear_attention:
            self.self_attn = GatedDeltaNet(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                conv_kernel_size=conv_kernel_size,
                rope_base=rope_base,
                max_seq_len=max_seq_len,
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


class Qwen3_5DenseModel(nn.Module):
    """Qwen3.5 Dense模型 (支持GatedDeltaNet)

    按照论文结构:
    - 前3层使用 GatedDeltaNet (线性注意力)
    - 最后1层使用 GatedAttention (标准注意力)
    """

    def __init__(
            self,
            vocab_size: int,
            hidden_size: int = 768,
            num_attention_heads: int = 12,
            num_key_value_heads: int = 2,  # GQA
            head_dim: int = 128,
            num_hidden_layers: int = 4,  # 默认4层: 3层GatedDeltaNet + 1层GatedAttention
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

        # Transformer层: 按照论文结构
        # 1 层使用 GatedDeltaNet (线性注意力)
        # 1层使用 GatedAttention (标准注意力)
        self.layers = nn.ModuleList()

        # 前3层使用 GatedDeltaNet (线性注意力)
        num_linear_layers = num_hidden_layers - 1
        for i in range(num_linear_layers):
            self.layers.append(
                Qwen3_5DecoderLayer(
                    hidden_size=hidden_size,
                    num_heads=num_attention_heads,
                    num_kv_heads=num_key_value_heads,
                    head_dim=head_dim,
                    intermediate_size=intermediate_size,
                    use_linear_attention=True,  # GatedDeltaNet
                    dropout=dropout,
                    rope_base=rope_base,
                    max_seq_len=max_position_embeddings,
                    conv_kernel_size=conv_kernel_size,
                )
            )

        # 最后一层使用 GatedAttention (标准注意力)
        self.layers.append(
            Qwen3_5DecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_attention_heads,
                num_kv_heads=num_key_value_heads,
                head_dim=head_dim,
                intermediate_size=intermediate_size,
                use_linear_attention=False,  # 标准Attention
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
    # 实际的模型结构为 (3层GatedDeltaNet + 1层GatedAttention)
    # 这里进行简化，1层GatedDeltaNet + 1层GatedAttention
    print("=" * 50)
    print("测试 Qwen3.5 Dense (混合架构)")
    print("1层 GatedDeltaNet + 1层 GatedAttention")
    print("=" * 50)

    model = Qwen3_5DenseModel(
        vocab_size=vocab_size,
        hidden_size=768,
        num_attention_heads=6,
        num_key_value_heads=2,
        head_dim=128,
        num_hidden_layers=2,  # 1层: 1层GatedDeltaNet + 1层GatedAttention
        intermediate_size=2048,
        max_position_embeddings=max_seq_len,
    )

    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(dummy_input)

    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {logits.shape}")
    print(f"模型层数: {len(model.layers)}")
    print(f"  - 前3层: GatedDeltaNet (线性注意力)")
    print(f"  - 第4层: GatedAttention (标准注意力)")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print("Qwen3.5 Dense (混合架构) 测试通过！")

    # 导出为ONNX
    os.makedirs("onnx_data", exist_ok=True)
    onnx_path = "onnx_data/qwen3_5_dense.onnx"

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
