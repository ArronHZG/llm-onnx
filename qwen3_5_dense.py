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
from gpt_rope import precompute_freqs_cis, apply_rotary_pos_emb
from norm_layer import GatedRMSNorm, ZeroCenteredRMSNorm
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

    参考 sglang 实现:
    - fused_gdn_gating: Mamba-style gate 计算
    - chunk_gated_delta_rule: 高效的chunk模式计算
    """

    @staticmethod
    def symbolic(g, q, k, v, a, b, A_log, dt_bias, head_dim, conv_kernel_size):
        """
        ONNX 符号定义
        """
        output = g.op(
            'custom::GatedDeltaRule',
            q, k, v, a, b, A_log, dt_bias,
            head_dim_i=head_dim,
            conv_kernel_size_i=conv_kernel_size,
        )
        output.setType(v.type())
        return output

    @staticmethod
    def forward(ctx, q, k, v, a, b, A_log, dt_bias, head_dim, conv_kernel_size):
        """
        GatedDeltaRule forward 实现

        参考 sglang: fused_gdn_gating

        输入形状 (head_first=False):
            q: [batch, seq_len, num_heads, head_dim]
            k: [batch, seq_len, num_heads, head_dim]
            v: [batch, seq_len, num_heads, head_dim]
            a: [batch, seq_len, num_heads]  # 门控输入
            b: [batch, seq_len, num_heads]  # beta 输入
            A_log: [num_heads]  # 状态转移矩阵对数
            dt_bias: [num_heads]  # 动态时间步长偏置

        输出形状:
            output: [batch, seq_len, num_heads, head_dim]
        """
        batch, seq_len, num_heads, head_dim = q.shape

        # ===== QK L2 归一化 =====
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        # ===== Mamba-style gate 计算 (参考 fused_gdn_gating) =====
        # g = -A_log.exp() * softplus(a + dt_bias)
        # beta = sigmoid(b)
        a_expanded = a.unsqueeze(-1)  # [B, S, H, 1]
        A_expanded = A_log.view(1, 1, num_heads, 1)  # [1, 1, H, 1]
        dt_bias_expanded = dt_bias.view(1, 1, num_heads, 1)  # [1, 1, H, 1]

        # softplus(a + dt_bias)
        x = a_expanded + dt_bias_expanded
        softplus_x = F.softplus(x)

        # g = -A_exp * softplus(a + dt_bias)
        g = -A_expanded.exp() * softplus_x  # [B, S, H, 1]
        g = g.squeeze(-1)  # [B, S, H]

        # beta = sigmoid(b)
        beta = torch.sigmoid(b)  # [B, S, H]

        # ===== GatedDeltaRule 计算 =====
        scale = 1.0 / (head_dim ** 0.5)
        output = torch.zeros_like(v)

        # 初始化状态 S: [batch, num_heads, head_dim, head_dim]
        S = torch.zeros(batch, num_heads, head_dim, head_dim, dtype=q.dtype, device=q.device)

        for i in range(seq_len):
            q_i = q[:, i, :, :]      # [B, H, D]
            k_i = k[:, i, :, :]      # [B, H, D]
            v_i = v[:, i, :, :]      # [B, H, D]
            g_i = g[:, i, :]          # [B, H]
            beta_i = beta[:, i, :]    # [B, H]

            # 扩展维度用于状态计算
            g_i = g_i.unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]
            beta_i = beta_i.unsqueeze(-1)           # [B, H, 1]

            # 状态更新: S = g * S + k * v
            kv_term = k_i.unsqueeze(-1) * v_i.unsqueeze(-2)  # [B, H, D, D]
            S = g_i * S + kv_term

            # 输出: o = beta * (q @ S) * scale
            o_i = torch.einsum('bhd,bhdm->bhm', q_i, S)  # [B, H, D]
            o_i = o_i * beta_i * scale

            output[:, i, :, :] = o_i

        # 保存用于 backward
        ctx.save_for_backward(q, k, v, g, beta, A_log, dt_bias)
        ctx.head_dim = head_dim
        ctx.conv_kernel_size = conv_kernel_size
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """简化的梯度实现"""
        q, k, v, g, beta, A_log, dt_bias = ctx.saved_tensors
        head_dim = ctx.head_dim

        batch, seq_len, num_heads, _ = q.shape
        scale = 1.0 / (head_dim ** 0.5)

        # 初始化梯度
        grad_q = torch.zeros_like(q)
        grad_k = torch.zeros_like(k)
        grad_v = torch.zeros_like(v)
        grad_a = torch.zeros_like(grad_output)
        grad_b = torch.zeros_like(grad_output)
        grad_A_log = torch.zeros_like(A_log)
        grad_dt_bias = torch.zeros_like(dt_bias)

        # 反向计算
        S = torch.zeros(batch, num_heads, head_dim, head_dim, dtype=q.dtype, device=q.device)

        for i in range(seq_len - 1, -1, -1):
            q_i = q[:, i, :, :]
            k_i = k[:, i, :, :]
            v_i = v[:, i, :, :]
            g_i = g[:, i, :].unsqueeze(-1).unsqueeze(-1)  # [B, H] -> [B, H, 1, 1]
            beta_i = beta[:, i, :].unsqueeze(-1)  # [B, H] -> [B, H, 1]
            do_i = grad_output[:, i, :, :]

            # 状态更新
            kv_term = k_i.unsqueeze(-1) * v_i.unsqueeze(-2)
            S = g_i * S + kv_term

            # 梯度计算
            grad_q[:, i, :, :] = torch.einsum('bhd,bhdm->bhm', do_i * beta_i * scale, S.transpose(-2, -1))
            # 简化 grad_v: grad_v = do * beta * scale * q (忽略 S)
            grad_v[:, i, :, :] = do_i * beta_i * scale * q_i

        # 简化的 a, b 梯度 (使用 grad_output 的统计信息)
        # a, b 的形状是 [B, S, H]，所以 grad_a, grad_b 也应该是 [B, S, H]
        grad_a = grad_output.sum(dim=3) * 0.01  # [B, S, H, D] -> [B, S, H]
        grad_b = grad_output.sum(dim=3) * 0.01  # [B, S, H, D] -> [B, S, H]

        # 注意: grad_q, grad_k, grad_v 已经是 [B, S, H, D] 形状，不需要 reshape

        return grad_q, grad_k, grad_v, grad_a, grad_b, grad_A_log, grad_dt_bias, None, None


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
            output.setType(q.type().with_sizes(q.sizes()))
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
    GatedDeltaRule 的模块化包装

    用于 ONNX 导出
    参考 sglang: fused_gdn_gating + chunk_gated_delta_rule
    """

    def __init__(self, head_dim: int, conv_kernel_size: int, num_heads: int):
        super().__init__()
        self.head_dim = head_dim
        self.conv_kernel_size = conv_kernel_size
        self.num_heads = num_heads
        self._export_op_name = 'sglang::GatedDeltaRule'

        # ===== Mamba-style gate 参数 =====
        # A_log: 状态转移矩阵对数 [num_heads]
        # dt_bias: 动态时间步长偏置 [num_heads]
        A = torch.empty(self.num_heads, dtype=torch.float32).uniform_(0, 16)
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.num_heads))
        self.D._no_weight_decay = True
        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(self.num_heads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # self.A_log = nn.Parameter(torch.zeros(num_heads))
        # self.dt_bias = nn.Parameter(torch.zeros(num_heads))

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        执行 GatedDeltaRule 计算

        Args:
            q: [batch, seq_len, num_heads, head_dim]
            k: [batch, seq_len, num_heads, head_dim]
            v: [batch, seq_len, num_heads, head_dim]
            a: [batch, seq_len, num_heads] - gate
            b: [batch, seq_len, num_heads] - beta
            A_log: [num_heads] - Mamba-style A_log 参数
            dt_bias: [num_heads] - Mamba-style dt_bias 参数

        Returns:
            output: [batch, seq_len, num_heads, head_dim]
        """
        return GatedDeltaRuleOp.apply(
            q, k, v, a, b, self.A_log, self.dt_bias,
            self.head_dim, self.conv_kernel_size
        )


class GatedDeltaNet(nn.Module):
    """
    基于DeltaNet优化后的GatedDeltaNet线性注意力机制

    关键改进 (参考NVlabs GatedDeltaNet):
    - ShortConvolution: 使用因果卷积处理局部上下文
    - QK归一化: 支持L2或Sum归一化，提升训练稳定性
    - SiLU激活: QK使用SiLU激活函数
    - Mamba-style gate: 使用A_log, dt_bias进行门控
    - 简化的状态参数

    结构:
        输入 x
        ├─→ in_proj_qkv ──→ Q, K, V (SiLU激活)
        ├─→ in_proj_z ────→ z (输出门控)
        ├─→ in_proj_gk ──→ gk (Mamba-style gate)
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
            use_gate: bool = True,  # 输出门控
            use_mamba_gate: bool = True,  # Mamba-style gate
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        assert num_kv_heads <= num_heads

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.conv_kernel_size = conv_kernel_size
        self.rope_base = rope_base
        self.use_gate = use_gate
        self.use_mamba_gate = use_mamba_gate

        self.key_dim = num_heads * head_dim
        self.value_dim = num_kv_heads * head_dim
        self.q_size = num_heads * head_dim

        # ===== 投影层 (参考DeltaNet) =====
        # QKV 投影 - 使用统一的维度
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        # z 投影 (输出门控)
        self.z_proj = nn.Linear(hidden_size, self.key_dim, bias=False)

        # a 投影 (状态参数)
        self.a_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # b 投影 (状态参数)
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)

        # ===== 简化的 ShortConvolution =====
        # Q: num_heads * head_dim
        self.q_conv = nn.Conv1d(
            in_channels=self.key_dim,
            out_channels=self.key_dim,
            kernel_size=conv_kernel_size,
            padding=conv_kernel_size - 1,
            groups=self.key_dim,
            bias=False
        )
        # K, V: num_kv_heads * head_dim
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

        # ===== GatedDeltaRule ONNX算子 =====
        self.gated_delta_rule_module = GatedDeltaRuleModule(
            head_dim, conv_kernel_size, num_heads,
        )

        # ===== 注意力输出归一化 =====
        # 使用 num_heads * head_dim 作为 norm 的输入维度
        norm_input_dim = num_heads * head_dim
        self.norm = GatedRMSNorm(norm_input_dim, eps=1e-6)

        # ===== 输出投影 =====
        # 需要将 num_heads * head_dim 映射到 hidden_size
        # 由于 out_proj 期望 value_dim (num_kv_heads * head_dim)，需要调整
        out_proj_input_dim = self.num_heads * self.head_dim
        self.out_proj = nn.Linear(out_proj_input_dim, hidden_size, bias=False)

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

        # 门控参数
        z = self.z_proj(x)
        a = self.a_proj(x)
        b = self.b_proj(x)

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

        # ===== SiLU激活 =====
        # 用relu替换 silu 简化模型
        q = F.relu(q)
        k = F.relu(k)
        v = F.relu(v)

        # ===== Reshape到heads =====
        # Q使用num_heads, KV使用num_kv_heads
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # ===== 应用RoPE到Q和K (在GQA之前) =====
        q = q.transpose(1, 2)  # [B, H, S, D]
        k = k.transpose(1, 2)

        q, k = apply_rotary_pos_emb(
            q, k,
            self.freqs_cos[:seq_len],
            self.freqs_sin[:seq_len]
        )

        # 转回 [B, S, H, D]
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()

        # ===== GQA: 扩展 K,V 到 Q 的头数 =====
        if self.num_kv_heads < self.num_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            v = v.repeat_interleave(repeat_factor, dim=2)

        # ===== Flatten 用于 GatedDeltaRule =====
        # q = q.view(batch_size, seq_len, -1)
        # k = k.view(batch_size, seq_len, -1)
        # v = v.view(batch_size, seq_len, -1)

        # ===== GatedDeltaRule 计算 =====
        # a 和 b 已经是 [B, S, num_heads] 形状，不需要 transpose
        # A_log 和 dt_bias 是 Mamba-style 门控参数
        attn_output = self.gated_delta_rule_module(q, k, v, a, b)

        # ===== Reshape 输出 (flatten 到 num_heads * head_dim) =====
        attn_output = attn_output.view(batch_size, seq_len, -1)  # [B, S, num_heads * head_dim]
        print(f"attn_output: {attn_output.shape}")

        # ===== 归一化, 应用z门控 =====
        # z 需要扩展到与 attn_output 相同的形状
        attn_output = self.norm(attn_output, z)

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
