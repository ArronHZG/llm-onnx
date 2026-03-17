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
from gpt_norm import GatedRMSNorm
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
    自定义ONNX算子：GatedDeltaRule

    将GatedDeltaNet的核心计算封装为可导出的ONNX算子

    Args:
        q: [batch, seq_len, num_kv_heads * head_dim]
        k: [batch, seq_len, num_kv_heads * head_dim]
        v: [batch, seq_len, num_kv_heads * head_dim]
        a: [batch, seq_len, num_kv_heads]
        b: [batch, seq_len, num_kv_heads]

    Returns:
        output: [batch, seq_len, num_kv_heads * head_dim]
    """

    @staticmethod
    def symbolic(g, q, k, v, a, b, head_dim, conv_kernel_size):
        """
        ONNX 符号定义 - 直接在 Function 类中定义
        这样 PyTorch 可以正确识别并导出自定义节点
        """
        return g.op(
            'sglang::GatedDeltaRule',
            q, k, v, a, b,
            head_dim_i=head_dim,
            conv_kernel_size_i=conv_kernel_size
        )

    @staticmethod
    def forward(ctx, q, k, v, a, b, head_dim, conv_kernel_size):
        """
        前向传播：GatedDeltaRule的核心计算
        """
        batch, seq_len, _ = q.shape
        kv_heads = k.shape[-1] // head_dim if head_dim > 0 else 1

        # Reshape to heads
        q_heads = q.view(batch, seq_len, kv_heads, head_dim)
        k_heads = k.view(batch, seq_len, kv_heads, head_dim)
        v_heads = v.view(batch, seq_len, kv_heads, head_dim)
        a_heads = a.view(batch, seq_len, kv_heads, 1)
        b_heads = b.view(batch, seq_len, kv_heads, 1)

        # 简化的DeltaNet计算
        output = torch.zeros_like(v_heads)

        for i in range(seq_len):
            # 获取当前位置的query
            q_i = q_heads[:, i, :, :]  # [batch, kv_heads, head_dim]

            # 简化的注意力计算
            if i == 0:
                output[:, i, :, :] = v_heads[:, i, :, :]
            else:
                # 滑动窗口内的注意力
                start_idx = max(0, i - conv_kernel_size)
                for j in range(start_idx, i + 1):
                    attn_weight = torch.sum(q_i * k_heads[:, j, :, :], dim=-1, keepdim=True)
                    attn_weight = attn_weight / math.sqrt(head_dim)

                    # 应用门控 a, b
                    gate = torch.sigmoid(a_heads[:, i, :, :] * b_heads[:, j, :, :])
                    attn_weight = attn_weight * gate

                    output[:, i, :, :] += attn_weight * v_heads[:, j, :, :]

        # 恢复形状
        output = output.view(batch, seq_len, -1)

        # 保存用于反向传播的张量
        ctx.save_for_backward(q, k, v, a, b)
        ctx.head_dim = head_dim
        ctx.conv_kernel_size = conv_kernel_size

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播
        """
        q, k, v, a, b = ctx.saved_tensors
        head_dim = ctx.head_dim
        conv_kernel_size = ctx.conv_kernel_size

        # 简化实现：返回梯度
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

        return grad_q, grad_k, grad_v, grad_a, grad_b, None, None


# ===== ONNX 算子注册 =====
def gated_delta_rule_onnx_symbolic(g, q, k, v, a, b, head_dim, conv_kernel_size):
    """
    GatedDeltaRuleOp 的 ONNX 符号表达

    这个函数将自定义 PyTorch 算子转换为 ONNX 格式
    创建一个自定义操作节点在 ONNX 中
    """
    # 直接创建自定义 ONNX 操作节点，保留在导出的模型中
    # domain 参数指定自定义操作所属的命名空间
    return g.op(
        'sglang::GatedDeltaRule',
        q, k, v, a, b,
        head_dim_i=head_dim,
        conv_kernel_size_i=conv_kernel_size
    )


# 在模块加载时注册 ONNX 符号
def _register_onnx_symbolic():
    """注册 GatedDeltaRuleOp 的 ONNX 符号"""
    try:
        # 对于 PyTorch 1.12+ 版本
        from torch.onnx import register_custom_op_symbolic
        register_custom_op_symbolic(
            'custom::gated_delta_rule_op',
            gated_delta_rule_onnx_symbolic,
            opset_version=14
        )
    except (ImportError, AttributeError):
        try:
            # 对于早期版本，尝试直接注册到 torch._C._jit_get_custom_op_schema
            import torch.onnx.symbolic as sym
            sym.register_custom_op_symbolic(
                'custom::gated_delta_rule_op',
                gated_delta_rule_onnx_symbolic,
                14
            )
        except Exception:
            pass

# 在导入时尝试注册
_register_onnx_symbolic()


class GatedDeltaRuleModule(nn.Module):
    """
    GatedDeltaRule的模块化包装

    用于ONNX导出
    """
    def __init__(self, head_dim: int, conv_kernel_size: int):
        super().__init__()
        self.head_dim = head_dim
        self.conv_kernel_size = conv_kernel_size
        # 设置符号名称供 ONNX 导出使用
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
        return GatedDeltaRuleOp.apply(q, k, v, a, b, self.head_dim, self.conv_kernel_size)


class GatedDeltaNet(nn.Module):
    """
    Qwen3.5 GatedDeltaNet 线性注意力机制

    这是Qwen3.5的核心创新，用线性注意力替代标准注意力:
    - 状态空间: 用Hidden State (A, B) 代替 KV Cache
    - 门控: 使用 sigmoid 门控 (z) 控制信息流（在外部处理）
    - 卷积: 使用卷积核处理离散状态
    - 归一化: 使用RMSNorm进行门控归一化

    结构:
        输入 x
        ├─→ in_proj_qkv ──→ 分割为 Q, K, V
        ├─→ in_proj_z ────→ z (门控，在外面处理)
        ├─→ in_proj_b ────→ b (状态)
        ├─→ in_proj_a ────→ a (状态)
        ├─→ conv1d ──────→ 状态卷积
        │
        └─→ GatedDeltaRule(ONNX算子) ──→ * sigmoid(z) ──→ RMSNorm(z) ──→ out_proj ──> 输出
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

        self.key_dim = num_kv_heads * head_dim
        self.value_dim = num_kv_heads * head_dim
        self.q_size = num_heads * head_dim

        # ===== 投影层 =====
        # QKV 投影 (合并)
        self.in_proj_qkv = nn.Linear(
            hidden_size,
            self.key_dim * 2 + self.key_dim,  # Q + K + V
            bias=False
        )

        # z 投影 (门控)
        self.in_proj_z = nn.Linear(hidden_size, self.value_dim, bias=False)

        # b, a 投影 (状态参数)
        self.in_proj_b = nn.Linear(hidden_size, num_kv_heads, bias=False)
        self.in_proj_a = nn.Linear(hidden_size, num_kv_heads, bias=False)

        # ===== 卷积层 (用于状态空间) =====
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            self.conv_kernel_size,
            self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=1
        )

        # ===== 状态参数 =====
        # A_log: 状态转移矩阵的对数 (可学习参数)
        self.A_log = nn.Parameter(torch.zeros(num_kv_heads))
        # dt_bias: 动态时间步长的偏置
        self.dt_bias = nn.Parameter(torch.ones(num_kv_heads))

        # ===== GatedDeltaRule ONNX算子 =====
        self.gated_delta_rule_module = GatedDeltaRuleModule(head_dim, conv_kernel_size)

        # ===== 注意力输出归一化 =====
        self.norm = GatedRMSNorm(self.value_dim, eps=1e-6)

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

    def gated_delta_rule(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            a: torch.Tensor,
            b: torch.Tensor,
    ) -> torch.Tensor:
        """
        GatedDeltaNet 核心计算

        这是一个简化的实现，用于ONNX导出
        实际sglang版本使用Triton kernel进行高度优化

        Args:
            q: [batch, seq_len, num_heads * head_dim]
            k: [batch, seq_len, num_kv_heads * head_dim]
            v: [batch, seq_len, num_kv_heads * head_dim]
            a: [batch, seq_len, num_kv_heads]
            b: [batch, seq_len, num_kv_heads]

        Returns:
            output: [batch, seq_len, num_kv_heads * head_dim] (未应用z门控)
        """
        # 使用ONNX算子进行计算
        return self.gated_delta_rule_module(q, k, v, a, b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, hidden_size]
        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = x.shape

        # ===== 投影 =====
        qkv = self.in_proj_qkv(x)
        q_size = self.key_dim
        q, kv = qkv.split([q_size, q_size * 2], dim=-1)
        k, v = kv.split([self.key_dim, self.value_dim], dim=-1)

        z = self.in_proj_z(x)
        b = self.in_proj_b(x)
        a = self.in_proj_a(x)

        # ===== 应用RoPE到Q和K =====
        # 形状: [batch_size, seq_len, num_kv_heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # 转换为 apply_rotary_pos_emb 期望的形状: [batch_size, num_kv_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        # 应用RoPE - 保持 num_kv_heads 作为 head 维度
        q, k = apply_rotary_pos_emb(
            q, k,
            self.freqs_cos[:seq_len],
            self.freqs_sin[:seq_len]
        )

        # 转换回原始形状: [batch_size, seq_len, num_kv_heads * head_dim]
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        q = q.view(batch_size, seq_len, -1)
        k = k.view(batch_size, seq_len, -1)

        # ===== GatedDeltaRule 计算 =====
        attn_output = self.gated_delta_rule(q, k, v, a, b)

        # ===== 在外部应用z门控 =====
        # z的sigmoid门控在这里应用，与GatedDeltaRule分离
        z_gate = torch.sigmoid(z)  # [batch_size, seq_len, value_dim]
        attn_output = attn_output * z_gate  # 元素级乘法

        # ===== 归一化 =====
        attn_output = attn_output.view(batch_size * seq_len, -1)
        z_reshaped = z.view(batch_size * seq_len, -1)
        attn_output = self.norm(attn_output, z_reshaped)
        attn_output = attn_output.view(batch_size, seq_len, -1)

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
            print(f"gate: {gate.shape}")
            print(f"q: {gate.shape}")
            print(f"k: {gate.shape}")
            print(f"v: {gate.shape}")
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
    print("3层 GatedDeltaNet + 1层 GatedAttention")
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
