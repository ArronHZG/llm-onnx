import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def precompute_freqs_cis(dim: int, end: int, rope_base: float = 10000.0):
    """
    预计算RoPE的cos和sin值
    Args:
        dim: 每个头的维度（d_model // n_heads）
        end: 最大序列长度
        rope_base: RoPE的基数，默认为10000
    Returns:
        freqs_cos: [end, dim] 预计算的cos值
        freqs_sin: [end, dim] 预计算的sin值
    """
    inv_freq = 1.0 / (rope_base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end, device=inv_freq.device)
    freqs = torch.outer(t, inv_freq).float()
    # 将freqs复制一份，以便与交错后的维度匹配
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    # 交错重复以匹配原始维度（因为每个复数维度对应两个实数维度）
    freqs_cos = freqs_cos.repeat_interleave(2, dim=-1)  # shape: [end, dim]
    freqs_sin = freqs_sin.repeat_interleave(2, dim=-1)  # shape: [end, dim]
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, freqs_cos, freqs_sin):
    """
    将旋转位置编码应用于查询和键
    Args:
        q: [batch_size, n_heads, seq_len, head_dim]
        k: [batch_size, n_heads, seq_len, head_dim]
        freqs_cos: [seq_len, head_dim]
        freqs_sin: [seq_len, head_dim]
    Returns:
        q_rot: 旋转后的查询
        k_rot: 旋转后的键
    """
    # 将cos和sin扩展到与q、k相同的形状
    cos = freqs_cos.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, head_dim]
    sin = freqs_sin.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, head_dim]

    # 旋转公式: q_rot = q * cos + rotate_half(q) * sin
    # rotate_half: 将后半部分维度与前半部分交换并取反
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


class MultiHeadAttentionWithRoPE(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, rope_base: float = 10000.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.rope_base = rope_base

        # 线性投影层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # 预计算的位置编码缓冲区（将在第一次前向传播时初始化）
        self.register_buffer('freqs_cos', None, persistent=False)
        self.register_buffer('freqs_sin', None, persistent=False)
        self.max_seq_len = 0

    def _init_rope(self, seq_len: int):
        """如果需要，初始化RoPE位置编码"""
        if self.freqs_cos is None or seq_len > self.max_seq_len:
            self.max_seq_len = max(seq_len, self.max_seq_len)
            # 预计算足够长度的位置编码
            self.freqs_cos, self.freqs_sin = precompute_freqs_cis(
                dim=self.d_k,
                end=self.max_seq_len,
                rope_base=self.rope_base
            )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        前向传播
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, 1, seq_len, seq_len] (因果掩码)
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # 初始化RoPE位置编码
        self._init_rope(seq_len)

        # 线性投影并拆分多头: [batch_size, seq_len, n_heads, d_k]
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 应用旋转位置编码
        q_rot, k_rot = apply_rotary_pos_emb(q, k,
                                           self.freqs_cos[:seq_len],
                                           self.freqs_sin[:seq_len])

        # 注意力计算: [batch_size, n_heads, seq_len, seq_len]
        attn_scores = torch.matmul(q_rot, k_rot.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 应用掩码（因果掩码）
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # 注意力权重和输出
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # [batch_size, n_heads, seq_len, d_k] -> [batch_size, seq_len, d_model]
        attn_output = torch.matmul(attn_weights, v).transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)

        # 最终投影
        output = self.w_o(attn_output)
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activate = nn.ReLU()  # GPT使用GELU激活

    def forward(self, x: torch.Tensor):
        """
        前向传播
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        x = self.linear1(x)
        x = self.activate(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class GPTTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, rope_base: float = 10000.0):
        super().__init__()
        # Pre-LN结构（GPT的特点）
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttentionWithRoPE(d_model, n_heads, dropout, rope_base)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def _create_causal_mask(self, seq_len: int, device: torch.device):
        """
        创建因果掩码（兼容ONNX）
        Args:
            seq_len: 序列长度
            device: 设备
        Returns:
            mask: [1, 1, seq_len, seq_len]
        """
        # 使用triu构造上三角掩码，避免动态条件判断
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        # 转换为0/1掩码（ONNX更兼容）
        mask = ~mask  # True表示可见，False表示不可见
        mask = mask.unsqueeze(0).unsqueeze(0)  # 扩展维度
        return mask

    def forward(self, x: torch.Tensor):
        """
        前向传播
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        # 创建因果掩码
        causal_mask = self._create_causal_mask(seq_len, x.device)

        # 自注意力子层（残差连接）
        x_residual = x
        x = self.ln1(x)
        x = self.attn(x, causal_mask)
        x = self.dropout(x)
        x = x + x_residual

        # 前馈子层（残差连接）
        x_residual = x
        x = self.ln2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x + x_residual

        return x


class GPTTransformerWithRoPE(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 768, n_heads: int = 12,
                 n_layers: int = 12, d_ff: int = 3072, dropout: float = 0.1,
                 rope_base: float = 10000.0, max_seq_len: int = 1024):
        super().__init__()
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 词嵌入dropout
        self.drop_emb = nn.Dropout(dropout)
        # Transformer层堆叠
        self.layers = nn.ModuleList([
            GPTTransformerBlock(d_model, n_heads, d_ff, dropout, rope_base)
            for _ in range(n_layers)
        ])
        # 最终层归一化
        self.ln_f = nn.LayerNorm(d_model)
        # 输出层
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # 保存配置参数
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.rope_base = rope_base

    def forward(self, input_ids: torch.Tensor):
        """
        完整GPT Transformer前向传播
        Args:
            input_ids: [batch_size, seq_len]
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """

        # 嵌入层
        x = self.embedding(input_ids)
        x = self.drop_emb(x)

        # 遍历Transformer层
        for layer in self.layers:
            x = layer(x)

        # 最终层归一化和输出
        x = self.ln_f(x)
        logits = self.head(x)

        return logits


# ONNX导出函数
def export_to_onnx(model: nn.Module, dummy_input: torch.Tensor, onnx_path: str):
    """
    将模型导出为ONNX格式
    Args:
        model: 训练好的模型
        dummy_input: 用于推断形状的虚拟输入 [batch_size, seq_len]
        onnx_path: ONNX文件保存路径
    """
    # 设置模型为评估模式（禁用dropout等随机操作）
    model.eval()

    # 导出配置（兼容ONNX）
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=False,
        input_names=['input_ids'],
        output_names=['logits'],
        # dynamic_axes={
        #     'input_ids': {0: 'batch_size', 1: 'seq_len'},  # 动态批次和序列长度
        #     'logits': {0: 'batch_size', 1: 'seq_len'}
        # }
    )
    print(f"模型已成功导出到: {onnx_path}")


def simplify_onnx(onnx_path: str, output_path: str = None):
    """
    使用 onnxsim 简化 ONNX 模型
    Args:
        onnx_path: 输入 ONNX 文件路径
        output_path: 输出 ONNX 文件路径（默认覆盖原文件）
    """
    try:
        import onnx
        import onnxsim
    except ImportError:
        print("警告: onnxsim 未安装，跳过简化步骤")
        return

    if output_path is None:
        output_path = onnx_path

    print(f"正在简化 ONNX 模型...")

    # 加载并简化模型
    model = onnx.load(onnx_path)
    check = False
    try:
        model_simp, check = onnxsim.simplify(model)
    except Exception as e:
        print(f"ONNX 模型简化失败: {e}")

    if check:
        onnx.save(model_simp, output_path)
        print(f"ONNX 模型简化成功: {output_path}")
    else:
        print("ONNX 模型简化失败")

    return


# 测试代码
if __name__ == "__main__":
    # 配置参数
    vocab_size = 50257  # GPT-2的词汇表大小
    batch_size = 2
    seq_len = 10

    # 1. 创建模型实例
    model = GPTTransformerWithRoPE(
        vocab_size=vocab_size,
        d_model=768,
        n_heads=12,
        n_layers=1,  # 简化版，实际GPT-2是12层
        d_ff=3072,
        max_seq_len=1024
    )

    # 2. 测试前向传播
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(dummy_input)
    print(f"前向传播输出形状: {logits.shape}")  # 预期: [2, 10, 50257]
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 3. 测试梯度
    loss = logits.sum()
    loss.backward()
    print("梯度计算成功")

    # 4. 测试不同序列长度
    seq_len2 = 5
    dummy_input2 = torch.randint(0, vocab_size, (batch_size, seq_len2))
    logits2 = model(dummy_input2)
    print(f"不同序列长度输出形状: {logits2.shape}")  # 预期: [2, 5, 50257]

    # 5. 导出ONNX（可选）
    export_onnx = False  # 设置为True以启用ONNX导出
    if export_onnx:
        output_path = "onnx_data/gpt_rope_transformer.onnx"
        export_to_onnx(model, dummy_input, output_path)

        # 验证ONNX模型（可选）
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX模型验证通过")

        sim_path = output_path.replace(".onnx", "_sim.onnx")
        simplify_onnx(output_path, sim_path)

