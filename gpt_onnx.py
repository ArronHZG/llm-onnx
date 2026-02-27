import onnxsim
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

output_path = "onnx_data/gpt_transformer.onnx"
batch_size = 2
seq_len = 10
vocab_size = 1234


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
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

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        前向传播
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, 1, seq_len, seq_len] (因果掩码)
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # batch_size = x.size(0)

        # 线性投影并拆分多头: [batch_size, seq_len, n_heads, d_k]
        q = self.w_q(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 注意力计算: [batch_size, n_heads, seq_len, seq_len]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 应用掩码（兼容ONNX）
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # 注意力权重和输出
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # [batch_size, n_heads, seq_len, d_k] -> [batch_size, seq_len, d_model]
        attn_output = torch.matmul(attn_weights, v).transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)

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
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # Pre-LN结构（GPT的特点）
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
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
        # seq_len = x.size(1)
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


class GPTTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 768, n_heads: int = 12,
                 n_layers: int = 12, d_ff: int = 3072, dropout: float = 0.1):
        super().__init__()
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 位置嵌入（GPT使用绝对位置嵌入）
        self.pos_embedding = nn.Embedding(1024, d_model)  # 最大序列长度1024
        # 词嵌入dropout
        self.drop_emb = nn.Dropout(0.1)
        # Transformer层堆叠
        self.layers = nn.ModuleList([
            GPTTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        # 最终层归一化
        self.ln_f = nn.LayerNorm(d_model)
        # 输出层
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor):
        """
        完整GPT Transformer前向传播
        Args:
            input_ids: [batch_size, seq_len]
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        # batch_size, seq_len = input_ids.size()

        # 位置编码
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)

        # 嵌入层
        x = self.embedding(input_ids) + self.pos_embedding(position_ids)
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
    # 设置模型为评估模式
    model.train()

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
    # 1. 创建模型实例
    model = GPTTransformer(
        vocab_size=vocab_size,
        d_model=768,
        n_heads=12,
        n_layers=1,  # 简化版，实际GPT-2是12层
        d_ff=3072
    )

    # 2. 测试前向传播
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(dummy_input)
    print(f"前向传播输出形状: {logits.shape}")  # 预期: [2, 10, 50257]

    # 3. 导出ONNX
    export_to_onnx(model, dummy_input, output_path)

    # 4. 验证ONNX模型（可选）
    import onnx

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX模型验证通过")

    sim_path = output_path.replace(".onnx", "_sim.onnx")
    simplify_onnx(output_path, sim_path)
