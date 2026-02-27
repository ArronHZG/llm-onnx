"""
PyTorch 实现 Attention 机制并导出 ONNX 模型
"""
import onnx
import onnxsim
import torch
import torch.nn as nn
import torch.nn.functional as F


# 测试前向传播
batch_size = 2
seq_len = 10
d_model = 512

class SelfAttention(nn.Module):
    """
    自注意力机制 (Self-Attention)

    公式: Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
    """

    def __init__(self, d_model: int, num_heads: int = 8):
        super(SelfAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度

        # Q, K, V 的线性变换
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # 输出线性变换
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入 tensor, shape (batch_size, seq_len, d_model)
            mask: 注意力掩码, shape (batch_size, seq_len, seq_len)

        Returns:
            输出 tensor, shape (batch_size, seq_len, d_model)
        """

        # 线性变换得到 Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 将 Q, K, V 分割成多个头
        # shape: (batch_size, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 计算注意力分数: Q * K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

        # 应用掩码（如果有）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)

        # 应用注意力到 V
        attn_output = torch.matmul(attn_weights, V)

        # 合并多个头的结果
        # shape: (batch_size, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # 输出线性变换
        output = self.W_o(attn_output)

        return output


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制的完整实现（包含残差连接和层归一化）
    """

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()

        self.attention = SelfAttention(d_model, num_heads)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # 自注意力
        output = self.attention(x, mask)
        # 残差连接和层归一化
        output = self.norm(x + self.dropout(output))
        return output


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


def export_to_onnx(model: nn.Module, output_path: str, simplify: bool = True):
    """
    导出 PyTorch 模型为 ONNX 格式

    Args:
        model: PyTorch 模型
        output_path: ONNX 文件输出路径
        simplify: 是否使用 onnxsim 简化模型
    """
    # 设置模型为评估模式
    model.eval()

    # 创建示例输入
    batch_size = 2
    seq_len = 10
    d_model = 512

    # 创建虚拟输入
    dummy_input = torch.randn(batch_size, seq_len, d_model)

    # 导出为 ONNX (使用 dynamo=False 以支持 GEMM)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        dynamo=False,
        do_constant_folding=True,
        input_names=['x'],
        output_names=['y'],
        # dynamic_axes={
        #     'input': {0: 'batch_size', 1: 'seq_len'},
        #     'output': {0: 'batch_size', 1: 'seq_len'}
        # },
        # 开启 ONNX 优化，使用 GEMM 替代 MatMul + Add
        verbose=False
    )
    print(f"ONNX 模型已导出到: {output_path}")

    # 使用 onnxsim 简化，输出到 _sim.onnx 文件
    if simplify:
        sim_path = output_path.replace(".onnx", "_sim.onnx")
        simplify_onnx(output_path, sim_path)
        output_path = sim_path

    return output_path


def main():
    # 模型参数
    d_model = 512
    num_heads = 8

    # 创建模型
    model = MultiHeadAttention(d_model, num_heads)
    print(f"模型结构:\n{model}\n")

    # 测试前向传播
    batch_size = 2
    seq_len = 10

    # 创建测试输入
    x = torch.randn(batch_size, seq_len, d_model)

    # 前向传播
    with torch.no_grad():
        output = model(x)

    print(f"输入 shape: {x.shape}")
    print(f"输出 shape: {output.shape}")

    # 导出为 ONNX（自动简化）
    onnx_path = "onnx_data/attention_model.onnx"
    export_to_onnx(model, onnx_path, simplify=False)

    # 验证 ONNX 模型
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX 模型验证通过!")


if __name__ == "__main__":
    main()
