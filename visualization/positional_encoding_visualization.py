import matplotlib.pyplot as plt
import numpy as np
import torch


class PositionalEncoding(torch.nn.Module):
    """Transformer 固定正弦余弦位置编码实现（兼容任意序列长度）"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        # 预计算位置编码：(max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        # 生成位置索引：(max_len, 1)，位置从0开始
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 计算频率项：10000^(2m/d_model)，m从0到d_model//2 - 1
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        # 偶数维度用正弦，奇数维度用余弦
        pe[:, 0::2] = torch.sin(position * div_term)  # 0,2,4...维度
        pe[:, 1::2] = torch.cos(position * div_term)  # 1,3,5...维度
        # 注册为缓冲区（非可学习参数），shape: (1, max_len, d_model)，适配batch维度
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：将位置编码与输入词嵌入相加
        Args:
            x: 输入词嵌入，shape: (batch_size, seq_len, d_model)
        Returns:
            x_with_pe: 注入位置信息后的输入，shape不变
        """
        seq_len = x.size(1)
        # 截取对应序列长度的PE，与输入相加
        x = x + self.pe[:, :seq_len, :]
        return x


# -------------------------- 可视化函数 --------------------------
def visualize_positional_encoding(d_model: int = 512, seq_len: int = 100):
    """
    可视化位置编码：
    1. 热力图：展示不同位置（行）、不同维度（列）的PE值分布
    2. 线图：展示前8个维度的PE值随位置的变化（体现频率差异）
    """
    # 初始化位置编码
    pe = PositionalEncoding(d_model=d_model, max_len=seq_len)
    # 获取PE矩阵：(seq_len, d_model)（去掉batch维度）
    pe_matrix = pe.pe.squeeze(0).detach().numpy()

    # 设置画布大小，适配两个子图
    plt.rcParams['figure.figsize'] = (16, 10)
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 1.2]})

    # 子图1：PE热力图（位置×维度）
    im = ax1.imshow(pe_matrix, cmap='viridis', aspect='auto')
    ax1.set_title(f'Positional Encoding heatmap (d_model={d_model}, seq_len={seq_len})', fontsize=14)
    ax1.set_xlabel('d_model', fontsize=12)
    ax1.set_ylabel('index seq_len', fontsize=12)
    # 添加颜色条，标注数值范围
    cbar1 = plt.colorbar(im, ax=ax1)
    cbar1.set_label('PE value (sine/cosine output)', fontsize=10)

    # 子图2：前8个维度的PE值随位置变化（体现频率差异）
    positions = np.arange(seq_len)
    for i in [0, 64, 128]:  # 取前8个维度，对比高频和低频
        ax2.plot(positions, pe_matrix[:, i], label=f'Dim {i}')
    ax2.set_title('The positional coding changes in the first 8 dimensions (reflecting frequency differences)）',
                  fontsize=14)
    ax2.set_xlabel('index', fontsize=12)
    ax2.set_ylabel('PE', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 调整布局，避免重叠
    plt.tight_layout()
    # 保存图片到image目录
    import os
    os.makedirs('image', exist_ok=True)
    plt.savefig('image/positional_encoding_visualization.png', dpi=300, bbox_inches='tight')
    # 显示图片
    plt.show()


# -------------------------- 测试与可视化 --------------------------
if __name__ == "__main__":
    # 配置参数（与Transformer常用配置一致）
    d_model = 512  # 词嵌入/PE维度
    seq_len = 100  # 序列长度
    # 执行可视化
    visualize_positional_encoding(d_model=d_model, seq_len=seq_len)
