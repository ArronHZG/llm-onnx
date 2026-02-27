import matplotlib.pyplot as plt
import numpy as np
import torch


class RoPE(torch.nn.Module):
    """RoPE rotary position encoding implementation (adapted for visualization, preserving core logic)"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        # 预计算频率项（与SinPE完全一致）
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        print(f"inv_freq: {inv_freq.shape}")
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        # 预计算位置（0到max_len-1）
        positions = torch.arange(max_len, dtype=torch.float)
        self.register_buffer('positions', positions, persistent=False)

    def _compute_rotary_emb(self, seq_len: int):
        """Compute sin and cos values for rotation"""
        # 截取当前序列长度的位置
        positions = self.positions[:seq_len].unsqueeze(1)  # (seq_len, 1)
        # 计算旋转角度：n * theta_m
        freqs = torch.matmul(positions, self.inv_freq.unsqueeze(0))  # (seq_len, d_model//2)
        # 直接返回sin和cos，形状为 (seq_len, d_model//2)
        sin = torch.sin(freqs)  # (seq_len, d_model//2)
        cos = torch.cos(freqs)  # (seq_len, d_model//2)
        return sin, cos

    def forward(self, x: torch.Tensor):
        """Forward pass: apply RoPE rotation to input vectors"""
        seq_len, d_model = x.shape[1], x.shape[2]
        assert d_model == self.d_model, "输入维度与RoPE维度不匹配"

        sin, cos = self._compute_rotary_emb(seq_len)
        print(f"sin: {sin.shape}, cos: {cos.shape}")
        # 应用旋转公式（奇偶维度拆分）
        x_even = x[..., ::2]  # 偶数维度 (batch_size, seq_len, d_model//2)
        x_odd = x[..., 1::2]  # 奇数维度 (batch_size, seq_len, d_model//2)
        # sin, cos 形状为 (seq_len, d_model//2)，自动广播到batch维度
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos
        # 重新交错奇偶维度以恢复原始形状
        x_rot = torch.zeros_like(x)
        x_rot[..., ::2] = x_rot_even
        x_rot[..., 1::2] = x_rot_odd
        # 为了可视化，返回扩展后的sin和cos（形状为 (seq_len, d_model)）
        sin_expanded = sin.repeat_interleave(2, dim=-1)  # (seq_len, d_model)
        cos_expanded = cos.repeat_interleave(2, dim=-1)  # (seq_len, d_model)
        return x_rot, sin_expanded, cos_expanded


# -------------------------- 可视化函数 --------------------------
def visualize_rope(d_model: int = 64, seq_len: int = 50):
    """
    Visualize core features of RoPE:
    1. Vector rotation effect at a single position (comparing distribution before and after rotation)
    2. RoPE rotation angle differences across positions (heatmap)
    3. Visual comparison between RoPE and SinPE (highlighting mechanism differences)
    """
    # 初始化RoPE和SinPE（用于对比）
    rope = RoPE(d_model=d_model, max_len=seq_len)

    # 模拟随机词嵌入（用于演示旋转效果）
    torch.manual_seed(42)  # 固定随机种子，保证结果可复现
    embedding = torch.ones(1, seq_len, d_model)  # (batch_size=1, seq_len, d_model)

    # 计算RoPE旋转后的向量、sin和cos
    rope_emb, sin_rope, cos_rope = rope(embedding)

    # 计算SinPE（用于对比）
    sin_pe = sin_rope.squeeze(0).detach().numpy()
    cos_pe = cos_rope.squeeze(0).detach().numpy()
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = sin_pe[:, 0::2]
    pe[:, 1::2] = cos_pe[:, 1::2]
    sin_emb = embedding + torch.from_numpy(pe).unsqueeze(0).to(embedding.device)

    # 转换为numpy数组（便于可视化）
    embedding_np = embedding.squeeze(0).detach().numpy()
    rope_emb_np = rope_emb.squeeze(0).detach().numpy()
    sin_emb_np = sin_emb.squeeze(0).detach().numpy()

    # 设置画布，4个子图展示核心信息
    plt.rcParams['figure.figsize'] = (20, 16)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    # Subplot 1: Vector rotation effect at a single position (position 10), using first 2 dimensions for intuitive display
    pos_idx = 10  # 选择第10个位置
    ax1.scatter(embedding_np[pos_idx, 0], embedding_np[pos_idx, 1],
                color='blue', label='Before rotation (original embedding)', s=80)
    ax1.scatter(rope_emb_np[pos_idx, 0], rope_emb_np[pos_idx, 1],
                color='red', label='After rotation (RoPE)', s=80)
    # 绘制旋转角度示意线
    ax1.arrow(0, 0, embedding_np[pos_idx, 0], embedding_np[pos_idx, 1],
              color='blue', linestyle='--', alpha=0.5, label='Before rotation vector')
    ax1.arrow(0, 0, rope_emb_np[pos_idx, 0], rope_emb_np[pos_idx, 1],
              color='red', linestyle='--', alpha=0.5, label='After rotation vector')
    ax1.set_title(f'Vector rotation effect at position {pos_idx} (first 2 dimensions)', fontsize=14)
    ax1.set_xlabel('Dimension 0', fontsize=12)
    ax1.set_ylabel('Dimension 1', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    # 设置坐标轴范围，使原点(0,0)居中，且x轴和y轴长度相同
    x_vals = [embedding_np[pos_idx, 0], rope_emb_np[pos_idx, 0]]
    y_vals = [embedding_np[pos_idx, 1], rope_emb_np[pos_idx, 1]]
    max_abs_x = max(abs(x) for x in x_vals)
    max_abs_y = max(abs(y) for y in y_vals)
    margin = 0.1  # 边距比例
    limit = max(max_abs_x, max_abs_y) * (1 + margin)  # 使用最大值确保x轴和y轴等长
    ax1.set_xlim(-limit, limit)
    ax1.set_ylim(-limit, limit)
    ax1.set_aspect('equal', adjustable='box')

    # Subplot 2: RoPE rotation angle heatmap (position × dimension)
    # Rotation angle = arctan2(sin, cos), reflecting rotation degree at different positions and dimensions
    rot_angle = np.arctan2(sin_rope.squeeze(0).detach().numpy(), cos_rope.squeeze(0).detach().numpy())
    im2 = ax2.imshow(rot_angle, cmap='hsv', aspect='equal')
    ax2.set_title(f'RoPE rotation angle heatmap (d_model={d_model}, seq_len={seq_len})', fontsize=14)
    ax2.set_xlabel('Dimension (d_model)', fontsize=12)
    ax2.set_ylabel('Position (seq_len)', fontsize=12)
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Rotation angle (radians)', fontsize=10)

    # Subplot 3: Vector distribution comparison between RoPE and SinPE (positions 0~20, first 2 dimensions)
    for i in range(20):
        ax3.scatter(rope_emb_np[i, 0], rope_emb_np[i, 1], color='red', alpha=0.6, label='RoPE' if i == 0 else "")
        ax3.scatter(sin_emb_np[i, 0], sin_emb_np[i, 1], color='blue', alpha=0.6, label='SinPE' if i == 0 else "")
    ax3.set_title('Vector distribution comparison: RoPE vs SinPE (first 20 positions, first 2 dimensions)', fontsize=14)
    ax3.set_xlabel('Dimension 0', fontsize=12)
    ax3.set_ylabel('Dimension 1', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal', adjustable='box')

    # Subplot 4: sin value variation of first 8 dimensions in RoPE (showing frequency characteristics, consistent with SinPE)
    positions = np.arange(seq_len)
    for i in [0, 31, 63]:
        ax4.plot(positions, sin_rope.squeeze(0).detach().numpy()[:, i], label=f'Dimension {i}')
    ax4.set_title(
        'sin value variation of first 8 dimensions in RoPE (showing frequency differences, consistent with SinPE)',
        fontsize=14)
    ax4.set_xlabel('Position', fontsize=12)
    ax4.set_ylabel('sin value (related to rotation angle)', fontsize=12)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    # 调整布局，避免重叠
    plt.tight_layout()
    # 保存图片（可选）
    plt.savefig('rope_visualization.png', dpi=300, bbox_inches='tight')
    # 显示图片
    plt.show()


# -------------------------- 执行可视化 --------------------------
if __name__ == "__main__":
    # 配置参数（适配可视化，维度不宜过大）
    d_model = 64  # 特征维度（较小维度便于可视化）
    seq_len = 50  # 序列长度
    # 执行可视化
    visualize_rope(d_model=d_model, seq_len=seq_len)
