import torch
import torch.nn as nn
import onnx

# 定义包含线性层的模型（nn.Linear底层会被导出为GEMM）
class LinearModel(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(in_features, out_features)  # y = x*W^T + b

    def forward(self, x):
        return self.linear(x)

if __name__ == "__main__":
    # 初始化模型
    model = LinearModel(in_features=10, out_features=5).eval()
    x = torch.randn(4, 10)  # 输入：batch_size=4，特征数=10

