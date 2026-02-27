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

# 初始化模型
model = LinearModel(in_features=10, out_features=5).eval()
x = torch.randn(4, 10)  # 输入：batch_size=4，特征数=10

# 导出ONNX
onnx_path = "linear_gemm_model.onnx"
torch.onnx.export(
    model,
    x,
    onnx_path,
    opset_version=17,
    input_names=["x"],
    output_names=["y"],
    do_constant_folding=True,
    # dynamo=True
)

# 验证模型
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
print("线性层模型导出成功，底层已转换为GEMM算子！")

# 查看GEMM算子（nn.Linear的权重W和偏置b会作为常量存入ONNX）
for node in onnx_model.graph.node:
    if node.op_type == "Gemm":
        print(f"Linear层对应的GEMM输入: {node.input}")  # 输入包含x、W、b
        print(f"Linear层对应的GEMM输出: {node.output}")