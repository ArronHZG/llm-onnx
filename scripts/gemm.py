

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