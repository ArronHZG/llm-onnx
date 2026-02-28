#!/usr/bin/env python3
"""
导出注意力模型为ONNX格式
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from attention import MultiHeadAttention
from utils.onnx_utils import export_and_simplify, validate_onnx


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

    # 导出为ONNX
    os.makedirs("onnx_data", exist_ok=True)
    onnx_path = "onnx_data/attention_model.onnx"

    # 导出并简化
    final_path = export_and_simplify(
        model=model,
        dummy_input=x,
        onnx_path=onnx_path,
        simplify=True,
        input_names=['x'],
        output_names=['y'],
        dynamic_axes={
            'x': {0: 'batch_size', 1: 'seq_len'},
            'y': {0: 'batch_size', 1: 'seq_len'}
        }
    )

    # 验证ONNX模型
    if validate_onnx(final_path):
        print("ONNX模型验证通过!")
    else:
        print("ONNX模型验证失败!")

    print(f"模型已导出到: {final_path}")


if __name__ == "__main__":
    main()

