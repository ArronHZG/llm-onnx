#!/usr/bin/env python3
"""
导出GPT Transformer模型为ONNX格式
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.gpt import GPTTransformer
from utils.onnx_utils import export_and_simplify, validate_onnx


def main():
    # 配置参数
    vocab_size = 50257  # GPT-2词汇表大小
    batch_size = 2
    seq_len = 10

    # 创建模型（简化版，1层）
    model = GPTTransformer(
        vocab_size=vocab_size,
        d_model=768,
        n_heads=12,
        n_layers=1,  # 简化版，实际GPT-2是12层
        d_ff=3072,
        max_seq_len=1024,
    )

    # 测试前向传播
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(dummy_input)
    print(f"前向传播输出形状: {logits.shape}")

    # 导出为ONNX
    os.makedirs("onnx_data", exist_ok=True)
    onnx_path = "onnx_data/gpt_transformer.onnx"

    # 导出并简化
    final_path = export_and_simplify(
        model=model,
        dummy_input=dummy_input,
        onnx_path=onnx_path,
        simplify=True,
        input_names=['input_ids'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'seq_len'},
            'logits': {0: 'batch_size', 1: 'seq_len'}
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

