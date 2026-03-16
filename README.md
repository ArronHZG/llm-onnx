# LLM-ONNX

PyTorch Transformer模型实现及ONNX导出工具集。

## 项目结构

```
llm-onnx/
├── gpt.py              # GPT Transformer（绝对位置嵌入）
├── gpt_rope.py         # GPT + RoPE（旋转位置编码）
├── gpt_moe.py          # GPT + MoE（专家混合）
├── attention.py        # 基础注意力机制
├── utils/             # ONNX工具函数
├── visualization/     # 可视化脚本
├── minimind/         # MiniMind模型实现
├── scripts/          # 导出脚本
└── onnx_data/       # 导出的ONNX模型
```

## 模型实现

### 1. GPT Transformer (`gpt.py`)

- 多头自注意力 + 因果掩码
- 绝对位置嵌入
- Pre-LN结构
- 前馈网络

### 2. GPT + RoPE (`gpt_rope.py`)

- 旋转位置编码（Rotary Position Embedding）
- 相比绝对位置编码，RoPE具有更好的外推能力

### 3. GPT + MoE (`gpt_moe.py`)

- 专家混合（Mixture of Experts）
- 稀疏门控机制
- Top-K专家路由

## 安装依赖

```bash
pip install -r requirements.txt
```

依赖包括：

- torch>=2.0.0
- onnx>=1.14.0
- onnxsim>=0.4.33
- numpy>=1.24.0
- matplotlib>=3.7.0

## 使用方法

### 可视化

```bash
# 可视化RoPE
python -m visualization.rope_visualization

# 可视化位置编码
python -m visualization.positional_encoding_visualization
```

### 直接运行

```bash
# 测试GPT模型
python gpt.py

# 测试RoPE版本
python gpt_rope.py

# 测试MoE版本
python gpt_moe.py
```

## 配置参数

| 参数          | 说明            | 默认值    |
|-------------|---------------|--------|
| vocab_size  | 词汇表大小         | 1234   |
| d_model     | 模型隐藏维度        | 768    |
| n_heads     | 注意力头数         | 12     |
| n_layers    | Transformer层数 | 1（简化版） |
| d_ff        | 前馈网络隐藏维度      | 3072   |
| max_seq_len | 最大序列长度        | 1024   |

## ONNX导出

导出功能由 `utils/onnx_utils.py` 提供：

- `export_to_onnx()` - 导出PyTorch模型为ONNX
- `simplify_onnx()` - 使用onnxsim简化模型
- `export_and_simplify()` - 一站式导出+简化
- `validate_onnx()` - 验证ONNX模型有效性

## 可视化输出

可视化结果保存在 `image/` 目录：

- `rope_visualization.png` - RoPE可视化
- `positional_encoding_visualization.png` - 位置编码可视化

## License

MIT License

