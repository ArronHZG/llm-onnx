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

* RMSNorm 使用 LayerNorm 替换
* SwiGLU, GeLU 使用 ReLU 替换
* 固定 dims 参数

| 文件                   | 模型              | 特点                                                               |
|----------------------|-----------------|------------------------------------------------------------------|
| `attention.py`       | 基础Attention     | 标准的Multi-Head Attention实现                                        |
| `gpt.py`             | GPT Transformer | 多头自注意力 + 因果掩码、绝对位置嵌入、Pre-LN结构、前馈网络                               |
| `gpt_rope.py`        | GPT + RoPE      | 旋转位置编码（Rotary Position Embedding），更好的外推能力                        |
| `gpt_moe.py`         | GPT + MoE       | 专家混合（Mixture of Experts）、稀疏门控机制、Top-K专家路由                        |
| `qwen3_dense.py`     | Qwen3 Dense     | RoPE、GQA(Grouped Query Attention)、SwiGLU激活                       |
| `qwen3_moe.py`       | Qwen3 MoE       | RoPE、GQA、QK归一化、MoE(8专家+Top2路由)、SwiGLU                            |
| `qwen3_5_dense.py`   | Qwen3.5 Dense   | Zero-Centered RMSNorm <br/> Gated Attention <br/> Gated DeltaNet |
| `qwen3_5_moe.py`     | Qwen3.5 moe     |                                                                  |
| `deepseekv3.py`      | DeepSeek V3     | (待实现)                                                            |
| `deepseekv32.py`     | DeepSeek V3.2   | (待实现)                                                            |
| `deepseek_engram.py` | DeepSeek engram | (待实现)                                                            |
| `deepseek_mhc.py`    | DeepSeek mhc    | (待实现)                                                            |

## 可视化结果

以下展示了各模型架构的计算图可视化：

### GPT系列模型

![GPT Transformer](image/gpt_transformer_sim.png)
**GPT Transformer**：基础Transformer架构，包含多头自注意力机制和前馈网络，采用绝对位置嵌入。

![GPT RoPE](image/gpt_transformer_rope_sim.png)
**GPT + RoPE**：使用旋转位置编码（Rotary Position Embedding）替代绝对位置嵌入，提升位置外推能力。

![GPT MoE](image/gpt_transformer_moe_sim.png)
**GPT + MoE**：引入专家混合（Mixture of Experts）机制，通过稀疏门控选择Top-K专家进行计算。

### Qwen3系列模型

![Qwen3 Dense](image/qwen3_dense_sim.png)
**Qwen3 Dense**：采用RoPE旋转位置编码、GQA（分组查询注意力）和SwiGLU激活函数。

![Qwen3 MoE](image/qwen3_moe_sim.png)
**Qwen3 MoE**：在Dense基础上引入MoE结构，8个专家+Top-2路由机制。

![Qwen3.5 Dense](image/qwen3_5_dense_sim.png)
**Qwen3.5 Dense**：包含零中心化RMSNorm、门控注意力机制。

### 位置编码可视化

![RoPE Visualization](image/rope_visualization.png)
**RoPE可视化**：展示旋转位置编码的工作原理，左图为位置嵌入的热力图可视化，右图为旋转矩阵的实部和虚部。

![Positional Encoding](image/positional_encoding_visualization.png)
**位置编码对比**：对比绝对位置编码与旋转位置编码的差异，展示不同位置编码方式对token位置信息的表示。

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

## 仓库

参考：

>
> https://github.com/rasbt/LLMs-from-scratch
>
> https://github.com/jingyaogong/minimind
>
> https://github.com/sgl-project/sglang
>
> https://github.com/NVlabs/GatedDeltaNet
>
> https://github.com/fla-org/flash-linear-attention

> https://github.com/sustcsonglin/linear-attention-and-beyond-slides

## License

MIT License

