# Qwen3.5 MFU (Model FLOPs Utilization) 计算详解

## 1. MFU 定义

$$\text{MFU} = \frac{\text{实测吞吐量 (tokens/s)} \times \text{模型 FLOPs/token}}{\text{GPU 理论峰值 FLOPS}}$$

或等价地：

$$\text{MFU} = \frac{\text{samples/s} \times \text{FLOPs}_{\text{total}}}{\text{GPU Peak FLOPS}}$$

---

## 2. 符号定义

| 符号         | 含义                                        |
|------------|-------------------------------------------|
| $b$        | batch size                                |
| $s$        | sequence length                           |
| $h$        | hidden_size                               |
| $n_h$      | num_heads (query heads)                   |
| $n_{kv}$   | num_kv_heads                              |
| $d$        | head_dim ($h = n_h \times d$)             |
| $L$        | 总层数                                       |
| $L_{gd}$   | GatedDeltaNet 层数                          |
| $L_{ga}$   | GatedAttention 层数 ($L = L_{gd} + L_{ga}$) |
| $E$        | 专家总数                                      |
| $k$        | top-k（每 token 激活的专家数）                     |
| $ffn\_{dim}$ | intermediate_size (FFN 中间维度)              |
| $V$        | vocab_size                                |
| $K$        | conv_kernel_size（ShortConvolution 卷积核大小）  |

---

## 3. 各组件 FLOPs 计算

### 3.1 Embedding 层

$$\text{FLOPs}_{\text{embed}} = 2bshV$$

### 3.2 RMSNorm（每层 2 次：input_layernorm + post_attention_layernorm）

$$\text{FLOPs}_{\text{norm/layer}} = 4bsh$$

### 3.3 GatedDeltaNet 层（线性注意力）

GagedDeltaNet 是 Qwen3.5 的核心创新，将标准注意力的 $O(s^2)$ 复杂度降至 $O(s)$。

**结构组成**：

- QKV 投影 + z 投影（输出门控）
- a, b 投影（Mamba-style gate 参数）
- ShortConvolution（因果卷积，kernel=K）
- GatedDeltaRule 核心递推计算
- out_proj 输出投影

**a) QKV 投影**：

$$\text{FLOPs}_{qkv} = 2bs[h(n_h d) + h(n_h d) + h(n_{kv} d)] = 2bsh(2n_h d + n_{kv} d)$$

**b) z 投影（输出门控）**：

$$\text{FLOPs}_z = 2bs \cdot h \cdot n_h d = 2bsh^2$$

**c) a, b 投影（Mamba-style gate）**：

$$\text{FLOPs}_{ab} = 4bs \cdot h \cdot n_h$$

**d) ShortConvolution**（Q, K, V 各一个深度可分离卷积，kernel=K）：

$$\text{FLOPs}_{conv} = 6Kbs \cdot h$$

**e) GatedDeltaRule 核心计算**（递推 s 步）：

对于序列中每个位置 $i$，执行：

1. **状态更新**：$S_i = g_i S_{i-1} + k_i v_i^T$
    - 矩阵乘法 $g_i S_{i-1}$：$2bn_h d^2$
    - 外积 $k_i v_i^T$：$bn_h d^2$
2. **输出计算**：$o_i = \beta_i q_i S_i / \sqrt{d}$
    - 矩阵向量乘 $q_i S_i$：$2bn_h d^2$

总计（s 步递推）：

$$\text{FLOPs}_{gdr} = bs(5n_h d^2)$$

> **注意**：这是 GagedDeltaNet 相比标准 Attention 的核心优势——复杂度从 $O(s^2)$ 降为 $O(s)$。

**f) out_proj**：

$$\text{FLOPs}_{out} = 2bs \cdot (n_h d) \cdot h = 2bsh^2$$

**单层 GagedDeltaNet 总计**：

$$\boxed{\text{FLOPs}_{GD/layer} = 4bsh^2 + 2bsh(2n_h d + n_{kv} d) + 4bs h n_h + 6Kbs h + 5bs n_h d^2}$$

简化形式（假设 $d \gg K$, $h = n_h d$）：

$$\boxed{\text{FLOPs}_{GD/layer} \approx 12bsh^2 + 10bs n_h d^2}$$

### 3.4 GagedAttention 层（标准多头注意力）

用于模型最后几层，保证生成质量。

**a) QKV 投影**（含 attn_output_gate 时 Q 维度翻倍）：

$$\text{FLOPs}_{qkv} = 2bs[h(2n_h d) + h(n_{kv} d) + h(n_{kv} d)] = 2bsh(2n_h d + 2n_{kv} d)$$

**b) 注意力分数计算**：$QK^T / \sqrt{d}$

$$\text{FLOPs}_{attn} = 2bn_h s^2 d$$

**c) Softmax + 加权求和**：$\text{AttnWeights} \times V$

$$\text{FLOPs}_{v} = 2bn_h s^2 d$$

**d) o_proj**：

$$\text{FLOPs}_o = 2bs \cdot (n_h d) \cdot h = 2bsh^2$$

**单层 GagedAttention 总计**：

$$\boxed{\text{FLOPs}_{GA/layer} = 6bsh^2 + 4bn_h s^2 d + 4bsh n_{kv} d}$$

### 3.5 MoE FFN 层（SwiGLU）

MoE 结构包含一个共享专家和 k 个路由专家。

**a) 共享专家 SwiGLU**（gate_proj + up_proj → SiLU(gate) × up → down_proj）：

$$\text{FLOPs}_{shared} = 2bs[2h \cdot ffn\_{dim} + ffn\_{dim} \cdot h] = 6bs \cdot h \cdot ffn\_{dim}$$

**b) 路由专家**（k 个激活的专家）：

$$\text{FLOPs}_{experts} = k \times 6bs \cdot h \cdot ffn\_{dim}$$

**c) Gate 路由网络**（TopKGate）：

$$\text{FLOPs}_{gate} = 2bs \cdot h \cdot E$$

**单层 MoE FFN 总计**：

$$\boxed{\text{FLOPs}_{MoE/layer} = 6(k+1)bs \cdot h \cdot ffn\_{dim} + 2bs \cdot h \cdot E}$$

### 3.6 LM Head

$$\text{FLOPs}_{lmhead} = 2bshV$$

---

## 4. 完整模型总 FLOPs

$$\begin{aligned}
\text{FLOPs}_{\text{total}} &= \underbrace{2bshV}_{\text{Embedding}} + \underbrace{2bshV}_{\text{LM Head}} \\
&+ L_{gd} \times (\underbrace{4bsh}_{\text{Norm}} + \underbrace{\text{FLOPs}_{GD/layer}}_{\text{GagedDeltaNet}} + \underbrace{\text{FLOPs}_{MoE/layer}}_{\text{MoE FFN}}) \\
&+ L_{ga} \times (\underbrace{4bsh}_{\text{Norm}} + \underbrace{\text{FLOPs}_{GA/layer}}_{\text{GagedAttention}} + \underbrace{\text{FLOPs}_{MoE/layer}}_{\text{MoE FFN}})
\end{aligned}$$

展开后：

$$\begin{aligned}
\text{FLOPs}_{\text{total}} &= 4bshV \\
&+ L_{gd}[12bsh^2 + 10bs n_h d^2 + 6(k+1)bsh \cdot ffn\_{dim} + 2bshE] \\
&+ L_{ga}[10bsh^2 + 4bn_h s^2 d + 4bsh n_{kv} d + 6(k+1)bsh \cdot ffn\_{dim} + 2bshE]
\end{aligned}$$

---

## 5. 各组件 FLOPs 量级对比

| 组件                  | FLOPs 量级                        | 对 seq_len 复杂度 | 备注             |
|---------------------|---------------------------------|---------------|----------------|
| Embedding / LM Head | $O(bshV)$                       | $O(1)$        | 与 V 成正比        |
| RMSNorm             | $O(bsh)$                        | $O(1)$        | 可忽略            |
| **GagedDeltaNet**   | **$O(bsh^2 + bsn_hd^2)$**       | **$O(s)$**    | **核心优势：线性复杂度** |
| GagedAttention      | $O(bsh^2 + bn_hs^2d)$           | $O(s^2)$      | 标准二次复杂度        |
| MoE FFN             | $O(k \cdot bsh \cdot ffn\_{dim})$ | $O(1)$        | 与激活专家数 k 成正比   |

---

## 6. Qwen3.5 MFU 优势分析

### 6.1 GagedDeltaNet vs 标准 Attention 的 FLOPs 对比

对于相同配置 $(b, s, h, n_h, d)$：

$$\frac{\text{FLOPs}_{GA}}{\text{FLOPs}_{GD}} \approx \frac{4bn_h s^2 d}{5bs n_h d^2} = \frac{4s}{5d}$$

当 $s > d$ 时（长序列场景），标准 Attention 的 FLOPs 远超 GagedDeltaNet。

**示例**（典型配置 $d=128, s=4096$）：

$$\frac{\text{FLOPs}_{GA}}{\text{FLOPs}_{GD}} \approx \frac{4 \times 4096}{5 \times 128} \approx 25.6$$

即标准 Attention 的 FLOPs 是 GagedDeltaNet 的 **25 倍以上**。

### 6.2 混合架构的设计意义

Qwen3.5 采用混合架构的原因：

1. **大部分层使用 GagedDeltaNet**：降低整体 FLOPs，提升推理吞吐
2. **最后几层使用 GagedAttention**：保证生成质量（全局注意力建模能力更强）
3. **MoE FFN**：增加模型容量而不显著增加计算量（只激活 k 个专家）

这种设计在**保持模型质量的同时最大化 MFU**。

---

## 7. 实际计算示例

以 Qwen3.5-MoE-A3B 为例（假设参数）：

| 参数                          | 值      |
|-----------------------------|--------|
| $h$ (hidden_size)           | 2048   |
| $n_h$ (num_heads)           | 16     |
| $n_{kv}$ (num_kv_heads)     | 16     |
| $d$ (head_dim)              | 128    |
| $L$ (总层数)                   | 28     |
| $L_{gd}$ (GagedDeltaNet 层)  | 24     |
| $L_{ga}$ (GagedAttention 层) | 4      |
| $E$ (专家数)                   | 64     |
| $k$ (top-k)                 | 8      |
| $ffn\_{dim}$                  | 5632   |
| $V$ (vocab_size)            | 152064 |
| $K$ (conv_kernel_size)      | 4      |
| $b$ (batch_size)            | 1      |
| $s$ (seq_len)               | 4096   |

**各部分 FLOPs（估算）**：

| 组件                    | FLOPs (×10⁹) | 占比   |
|-----------------------|--------------|------|
| Embedding + LM Head   | ~25.3        | < 1% |
| GagedDeltaNet 层 (×24) | ~2,400       | ~35% |
| GagedAttention 层 (×4) | ~1,200       | ~17% |
| MoE FFN 层 (×28)       | ~3,300       | ~48% |
| **Total**             | **~6,925**   | 100% |

**每个 token 的 FLOPs**：

$$\frac{\text{FLOPs}_{\text{total}}}{bs} = \frac{6.925 \times 10^{12}}{4096} \approx 1.69 \times 10^9 \text{ FLOPs/token}$$

**MFU 计算**（假设 A100 GPU，理论峰值 312 TFLOPS FP16）：

若实测吞吐量为 150,000 tokens/s：

$$\text{MFU} = \frac{150000 \times 1.69 \times 10^9}{312 \times 10^{12}} \approx 0.81 = 81\%$$

---

## 8. 反向传播（训练）FLOPs 计算

### 8.1 前向 vs 反向 FLOPs 的关系

在深度学习中，**反向传播的 FLOPs 通常约为前向传播的 2 倍**。原因如下：

- 每个前向操作（如矩阵乘法 $Y = XW$）在反向时需要计算两个梯度：
  - $\frac{\partial L}{\partial W} = X^T \cdot \frac{\partial L}{\partial Y}$（参数梯度）
  - $\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} \cdot W^T$（输入梯度）
- 这两个矩阵乘法的 FLOPs 与前向的矩阵乘法在同一量级
- 因此每个线性层的反向 FLOPs ≈ 2 × 前向 FLOPs

对于非线性操作（如 Softmax、SiLU、RMSNorm 等），反向 FLOPs 通常与前向相当或略小。

$$\text{FLOPs}_{\text{backward}} \approx 2 \times \text{FLOPs}_{\text{forward}}$$

$$\text{FLOPs}_{\text{train}} = \text{FLOPs}_{\text{forward}} + \text{FLOPs}_{\text{backward}} + \text{FLOPs}_{\text{optimizer}} \approx 3 \times \text{FLOPs}_{\text{forward}}$$

> **注意**：优化器状态更新（如 Adam 的动量和方差更新）通常只占总 FLOPs 的 5% 左右，可近似忽略或单独估算。

### 8.2 各组件反向 FLOPs 详细计算

#### 8.2.1 线性层 (Linear / MatMul)

**前向**：$Y = XW^T$，其中 $X \in \mathbb{R}^{M \times K}$, $W \in \mathbb{R}^{N \times K}$, $Y \in \mathbb{R}^{M \times N}$

$$\text{FLOPs}_{fwd} = 2MNK$$

**反向**：
- $\nabla_W = X^T \nabla_Y$ → $2MNK$
- $\nabla_X = \nabla_Y W$ → $2MNK$

$$\boxed{\text{FLOPs}_{bwd} = 4MNK = 2 \times \text{FLOPs}_{fwd}}$$

#### 8.2.2 RMSNorm

**前向**：归一化 + 缩放

$$\text{FLOPs}_{fwd} = 6bs h$$

**反向**：需要重新计算均值/方差并传播梯度

$$\text{FLOPs}_{bwd} \approx 10bs h$$

$$\boxed{\text{FLOPs}_{bwd} \approx 1.7 \times \text{FLOPs}_{fwd}}$$

#### 8.2.3 GagedDeltaRule 核心递推

**前向**：s 步递推，每步 $5bn_h d^2$

$$\text{FLOPs}_{fwd} = 5bs n_h d^2$$

**反向**：反向递推同样 s 步，且每步需要：
- 状态矩阵 $S_i$ 的梯度回传（涉及更多矩阵运算）
- 对 $q_i, k_i, v_i, g_i, \beta_i$ 分别求导

$$\text{FLOPs}_{bwd} \approx 12bs n_h d^2$$

$$\boxed{\text{FLOPs}_{bwd} \approx 2.4 \times \text{FLOPs}_{fwd}}$$

> GagedDeltaRule 的反向略高于 2x，因为递推状态的梯度依赖链较长。

#### 8.2.4 标准 Attention（QK^T 和 AttnWeights × V）

**前向**：

$$\text{FLOPs}_{fwd} = 4bn_h s^2 d$$

**反向**：
- $\nabla_Q = \nabla_{attn} K^T$ → $2bn_h sd^2$
- $\nabla_K = Q^T \nabla_{attn}$ → $2bn_h s^2 d$
- $\nabla_V = \text{AttnWeights}^T \nabla_{out}$ → $2bn_h s^2 d$
- Softmax 反向：$\approx 2bn_h s^2$

$$\text{FLOPs}_{bwd} \approx 8bn_h s^2 d$$

$$\boxed{\text{FLOPs}_{bwd} \approx 2 \times \text{FLOPs}_{fwd}}$$

#### 8.2.5 SwiGLU（MoE FFN）

**前向**：gate_proj + up_proj + SiLU × down_proj

$$\text{FLOPs}_{fwd} = 6bs \cdot h \cdot ffn\_{dim}$$

**反向**：
- gate_proj, up_proj, down_proj 各自的反向 ≈ 2× 前向
- SiLU 反向 ≈ 前向

$$\text{FLOPs}_{bwd} \approx 14bs \cdot h \cdot ffn\_{dim}$$

$$\boxed{\text{FLOPs}_{bwd} \approx 2.33 \times \text{FLOPs}_{fwd}}$$

#### 8.2.6 ShortConvolution（深度卷积）

**前向**：

$$\text{FLOPs}_{fwd} = 2Kbs h$$

**反向**（卷积核梯度和输入梯度）：

$$\text{FLOPs}_{bwd} \approx 4Kbs h$$

$$\boxed{\text{FLOPs}_{bwd} = 2 \times \text{FLOPs}_{fwd}}$$

#### 8.2.7 SiLU / Sigmoid / Softplus 等激活函数

$$\boxed{\text{FLOPs}_{bwd} \approx 1 \sim 1.5 \times \text{FLOPs}_{fwd}}$$

#### 8.2.8 Embedding 层

**前向**：查表操作，无算术 FLOPs（或计为 $2bshV$）

**反向**（Embedding 梯度累积）：

$$\text{FLOPs}_{bwd} \approx 2bshV$$

$$\boxed{\text{FLOPs}_{bwd} \approx 1 \times \text{FLOPs}_{fwd}}$$

### 8.3 各层完整前后向 FLOPs 汇总

#### 单层 GagedDeltaNet 完整前后向

| 操作 | 前向 FLOPs | 反向 FLOPs | 反向/前向比 |
|------|-----------|-----------|------------|
| QKV 投影 | $2bsh(2n_hd + n_{kv}d)$ | $4bsh(2n_hd + n_{kv}d)$ | 2.0× |
| z 投影 | $2bsh^2$ | $4bsh^2$ | 2.0× |
| a, b 投影 | $4bsh n_h$ | $8bsh n_h$ | 2.0× |
| ShortConv | $6Kbs h$ | $12Kbs h$ | 2.0× |
| GagedDeltaRule | $5bs n_h d^2$ | $12bs n_h d^2$ | ~2.4× |
| out_proj | $2bsh^2$ | $4bsh^2$ | 2.0× |

$$\boxed{\begin{aligned}
\text{FLOPs}_{GD/layer}^{fwd} &\approx 12bsh^2 + 10bs n_h d^2 \\
\text{FLOPs}_{GD/layer}^{bwd} &\approx 26bsh^2 + 24bs n_h d^2 \\
\text{Ratio}_{GD} &\approx 2.15 \times
\end{aligned}}$$

#### 单层 GagedAttention 完整前后向

| 操作 | 前向 FLOPs | 反向 FLOPs | 反向/前向比 |
|------|-----------|-----------|------------|
| QKV 投影 | $2bsh(2n_hd + 2n_{kv}d)$ | $4bsh(2n_hd + 2n_{kv}d)$ | 2.0× |
| Attention (QK^T) | $2bn_h s^2 d$ | $4bn_h s^2 d$ | 2.0× |
| Attention (AttnW × V) | $2bn_h s^2 d$ | $4bn_h s^2 d$ | 2.0× |
| o_proj | $2bsh^2$ | $4bsh^2$ | 2.0× |

$$\boxed{\begin{aligned}
\text{FLOPs}_{GA/layer}^{fwd} &= 6bsh^2 + 4bn_h s^2 d \\
\text{FLOPs}_{GA/layer}^{bwd} &= 14bsh^2 + 12bn_h s^2 d \\
\text{Ratio}_{GA} &\approx 2.2 \times
\end{aligned}}$$

#### 单层 MoE FFN 完整前后向

| 操作 | 前向 FLOPs | 反向 FLOPs | 反向/前向比 |
|------|-----------|-----------|------------|
| 共享专家 SwiGLU | $6bsh \cdot ffn\_{dim}$ | $14bsh \cdot ffn\_{dim}$ | ~2.33× |
| k 个路由专家 | $6kbsh \cdot ffn\_{dim}$ | $14kbsh \cdot ffn\_{dim}$ | ~2.33× |
| Gate 路由 | $2bshE$ | $4bshE$ | 2.0× |

$$\boxed{\begin{aligned}
\text{FLOPs}_{MoE/layer}^{fwd} &= 6(k+1)bsh \cdot ffn\_{dim} + 2bshE \\
\text{FLOPs}_{MoE/layer}^{bwd} &\approx 14(k+1)bsh \cdot ffn\_{dim} + 4bshE \\
\text{Ratio}_{MoE} &\approx 2.25 \times
\end{aligned}}$$

### 8.4 训练总 FLOPs 公式

$$\begin{aligned}
\text{FLOPs}_{\text{train}} &= \text{FLOPs}_{\text{forward}} + \text{FLOPs}_{\text{backward}} + \text{FLOPs}_{\text{optimizer}} \\
&\approx 3 \times \text{FLOPs}_{\text{forward}}
\end{aligned}$$

更精确的表达式：

$$\begin{aligned}
\text{FLOPs}_{\text{train}} &= 4bshV \quad (\text{Embedding fwd+bwd}) \\
&+ L_{gd}[38bsh^2 + 34bs n_h d^2 + 20(k+1)bsh \cdot ffn\_{dim} + 6bshE] \\
&+ L_{ga}[20bsh^2 + 16bn_h s^2 d + 8bsh n_{kv}d + 20(k+1)bsh \cdot ffn\_{dim} + 6bshE] \\
&+ \underbrace{\text{FLOPs}_{\text{optimizer}}}_{\text{见下节}}
\end{aligned}$$

### 8.5 优化器 FLOPs

以 **AdamW** 为例（最常用优化器），每步需要对每个参数执行：

1. **动量更新** ($m_t$)：$m_t = \beta_1 m_{t-1} + (1-\beta_1) g$
2. **方差更新** ($v_t$)：$v_t = \beta_2 v_{t-1} + (1-\beta_2) g^2$
3. **偏差校正**：$\hat{m}_t = m_t / (1-\beta_1^t)$, $\hat{v}_t = v_t / (1-\beta_2^t)$
4. **参数更新**：$w_{t+1} = w_t - \eta \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$
5. **权重衰减**：$w_{t+1} = w_{t+1} - \lambda \eta w_t$

每个参数约 **~10 次 FLOPs**。

$$\text{FLOPs}_{\text{optimizer}} \approx 10 \times |\theta|$$

其中 $|\theta|$ 为模型总参数量。

对于 MoE 模型，注意：
- 总参数量包含所有专家的参数（即使不全部激活）
- 但 ZeRO 等技术可以分片存储优化器状态，不影响 FLOPs 计算但影响显存

**AdamW 优化器占训练总 FLOPs 的比例**：

$$\frac{\text{FLOPs}_{\text{opt}}}{\text{FLOPs}_{\text{train}}} \approx \frac{10|\theta|}{3 \times \text{FLOPs}_{\text{forward/token}} \times bs}$$

对于大模型长序列场景，这个比例通常为 **3%~8%**。

### 8.6 训练 MFU 计算

$$\text{MFU}_{\text{train}} = \frac{\text{实测训练吞吐量 (samples/s)} \times \text{FLOPs}_{\text{train/sample}}}{\text{GPU Peak FLOPS}}$$

或按 token 计算：

$$\text{MFU}_{\text{train}} = \frac{\text{实测吞吐量 (tokens/s)} \times \text{FLOPs}_{\text{train/token}}}{\text{GPU Peak FLOPS}}$$

其中：

$$\text{FLOPs}_{\text{train/token}} = \frac{\text{FLOPs}_{\text{train}}}{bs}$$

### 8.7 推理 vs 训练 MFU 对比

| 场景 | FLOPs/token | 相对比例 | 说明 |
|------|-------------|---------|------|
| **推理（仅前向）** | $\text{FLOPs}_{fwd}/(bs)$ | **1×** | 基准 |
| **训练（前向+反向+优化器）** | $\text{FLOPs}_{train}/(bs)$ | **~3×** | 含优化器 |
| **训练（前向+反向）** | $(\text{FLOPs}_{fwd}+\text{FLOPs}_{bwd})/(bs)$ | **~2.9×** | 不含优化器 |

**示例**（沿用第 7 节配置）：

| 场景 | 每个 token FLOPs (×10⁹) | A100 上理论最大 tokens/s (MFU=100%) |
|------|------------------------|----------------------------------|
| 推理 | 1.69 | ~184,615 |
| 训练（含优化器） | ~5.07 | ~61,538 |
| 训练（不含优化器） | ~4.90 | ~63,673 |

若实测训练吞吐量为 45,000 tokens/s：

$$\text{MFU}_{\text{train}} = \frac{45000 \times 5.07 \times 10^9}{312 \times 10^{12}} \approx 0.73 = 73\%$$

---

## 9. 影响 MFU 的关键因素

1. **序列长度 $s$**：越长，GagedDeltaNet 优势越明显
2. **GagedDeltaNet 占比 $L_{gd}/L$**：占比越高，总 FLOPs 越低
3. **top-k 值 $k$**：影响 MoE FFN 的计算量
4. **GPU 利用率**：内存带宽、算力利用率、通信开销等
5. **Batch size $b$**：较大 batch size 可提升 GPU 算力利用率
6. **训练 vs 推理**：训练 FLOPs 约为推理的 3 倍

