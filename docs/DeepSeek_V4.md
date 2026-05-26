# DeepSeek-V4：迈向高效百万Token上下文智能

**DeepSeek-AI**

research@deepseek.com

## 摘要

我们发布了 DeepSeek-V4 系列的预览版本，包括两个强大的混合专家（MoE）语言模型——**DeepSeek-V4-Pro**（1.6T参数，49B激活参数）和 **DeepSeek-V4-Flash**（284B参数，13B激活参数），两者均支持一百万Token的上下文长度。

DeepSeek-V4 系列在架构和优化方面进行了多项关键升级：

1. **混合注意力架构**：结合压缩稀疏注意力（CSA）和重度压缩注意力（HCA），提高长上下文效率
2. **流形约束超连接（mHC）**：增强传统残差连接
3. **Muon 优化器**：实现更快的收敛和更高的训练稳定性

我们在超过32T Token的多样化高质量数据上对两个模型进行预训练，随后通过全面的后训练流程进一步释放和增强它们的能力。

### 核心性能亮点

| 指标 | DeepSeek-V4-Pro vs V3.2 |
|------|------------------------|
| 单Token推理FLOPs (1M上下文) | 仅需 27% |
| KV Cache大小 (1M上下文) | 仅需 10% |

DeepSeek-V4-Pro-Max（最大推理努力模式）重新定义了开源模型的最新水平。模型检查点可在 https://huggingface.co/collections/deepseek-ai/deepseek-v4 获取。

---

## 目录

1. [引言](#1-引言)
2. [架构](#2-架构)
3. [通用基础设施](#3-通用基础设施)
4. [预训练](#4-预训练)
5. [后训练](#5-后训练)
6. [结论、局限性与未来方向](#6-结论局限性与未来方向)

---

## 1. 引言

推理模型（reasoning models）的出现建立了一种新的**测试时扩展（test-time scaling）**范式，推动大型语言模型（LLMs）取得了显著的性能提升。然而，这种扩展范式从根本上受到标准注意力机制（vanilla attention mechanism）**二次计算复杂度**的限制，这对超长上下文和推理过程构成了难以逾越的瓶颈。

与此同时，长周期场景和任务的涌现——从复杂的智能体工作流到大规模跨文档分析——也使得高效支持超长上下文成为未来进展的关键。虽然最近的开源努力在通用能力方面取得了进展，但这种处理超长序列的核心架构效率低下仍然是主要障碍。

为了打破超长上下文的效率壁垒，我们开发了 DeepSeek-V4 系列。通过架构创新，DeepSeek-V4 系列在处理超长序列的计算效率上实现了飞跃式提升，使得高效支持**百万Token的上下文长度**成为可能。

### 与 DeepSeek-V3 的关键差异

| 创新点 | 说明 |
|--------|------|
| 混合注意力 (CSA + HCA) | CSA 压缩 KV 缓存后执行稀疏注意力；HCA 更激进地压缩但保持密集注意力 |
| 流形约束超连接 (mHC) | 升级传统残差连接，增强信号传播稳定性 |
| Muon 优化器 | 更快收敛、更好的训练稳定性 |

### 核心评估结果摘要

- **知识**：DeepSeek-V4-Pro-Max 在 SimpleQA 和 Chinese-SimpleQA 上显著超越所有开源模型（+20个百分点），与 Gemini-3.1-Pro 的差距大幅缩小
- **推理**：优于 GPT-5.2 和 Gemini-3.0-Pro，略逊于 GPT-5.4 和 Gemini-3.1-Pro（约落后3-6个月）
- **智能体**：内部评估中超越 Claude Sonnet 4.5，接近 Opus 4.5 水平
- **长上下文**：在百万Token上下文窗口上甚至在学术基准中超越 Gemini-3.1-Pro
- **DeepSeek-V4-Flash-Max**：以高性价比架构实现与 GPT-5.2/Gemini-3.0-Pro 相当的推理性能

---

## 2. 架构

总体而言，DeepSeek-V4 系列保留了 Transformer 架构和多Token预测（MTP）模块，同时在 DeepSeek-V3 的基础上引入了几项关键升级：

1. 引入**流形约束超连接mHC**来强化传统残差连接
2. 设计**混合注意力架构**，通过 CSA 和 HCA 大幅提高长上下文效率
3. 采用 **Muon** 作为优化器

对于混合专家（MoE）组件，我们仍然采用 DeepSeekMoE 架构。多Token预测（MTP）配置保持与 DeepSeek-V3 完全一致。

### 2.1 继承自 DeepSeek-V3 的设计

#### 混合专家（Mixture-of-Experts）

DeepSeek-V4 系列采用 DeepSeekMoE 范式用于前馈网络（FFN），设置了细粒度的路由专家和共享专家。主要变更：

- **激活函数变更**：计算亲和性分数的激活函数从 `Sigmoid(·)` 改为 `Sqrt(Softplus(·))`
- **负载均衡**：采用无辅助损失策略 + 轻微的逐序列平衡损失
- **Hash 路由**：初始几个 Transformer 块中的密集 FFN 层替换为采用 Hash 路由的 MoE 层

#### 多Token预测（Multi-Token Prediction, MTP）

与 DeepSeek-V3 一样，DeepSeek-V4 系列也设置了 MTP 模块和目标，策略不做修改。

### 2.2 流形约束超连接（Manifold-Constrained Hyper-Connections, mHC）

DeepSeek-V4 系列引入 mHC 来强化相邻 Transformer 块之间的传统残差连接。核心思想是将残差映射约束到特定的**流形（manifold）**上，从而增强信号跨层传播的稳定性，同时保持模型的表达能力。

#### 标准超连接（Standard Hyper-Connections）

标准 HC 将残差流的宽度扩展 $n_{hc}$ 倍：

- 残差流形状：从 $\mathbb{R}^d$ → $\mathbb{R}^{n_{hc} \times d}$
- $d$ = 实际层输入的隐藏大小

设 $X_l = [x_{l,1}; \ldots; x_{l,n_{hc}}]^T \in \mathbb{R}^{n_{hc} \times d}$ 为第 $l$ 层之前的残差状态。HC 引入三个线性映射：

- 输入映射：$A_l \in \mathbb{R}^{1 \times n_{hc}}$
- 残差变换：$B_l \in \mathbb{R}^{n_{hc} \times n_{hc}}$
- 输出映射：$C_l \in \mathbb{R}^{n_{hc} \times 1}$

**残差状态更新公式**：

$$X_{l+1} = B_l X_l + C_l F_l(A_l X_l)$$

其中 $F_l$ 表示第 $l$ 层（如 MoE 层），其输入和输出形状均为 $\mathbb{R}^d$。

> **关键特性**：HC 将残差宽度与实际隐藏大小解耦，提供了互补的扩展轴，且计算开销极小（因为 $n_{hc}$ 通常远小于隐藏大小 $d$）。然而，当堆叠多层时训练会频繁出现数值不稳定。

#### 流形约束残差映射

mHC 的核心创新是将残差映射矩阵 $B_l$ 约束到**双重随机矩阵（Birkhoff 多面体）**的流形 $\mathcal{M}$ 上：

$$B_l \in \mathcal{M} := \{M \in \mathbb{R}^{n \times n} \mid M\mathbf{1}_n = \mathbf{1}_n,\; \mathbf{1}_n^T M = \mathbf{1}_n^T,\; M \geq 0\}$$

**约束效果**：

| 特性 | 说明 |
|------|------|
| 谱范数有界 | $\|B_l\|_2 \leq 1$，残差变换是非扩张的 |
| 数值稳定性 | 正向传播和反向传播期间都更稳定 |
| 封闭性 | 集合 $\mathcal{M}$ 在乘法下封闭，保证深层堆叠稳定性 |

输入变换 $A_l$ 和输出变换 $C_l$ 也通过 Sigmoid 函数被约束为非负且有界，避免信号抵消风险。

#### 动态参数化

三个线性映射的参数动态生成，分解为：
- **动态（依赖输入）组件**
- **静态（不依赖输入）组件**

给定输入 $X_l \in \mathbb{R}^{n_{hc} \times d}$，首先展平并归一化：

$$\hat{X}_l = \text{RMSNorm}(\text{vec}(X_l)) \in \mathbb{R}^{1 \times n_{hc}d}$$

然后按照标准 HC 生成无约束原始参数，再通过以下方式施加约束：

| 映射 | 约束方法 |
|------|----------|
| 输入映射 $A_l$ | Sigmoid 函数确保非负性和有界性 |
| 输出映射 $C_l$ | Sigmoid 函数确保非负性和有界性 |
| 残差映射 $\tilde{B}_l$ | Sinkhorn-Knopp 算法投影到双重随机矩阵流形 $\mathcal{M}$ |

Sinkhorn-Knopp 算法迭代过程（选择 $t_{max} = 20$）：

$$M^{(t)} = T_r(T_c(M^{(t-1)}))$$

其中 $T_r$ 和 $T_c$ 分别表示行归一化和列归一化操作。

### 2.3 CSA 与 HCA 混合注意力

当上下文长度达到极端规模时，注意力机制成为模型的主要计算瓶颈。DeepSeek-V4 设计了两种高效注意力架构：

| 注意力类型 | 缩写 | 特点 |
|-----------|------|------|
| **压缩稀疏注意力** | CSA | 每 $m$ 个Token压缩为1个KV条目 + 稀疏注意力 |
| **重度压缩注意力** | HCA | 更大压缩率 $m' (\gg m)$ + 密集注意力 |

两者交错配置使用，大幅降低长文本场景下注意力的计算成本。

#### 2.3.1 压缩稀疏注意力（Compressed Sparse Attention, CSA）

**核心流程**：

```
输入隐藏状态 H ∈ R^(n×d)
       ↓
计算两组 KV 条目 (Ca, Cb) 及压缩权重 (Za, Zb)
       ↓
每 m 个 KV 条目 → 压缩为 1 个 C^Comp 条目（重叠压缩）
       ↓
闪电索引器 (Lightning Indexer): top-k 稀疏选择
       ↓
共享键值 MQA 核心注意力
       ↓
分组输出投影
```

**压缩键值条目详细过程**：

设 $H \in \mathbb{R}^{n \times d}$ 为输入隐藏状态序列：

1. 计算两组 KV 条目及权重：
   - $C_a = H \cdot W_a^{KV}, \quad C_b = H \cdot W_b^{KV}$
   - $Z_a = H \cdot W_a^Z, \quad Z_b = H \cdot W_b^Z$

2. 每 $m$ 个 KV 条目压缩为 1 个（重叠压缩，实际压缩率为 $1/m$）：

$$[S_a^{mi:m(i+1)-1}; S_b^{m(i-1):mi-1}] = \text{Softmax}_{row}([Z_a^{mi:m(i+1)-1} + B_a; Z_b^{m(i-1):mi-1} + B_b])$$

$$C_i^{\text{Comp}} = \sum_{j=mi}^{m(i+1)-1} S_j^a \odotimes C_j^a + \sum_{j=m(i-1)}^{mi-1} S_j^b \odotimes C_j^b$$

**闪电索引器（稀疏选择）**：

对于查询Token $t$，以低秩方式生成索引器查询：

$$c_Q^t = h_t \cdot W^{DQ}, \quad [q_{I,t,1}; \ldots; q_{I,t,n_{Ih}}] = q_I^t = c_Q^t \cdot W^{IUQ}$$

索引分数计算（含 ReLU 非线性）：

$$I_{t,s} = \sum_{h=1}^{n_{Ih}} w_{I,t,h} \cdot \text{ReLU}(q_{I,t,h} \cdot K_I^s{}^{\text{Comp}})$$

Top-k 选择器保留 $k$ 个最相关的压缩 KV 条目用于核心注意力。

**共享键值 MQA**：

从压缩潜向量 $c_Q^t$ 生成注意力查询：

$$[q_{t,1}; \ldots; q_{t,n_h}] = q_t = c_Q^t \cdot W^{UQ}$$

对每个查询头执行 MQA：

$$o_{t,i} = \text{CoreAttn}(\text{query}=q_{t,i},\; \text{key}=C_t^{\text{Sprs}},\; \text{value}=C_t^{\text{Sprs}})$$

**分组输出投影**：将 $n_h$ 个输出分为 $g$ 组，每组独立投影到中间维度 $d_g$，最终合并为 $d$ 维输出。

#### 2.3.2 重度压缩注意力（Heavily Compressed Attention, HCA）

HCA 的压缩策略与 CSA 类似，但关键区别：

| 特征 | CSA | HCA |
|------|-----|-----|
| 压缩率 | $m$（较小，如4） | $m'$（很大，如128） |
| 重叠压缩 | 是 | 否 |
| 注意力类型 | 稀疏（top-k） | 密集（全量） |

HCA 压缩公式（无重叠）：

$$S_i^{m'i:m'(i+1)-1} = \text{Softmax}_{row}(Z_{m'i:m'(i+1)-1} + B)$$

$$C_i^{\text{Comp}} = \sum_{j=m'i}^{m'(i+1)-1} S_j \odotimes C_j$$

后续同样采用共享键值 MQA 和分组输出投影。

#### 2.3.3 其他细节

**查询和键值条目归一化**：核心注意力前对每个查询头和压缩 KV 条目头执行额外 RMSNorm，防止注意力 logit 爆炸。

**部分旋转位置嵌入（RoPE）**：对注意力查询、KV 条目和核心注意力输出的**最后64维**应用 RoPE。对输出还应用位置 $-i$ 的 RoPE 以携带相对位置信息。

**滑动窗口注意力分支**：为建模局部依赖关系，引入补充的滑动窗口注意力分支（窗口大小 $n_{win}=128$），处理最近 $n_{win}$ 个未压缩Token。

**注意力汇（Attention Sink）**：设置可学习的汇 logit $\{z'_1, \ldots, z'_{n_h}\}$，第 $h$ 头的 $\text{Exp}(z'_h)$ 加到注意力分数分母，允许调整总注意力分数不为1。

#### 2.3.4 效率讨论

DeepSeek-V4 系列注意力模块的效率优化策略：

| 优化技术 | 效果 |
|----------|------|
| 混合存储格式（RoPE用BF16 + 其余用FP8） | KV缓存减少近50% |
| 闪电索引器 FP4 计算 | 加速极端长上下文注意力 |
| 较小的注意力 top-k | 提高中短文本效率 |
| CSA/HCA 压缩注意力 | 大幅减少KV缓存和计算FLOPs |

**对比基线（BF16 GQA, 头维度128）**：

| 上下文长度 | DeepSeek-V4-Pro | DeepSeek-V4-Flash |
|-----------|-------------------|-------------------|
| 1M Token KV缓存 | 约 V3.2 的 **2%** | 约 V3.2 的 **1.4%** |
| 1M Token FLOPs | V3.2 的 **27%** | V3.2 的 **10%** |

### 2.4 Muon 优化器

我们在 DeepSeek-V4 系列的大多数模块中采用 **Muon 优化器**，因为它具有更快的收敛速度和更好的训练稳定性。

#### 基本配置

| 组件 | 优化器 |
|------|--------|
| embedding 模块 | AdamW |
| 预测头模块 | AdamW |
| mHC 静态偏置和门控因子 | AdamW |
| RMSNorm 权重 | AdamW |
| **其他所有模块** | **Muon** |

Muon 配置：动量 $\mu = 0.95$，权重衰减 $\lambda = 0.1$，更新矩阵 RMS 重缩放因子 $\gamma = 0.18$（复用 AdamW 学习率）。

#### 混合 Newton-Schulz 迭代

Newton-Schulz 迭代用于正交化，共 **10次迭代**分两个阶段：

| 阶段 | 迭代次数 | 系数 $(a, b, c)$ | 目的 |
|------|---------|------------------|------|
| 快速收敛 | 前8步 | $(3.4445, -4.7750, 2.0315)$ | 使奇异值接近1 |
| 精确稳定 | 后2步 | $(2, -1.5, 0.5)$ | 奇异值精确稳定在1 |

迭代公式：

$$M_k = a M_{k-1} + b(M_{k-1} M_{k-1}^T)M_{k-1} + c(M_{k-1} M_{k-1}^T)^2 M_{k-1}$$

#### 避免 QK-Clip

由于 DeepSeek-V4 的注意力架构允许直接在注意力查询和 KV 条目上应用 RMSNorm，有效防止注意力 logit 爆炸，因此**不需要 QK-Clip 技术**。

---

## 3. 通用基础设施

### 3.1 专家并行中的细粒度通信-计算重叠

混合专家（MoE）通过专家并行（EP）加速，但 EP 需要复杂节点间通信。我们的方案：**将通信和计算融合到单个流水线内核**。

#### MoE 层四阶段分解

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Dispatch   │ →  │  Linear-1   │ →  │  Linear-2   │ →  │   Combine    │
│ (通信受限)   │    │ (计算受限)   │    │ (计算受限)   │    │ (通信受限)   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
     ↑ 通信延迟可隐藏于计算之下 ↑
```

**关键洞察**：单 MoE 层内通信总时间 < 计算总时间，融合后计算仍是主导瓶颈。

#### 细粒度波次调度（Wave-based Scheduling）

将专家分割为**波次（waves）**，每个波次包含一小部分专家：

```
稳态并发执行：
┌──────────────────────────────────────────────────┐
│ 当前波次计算 │ 下一波次Token传输 │ 已完成专家结果发送 │
└──────────────────────────────────────────────────┘
```

**性能收益**（验证于 NVIDIA GPU 和华为 Ascend NPU）：

| 场景 | 加速比 |
|------|--------|
| 通用推理工作负载 | **1.50 ~ 1.73×** |
| RL rollout / 高速Agent服务 | 高达 **1.96×** |

已作为 **MegaMoE** 组件开源于 DeepGEMM：https://github.com/deepseek-ai/DeepGEMM/pull/304

#### 硬件设计建议

| 方面 | 建议 |
|------|------|
| 计算-通信比 | 目标平衡点：$C/B \leq 2d = 6144$ FLOPs/Byte（每GBps带宽支撑6.1TFLOP/s计算） |
| 功耗预算 | 极端内核融合使计算/内存/网络同时高负载，需预留足够功耗余量 |
| 通信原语 | 低延迟跨GPU信号可使push方案可行 |
| 激活函数 | 建议替换 SwiGLU 为低成本逐元素激活（无指数/除法运算） |

### 3.2 使用 TileLang 进行灵活高效的内核开发

采用 **TileLang**（领域特定语言DSL）开发融合内核，替代数百个细粒度 Torch ATen 算子。

#### 核心优势

| 特性 | 说明 |
|------|------|
| Host Codegen | 将主机端逻辑移入生成代码，CPU验证开销从数十μs降至 <1μs |
| Z3 SMT求解器 | 形式化整数分析，解锁向量化等高级优化 |
| 位级reproducibility | 对齐NVCC变换规则，实现与手写CUDA位级一致输出 |

### 3.3 高性能批量不变且确定性的内核库

为实现端到端的**位级批量不变确定性内核**：

**批量不变性解决方案**：

| 挑战 | 解决方案 |
|------|----------|
| 注意力 split-KV 导致wave-quantization | 双内核策略（单SM高吞吐 + 多SM低延迟） |
| cuBLAS无法批量不变 | 用 DeepGEMM 端到端替代 |

**确定性解决方案**：

| 非确定性来源 | 解决方案 |
|-------------|----------|
| 注意力反向原子加 | 每SM分配独立累冲区 + 全局确定性求和 |
| MoE反向写入冲突 | rank内token顺序预处理 + rank间缓冲区隔离 |
| mHC小批量split-k | 分离输出各split部分 + 后续kernel确定性归约 |

### 3.4 训练框架

#### 3.4.1 Muon 的高效实现

**ZeRO bucket 分配混合策略**：

| 参数类型 | 策略 |
|----------|------|
| 密集参数 | 限制ZeRO并行度上限 + 背包算法分配 + <10% padding开销 |
| MoE参数 | 独立优化每个专家 + 无并行度限制 + 可忽略padding |
| 额外优化 | 同形状参数自动合并批量化NS迭代 + BF16随机量化梯度 + 两阶段reduce-scatter |

#### 3.4.2 mHC 的高效且节省内存的实现

| 优化策略 | 效果 |
|----------|------|
| 融合内核 | 直接计算开销降低 |
| 选择性重算检查点 | 平衡内存节省与计算开销 |
| DualPipe 1F1B调整 | 适应增加的管道通信 |

**整体开销**：mHC wall-time 开销仅占重叠 1F1B 管道阶段的 **6.7%**

#### 3.4.3 长上下文注意力的上下文并行

传统CP沿序列划分，但压缩注意力带来两挑战：
1. 各rank压缩后长度不同（尾部Token不足 $m$ 个被丢弃）
2. 压缩需要 $m$ 个连续KV条目，可能跨越CP边界

**两阶段通信方案**：

```
阶段1: rank i 发送最后 m 个未压缩KV → rank i+1
       rank i+1 合并接收条目与本地压缩 → 固定长度 s/m + 1 个压缩条目

阶段2: all-gather 收集所有本地压缩条目
       融合 select-and-pad 算子重组为完整压缩KV集合（总长度 cp_size × s/m）
```

#### 3.4.4 扩展自动微分以支持灵活的激活检查点

**张量级激活检查点机制**：

- 开发者只需实现正向传播 + 标注需要检查点的张量
- 框架通过 TorchFX 追踪计算图，自动识别最小重计算子图
- **零GPU内存复制**：直接释放标注张量GPU内存，复用重算张量的存储指针
- **自动去重**：跟踪底层存储指针，对共享存储的张量自动去重重算

### 3.5 推理框架

#### 3.5.1 KV 缓存结构与管理

**异构KV条目类型**：

| 来源 | 类型 | 特点 |
|------|------|------|
| CSA | 压缩KV + Indexer KV | 不同嵌入维度，更新规则复杂 |
| HCA | 压缩KV | 大压缩率 |
| SWA | 未压缩KV | 独立缓存策略，固定窗口大小 |
| CSA/HCA尾部 | 未压缩状态 | 等待压缩的缓冲Token |

**管理策略**：

| 挑战 | 解决方案 |
|------|----------|
| 多样化缓存策略（特别是SWA） | 状态缓存：SWA + 未压缩尾部视为状态空间模型，预分配固定大小池 |
| 高性能内核对齐要求 | 稀疏注意力内核协同设计：每块Token数可为 lcm(m, m') 的任意倍数 |

#### 3.5.2 磁盘 KV 缓存存储

**三种 SWA 存储策略**（存储-计算权衡）：

| 策略 | 存储开销 | 计算冗余 | 适用场景 |
|------|----------|----------|----------|
| **全量SWA缓存** | 高（8×压缩KV） | 零 | SSD存储系统不适用（写密集模式） |
| **周期检查点** | 可调 | 可调（重算尾部） | 通用场景推荐 |
| **零SWA缓存** | 零 | 高（重算 $n_{win} \cdot L$ tokens） | 存储极度敏感场景 |

CSA/HCA 压缩KV：直接全量存储到磁盘，命中时直接读取复用。

---

## 4. 预训练

### 4.1 数据构建

在 DeepSeek-V3 基础上构建更多样化、更高质量、更长有效上下文的训练语料库：

| 数据类别 | 策略 |
|----------|------|
| 网络数据 | 过滤去除批量自动生成/模板化内容，防模型崩溃 |
| 数学/编程 | 核心组件，mid-training融入Agent数据增强编码能力 |
| 多语言 | 更大语料库，提高长尾知识捕获 |
| **长文档**（重点） | 优先科学论文、技术报告等高学术价值材料 |

**预训练语料规模**：**>32T Token**

### 4.2 预训练配置

#### 4.2.1 模型配置对比

| 参数 | DeepSeek-V4-Flash | DeepSeek-V4-Pro |
|------|-------------------|-----------------|
| Transformer层数 | **43** | **61** |
| 隐藏维度 $d$ | **4096** | **7168** |
| 前两层注意力 | 纯滑动窗口 | HCA |
| 后续层注意力 | CSA/HCA交错 | CSA/HCA交错 |
| CSA压缩率 $m$ | 4 | 4 |
| HCA压缩率 $m'$ | 128 | 128 |
| 查询头数 $n_h$ | 64 | **128** |
| 头维度 $c$ | 512 | 512 |
| 查询压缩维度 $d_c$ | 1024 | 1536 |
| 输出投影组数 $g$ | 8 | 16 |
| 中间输出维度 $d_g$ | 1024 | 1024 |
| 滑动窗口大小 $n_{win}$ | 128 | 128 |
| MoE：共享专家数 | 1 | 1 |
| MoE：路由专家数 | **256** | **384** |
| 专家中间维度 | 2048 | 3072 |
| 激活专家数/Token | 6 | 6 |
| MTP深度 | 1 | 1 |
| mHC扩展因子 $n_{hc}$ | 4 | 4 |
| Sinkhorn-Knopp迭代次数 | 20 | 20 |
| **总参数** | **284B** | **1.6T** |
| **激活参数/Token** | **13B** | **49B** |

> 注：前3个MoE层使用 Hash 路由策略

#### 4.2.2 训练配置

| 配置项 | Flash | Pro |
|--------|-------|-----|
| 优化器（大部分） | Muon | Muon |
| 优化器（embedding/预测头/RMSNorm） | AdamW | AdamW |
| AdamW: $(\beta_1, \beta_2, \epsilon)$ | (0.9, 0.95, 1e-20) | (0.9, 0.95, 1e-20) |
| AdamW: weight_decay | 0.1 | 0.1 |
| Muon: 动量 | 0.95 | 0.95 |
| Muon: weight_decay | 0.1 | 0.1 |
| Muon: RMS重缩放 $\gamma$ | 0.18 | — |
| **训练Token数** | **32T** | **33T** |
| 最大批量大小 | **75.5M** | **94.4M** |
| 峰值学习率 | $2.7 \times 10^{-4}$ | $2.0 \times 10^{-4}$ |
| 结束学习率 | $2.7 \times 10^{-5}$ | $2.0 \times 10^{-5}$ |
| 预热步数 | 2000 | 2000 |
| 序列长度进度 | 4K → 16K → 64K → **1M** | 4K → 16K → 64K → **1M** |
| 稀疏注意力引入时机 | 64K长度开始 | 64K长度开始（更长预热阶段） |
| 无辅助loss偏置更新速度 | 0.001 | 0.001 |
| 平衡损失权重 | 0.0001 | 0.0001 |
| MTP损失权重 | 0.3（大部分）/ 0.1（衰减后） | 0.3（大部分）/ 0.1（衰减后） |

#### 4.2.3 缓解训练不稳定性

训练万亿参数MoE模型面临重大稳定性挑战。两种实用技术：

##### 预期路由（Anticipatory Routing）

**核心思想**：解耦主干网络和路由网络的同步更新

$$\text{第 } t \text{ 步:}\quad \text{特征计算使用 } \theta_t,\; \text{路由索引使用 } \theta_{t-\Delta t}$$

**工程优化**：
- 预计算路由索引仅需单次前向传播
- 管道执行与EP通信精心重叠编排 → 额外wall-time开销约 **20%**
- 自动检测机制：loss spike时触发短期回滚+激活Anticipatory Routing，稳定后恢复标准训练

##### SwiGLU 钳位（Clamping）

在整个训练过程中：

| 分量 | 钳位范围 |
|------|----------|
| 线性分量 | **[-10, 10]** |
| 门控分量上界 | **10** |

效果：有效消除异常值，大幅稳定训练，**不损害性能**。

### 4.3 评估

#### 4.3.1 评估基准覆盖

| 维度 | 基准测试 |
|------|----------|
| **世界知识** | AGIEval, C-Eval, CMMLU, MMLU, MMLU-Redux, MMLU-Pro, MMMLU, MultiLoKo, Simple-QA verified, SuperGPQA, FACTS Parametric, TriviaQA |
| **语言理解与推理** | BBH, DROP, HellaSwag, CLUEWSC, WinoGrande |
| **编码与数学** | BigCodeBench, HumanEval, GSM8K, MATH, MGSM, CMath |
| **长上下文** | LongBench-V2 |

#### 4.3.2 基础模型评估结果

**DeepSeek-V4-Flash-Base vs DeepSeek-V3.2-Base**：

尽管激活参数和总参数都少得多，V4-Flash-Base 在广泛基准上**优于** V3.2-Base，特别是在世界知识和长上下文任务上优势明显——证明架构改进和数据质量的优势。

**DeepSeek-V4-Pro-Base**：

在几乎所有类别建立对 V3.2-Base 和 V4-Flash-Base 的**近乎普遍优势**，达到 DeepSeek 基础模型的**新性能高度**：

- 知识密集型评估：**大幅增益**
- 长上下文理解：**显著推进**
- 推理和代码基准：**全面超越前辈**

→ 确认为 **DeepSeek 系列最强基础模型**

---

## 5. 后训练

### 5.1 后训练管道

**关键方法论替换**：混合RL阶段完全被 **在线策略蒸馏（On-Policy Distillation, OPD）** 取代。

#### 5.1.1 专家训练（两阶段范式）

```
┌─────────────────────────────────────────────────────────┐
│                   专家训练流程                          │
│                                                         │
│  ┌──────────┐    ┌───────────┐    ┌──────────────────┐  │
│  │ 基础模型  │ →  │ SFT微调   │ →  │ GRPO强化学习    │  │
│  │          │    │(高质量数据)│    │(领域特定奖励)   │  │
│  └──────────┘    └───────────┘    └──────────────────┘  │
│         ↓                    ↓                    ↓        │
│   数学专家              编程专家             Agent专家     │
│   推理专家              ...                  ...        │
│                                                         │
│         ┌──────────────────────────────────────────┐   │
│         │      OPD: 多教师 → 单一学生模型        │   │
│         └──────────────────────────────────────────┘   │
│                         ↓                           │
│                  DeepSeek-V4-Pro/Flash            │
└─────────────────────────────────────────────────┘
```

##### 三种推理努力模式

| 模式 | 特点 | 典型用例 | 响应格式 |
|------|------|----------|----------|
| **Non-think** | 快速、直觉响应 | 日常任务、应急反应 | `summary` |
| **Think** | 有意识逻辑分析 | 复杂问题求解、规划 | `thinking tokens` + `summary` |
| **Think Max** | 推理推至极限 | 探索模型推理能力边界 | 系统提示 + `thinking tokens` + `summary` |

**Think Max 系统提示注入**：

> **Reasoning Effort: Absolute maximum with no shortcuts permitted.**
> You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause...

##### 生成式奖励模型（GRM）

**摒弃传统标量奖励模型**，采用 GRM 评估难验证任务：

- 策划规则引导的RL数据
- **直接对GRM本身应用RL优化**
- Actor网络原生充当GRM → 评估能力与生成能力联合优化
- 仅需少量多样化人类注释即可泛化到复杂任务

##### 工具调用 Schema

引入新 schema：特殊 `"|DSML|"` token + XML格式工具调用

```xml
<|DSML|tool_calls>
<|DSML|invoke name="$TOOL_NAME">
<|DSML|parameter name="$PARAM" string="true|false">$VALUE</|DSML|parameter>
</|DSML|invoke>
</|DSML|tool_calls>
```

实验表明XML格式有效**减少转义失败和工具调用错误**。

##### 交错思考（Interleaved Thinking）

利用百万Token上下文窗口优化Agent环境中的思考管理：

| 场景 | 策略 |
|------|------|
| **工具调用场景** | 保留**完整**推理历史跨所有轮次（包括用户消息边界） |
| **一般对话场景** | 新用户消息到达时丢弃之前推理内容（保持简洁） |

##### 快速指令（Quick Instruction）

消除辅助任务小模型的冗余prefilling：

```markdown
...<|User|>{prompt}<|action|>      ← 判断是否需要搜索
<|User|>{prompt}<|title|>       ← 生成对话标题
<|User|>{prompt}<|query|>        ← 生成搜索查询
...
```

→ 直接复用已有KV cache，**完全避免冗余prefilling**，显著降低TTFT

#### 5.1.2 在线策略蒸馏（OPD）

**OPD目标函数**（多教师）：

$$L_{\text{OPD}}(\theta) = \sum_{i=1}^{N} w_i \cdot D_{\text{KL}}(\pi_\theta \| \pi_{E_i})$$

**关键技术决策：全词汇logit蒸馏**

| 方案 | 优点 | 缺点 |
|------|------|------|
| token级KL估计（复用RL框架） | 资源高效 | 高方差梯度估计，训练不稳定 |
| **全词汇logit蒸馏（本文采用）** | **稳定梯度估计，忠实蒸馏教师知识** | 工程挑战更大 |

> 超过**十多个**覆盖各领域的教师模型用于蒸馏单一学生模型

### 5.2 后训练基础设施

#### 5.2.1 FP4 量化感知训练（QAT）

FP4 (MXFP4) 量化应用于：

1. **MoE专家权重** → GPU内存占用主要来源
2. **CSA索引器的QK路径** → QK激活缓存/加载/乘法全程FP4，加速长上下文注意力分数计算
3. **索引分数** $I_{:,:}$：FP32 → **BF16** → top-k选择器**2×加速**，保持99.7%召回率

**FP4→FP8反量化无损**的关键条件：FP8 (E4M3) 比FP4 (E2M1) 多2个指数位 → 更大动态范围可完全吸收FP4子块的细粒度scale信息。

#### 5.2.2 全词汇OPD的高效教师调度

| 挑战 | 解决方案 |
|------|----------|
| 万亿参数教师存储 | 卸载到集中分布式存储 + ZeRO-like按需加载 |
| 全词汇logit显式物化内存爆炸 | 仅缓存最后一层教师hidden state → 按需通过预测头重建logits |
| 教师预测头GPU内存 | 按教师index排序样本 → 每mini-batch仅加载1个教师头 |
| 异步加载/卸载 | 所有参数和hidden state操作后台异步执行，不阻塞关键路径 |
| KL散度计算 | 专用TileLang kernel加速 + 减少动态内存分配 |

#### 5.2.3 可抢占和容错的Rollout服务

**Token粒度Write-Ahead Log (WAL)**：

- 每生成1个新Token立即追加到对应请求的WAL
- 抢占时：暂停引擎 + 保存未完成请求的KV cache
- 恢复时：持久化WAL + 保存的KV cache继续解码
- 致命硬件错误：用持久化Token重跑prefill重建KV cache

> **重要**：从头重新生成是不正确的（引入长度偏差——短响应更容易在中断中存活）

#### 5.2.4 百万Token上下文的RL框架扩展

**Rollout数据格式分解**：

| 类型 | 内容 | 加载方式 |
|------|------|----------|
| 轻量级元数据 | 全局shuffle + packing布局计算 | 一次性全部加载 |
| 重型per-token字段 | 实际Token数据 | 共享内存data loader按mini-batch粒度加载即释放 |

设备上mini-batch数量根据工作负载动态确定 → 计算吞吐量与I/O重叠的高效权衡。

#### 5.2.5 Agent AI沙箱基础设施：DSec

**DeepSeek Elastic Compute (DSec)** — 生产级沙箱平台：

```
┌─────────────────────────────────────────────────────────┐
│                    DSec 架构                          │
│                                                         │
│  ┌──────────┐    ┌───────────┐    ┌────────────────┐  │
│  │ Apiserver │ ←→ │   Edge    │ ←→ │   Watcher    │  │
│  │ (API网关) │ RPC│ (每主机代理)│ RPC│ (集群监视器) │  │
│  └──────────┘    └───────────┘    └────────────────┘  │
│       ↕               ↕                    ↕           │
│   3FS 分布式文件系统                              │
│                                                         │
│  四种执行基底（统一Python SDK - libdsec）：          │
│  • Function Call → 预热容器池（无冷启动）          │
│  • Container → Docker兼容 + EROFS按需加载          │
│  • microVM → Firecracker VM级隔离                     │
│  • fullVM → QEMU 支持任意Guest OS                 │
│                                                         │
│  单集群管理 **数十万** 并发沙箱实例                   │
└─────────────────────────────────────────────────┘
```

**四大核心设计**：

| 设计 | 说明 |
|------|------|
| **统一接口下的四种执行基底** | 切换仅需改参数，命令执行/文件传输/TTY访问统一API |
| **分层存储快速镜像加载** | Container用EROFS层，microVM用overlaybd链式快照，毫秒级恢复 |
| **大规模并发密度优化** | 消除虚拟环境重复页缓存 + 容器运行时自旋锁竞争缓解 |
| **轨迹日志与抢占安全恢复** | 全局有序轨迹日志 → client快进/细粒度溯源/确定性回放 |

### 5.3 标准基准评估

#### 5.3.1 评估设置概览

| 类别 | 数据集 |
|------|--------|
| **知识&推理** | MMLU-Pro, GPQA, HLE, SimpleQA-Verified, Chinese-SimpleQA, LiveCodeBench-v6, Codeforces, HMMT 2026 Feb, Apex, Apex Shortlist, IMOAnswerBench, PutnamBench |
| **编码** | LiveCodeBench-v6 + 内部Codeforces (14场比赛114题, 2025.5-11) |
| **1M-Token上下文** | OpenAI MRCR + CorpusQA |
| **Agent** | Terminal Bench 2.0, SWE-Verified, SWE Multilingual, SWE-Pro, BrowseComp, MCPAtlas Public, GDPval-AA, Tool-Decathlon |

**代码Agent设置**：内部框架（bash工具 + 文件编辑工具），最多500交互步，最大上下文512K tokens。

#### 5.3.2 评估结果

##### 知识

DeepSeek-V4-Pro-Max 在开源LLM中建立**新的SOTA**：

| 基准 | 表现 |
|------|------|
| SimpleQA-Verified | **超越所有开源基线20个百分点** |
| MMLU-Pro / GPQA / HCE | 略优Kimi/GLM，略逊前沿闭源模型 |
| Gemini-3.1-Pro | 仍有差距但已大幅缩小 |

##### 推理

| 模型 | 代码竞赛 | 形式化数学 |
|------|----------|-----------|
| **V4-Pro-Max** | 匹配 **GPT-5.4**（首次开放模型匹敌闭源！） | SOTA（agentic + compute-intensive均优） |
| **V4-Flash-Max** | 超越 K2.6-Thinking | 强劲 |

**Codeforces排名**：V4-Pro-Max 当前人类候选人排名第 **23** 位。

##### Agent

| 任务类型 | 表现 |
|----------|------|
| 代码Agent（SWE/Terminal-Bench等） | 匹配 K2.6 / GLM-5.1，仍落后闭源竞品 |
| MCPAtlas / Toolathlon（广泛工具/MCP服务） | **出色泛化能力**，非仅内部框架强 |

##### 1M-Token上下文

| 任务 | vs Gemini-3.1-Pro | vs Claude Opus 4.6 |
|------|---------------------|--------------------|
| MRCR（上下文检索） | **超越** | 仍落后 |
| CorpusQA（真实场景类） | **超越** | — |

检索性能在 **128K 内高度稳定**，100万Token时仍非常强。

##### 推理努力模式对比

Max模式（更长上下文 + 减少长度惩罚）在最具挑战性任务上**一致优于** High模式。通过扩展测试时计算，V4系列相比前代实现**实质性改进**；在HLE等任务上 V4-Pro 展示出比 V3.2 **更高的Token效率**。

### 5.4 真实世界任务性能

#### 5.4.1 中文写作

**功能写作**（vs Gemini-3.1-Pro）：

| 指标 | 结果 |
|------|------|
| 总胜率 | **62.7% vs 34.1%** |
| 主要优势领域 | 报告(66.4%)、方案策划(62.2%)、技术文本(75.9%) |

**创意写作**（vs Gemini-3.1-Pro）：

| 维度 | 胜率 |
|------|------|
| 指令遵循 | **60.0%** |
| 写作质量 | **77.5%** |

> 但在高复杂度约束/多轮场景中，Claude Opus 4.5 仍保持优势（52.0% vs 45.9%）

#### 5.4.2 搜索

**智能体搜索 vs RAG**（DeepSeek-V4-Pro）：

| 难度 | Agent胜率 | RAG胜率 | 平局 |
|------|-----------|---------|------|
| 简单-客观问答 | 56.1% | 21.9% | 21.9% |
| 简单-主观问答 | 61.7% | 17.4% | 20.9% |
| 困难-客观问答 | 60.7% | 19.6% | 19.6% |
| 困难-主观问答 | **68.5%** | 14.7% | 16.8% |
| **总计** | **61.7%** | 18.3% | 20.0% |

成本：智能体搜索仅比标准RAG **略贵**。

#### 5.4.3 白领任务

30项高级中文专业任务，13个关键行业，四维人工评估（vs Opus-4.6-Max）：

| 维度 | V4-Pro-Max得分 | Opus-4.6-Max得分 |
|------|---------------|-----------------|
| **任务完成度** | **98.32** | 96.68 |
| **内容质量** | **88.88** | 78.00 |
| 指令遵循 | 87.76 | 88.88 |
| 格式美观度 | 76.68 | 72.68 |
| **整体不败率** | **63%** | — |

**V4-Pro-Max 优势**：主动预见隐含用户意图、提供补充见解和自验证步骤、擅长长文深度生成、严格遵循正式专业规范。

**不足**：偶尔忽略特定格式约束、长文本摘要能力待提升、演示文稿视觉设计有较大改进空间。

#### 5.4.4 代码Agent

内部R&D基准（~200任务 → 30题评估集，PyTorch/CUDA/Rust/C++）：

| 模型 | 通过率 |
|------|--------|
| Haiku 4.5 (Opus 4.6 Thinking) | 13% |
| Sonnet 4.5 (Opus 4.5 Thinking) | 47% |
| **DeepSeek-V4-Pro-Max** | **67%** |
| Opus 4.5 | 70% |
| Opus 4.6 | 80% |

**开发者调查**（N=85，日常Agent编码）：

| 选项 | 占比 |
|------|------|
| ✅ 是，可作为默认主要编码模型 | **52%** |
| 👍 倾向于是 | **39%** |
| ❌ 否 | <9% |

反馈：大多数任务满意，但有小错误/模糊提示误解/偶尔过度思考。

---

## 6. 结论、局限性与未来方向

### 总结

DeepSeek-V4 系列通过以下创新实现**百万Token上下文的高效原生支持**：

| 创新领域 | 核心贡献 |
|----------|----------|
| **混合注意力 (CSA+HCA)** | 长序列效率飞跃式提升 |
| **流形约束超连接 (mHC)** | 强化残差连接稳定性 |
| **Muon优化器** | 更快收敛、更好稳定性 |
| **基础设施优化** | 从训练到推理的全栈效率突破 |

**评估结果确认**：

- ✅ **DeepSeek-V4-Pro-Max**：重新定义开源模型SOTA，知识基准大幅领先，推理性能接近前沿闭源模型
- ✅ **DeepSeek-V4-Flash-Max**：高性价比架构达到与 GPT-5.2/Gemini-3.0-Pro 相当的性能
- ✅ **开启开源模型百万级长度上下文新时代**

### 局限性

| 方面 | 当前局限 |
|------|----------|
| **架构复杂度** | 为最小化风险保留许多已验证组件，架构相对复杂 |
| **训练稳定性理论** | Anticipatory Routing 和 SwiGLU Clamping 底层原理尚未充分理解 |
| **长上下文极限** | 128K后性能下降可见（虽仍强于竞品） |

### 未来方向

| 方向 | 计划 |
|------|------|
| **架构精简** | 提炼至最本质设计，在不牺牲性能前提下更加优雅 |
| **训练稳定性** | 积极研究基础问题 + 加强内部指标监控，走向更有原则性的稳定大规模训练方法 |
| **新维度稀疏性** | 探索更稀疏的嵌入模块等，进一步提高计算/内存效率 |
| **低延迟架构** | 持续研究使长上下文部署和交互更响应迅速 |
| **长周期Agent任务** | 持续迭代探索 |
| **多模态整合** | 将多模态能力融入模型 |
| **数据策展** | 开发更好的数据策展和合成策略，持续增强智能性/鲁棒性/实用性 |

---

## 附录

### A. 作者列表与致谢

#### A.1 作者列表（按字母顺序，* 表示已离队）

**研究与工程团队**（300+人）：

Anyi Xu, Bangcai Lin, Bing Xue, Bingxuan Wang*, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenhao Xu, Chenze Shao, Chong Ruan*, Conner Sun, Damai Dai, Daya Guo*, Dejian Yang, Deli Chen, Donghao Li, Erhang Li, Fangyun Lin, Fangzhou Yuan, Feiyu Xia, Fucong Dai, Guangbo Hao, Guanting Chen, Guoai Cao, Guolai Meng, Guowei Li, Han Yu, Han Zhang, Hanwei Xu, Hao Li, Haofen Liang, Haoling Zhang, Haoming Luo, Haoran Wei*, Haotian Yuan, Haowei Zhang*, Haowen Luo, Haoyu Chen, Haozhe Ji, Honghui Ding, Hongxuan Tang, Huanqi Cao, Huazuo Gao, Hui Qu, Hui Zeng, J. Yang, J.Q. Zhu, Jia Yu, Jialiang Huang, Jiasheng Ye, Jiashi Li, Jiaxin Xu, Jiewen Hu, Jin Yan, Jingchang Chen, Jingli Zhou, Jingting Xiang, Jingyang Yuan, Jingyuan Cheng, Jinhua Zhu, Jiping Yu, Joseph Sun, Jun Ran*, Junguang Jiang, Junjie Qiu, Junlong Li*, Junxiao Song, Kai Dong, Kaige Gao, Kang Guan, Kexing Zhou, Kezhao Huang*, Kuai Yu, Lean Wang, Lecong Zhang, Lei Wang, Li Zhang, Liang Zhao, Lihua Guo, Lingxiao Luo, Linwang Ma, Litong Wang, Liyu Cai, Liyue Zhang, Longhao Chen, M.S Di, M.Y Xu, Max Mei, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingxu Zhou, Panpan Huang, Peixin Cong, Peiyi Wang, Qiancheng Wang, Qihao Zhu, Qingyang Li, Qinyu Chen, Qiushi Du, Qiwei Jiang, Rui Tian, Ruifan Xu, Ruijie Lu, Ruiling Xu, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, Runqian Chen, Runqiu Yin, Runxin Xu, Ruomeng Shen, Ruoyu Zhang, S.H Liu, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shaofei Cai, Shaoheng Nie, Shaoyuan Chen, Shengding Hu, Shengyu Liu, Shiqiang Hu, Shirong Ma, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, Shuying Yu, Songyang Zhou, Tao Ni, Tao Yun, Tian Jin, Tian Pei, Tian Ye, Tianle Lin, Tianran Ji, Tianyi Cui, Tianyuan Yue, Tingting Yu, Tun Wang, W. Zhang, Wangding Zeng, Weilin Zhao, Wen Liu, Wenfeng Liang, Wenjie Pang, Wenjing Luo, Wenjing Yao, Wenjun Gao, Wenkai Yang, Wenlve Huang, Wentao Zhang, Wenting Ma, Xi Gao, Xiang He, Xiangwen Wang, Xiao Bi, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaokang Zhang, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xingchen Liu, Xingkai Yu, Xingyou Li, Xinyu Yang, Xu Chen, Xuanyu Wang, Xuecheng Su, Xuheng Lin, Xuwei Fu, Y.C Yan, Y.Q Wang*, Y.W Ma, Yanfeng Luo, Yang Zhang, Yanhong Xu, Yanru Ma, Yanwen Huang, Yao Li, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Qian, Yi Yu, Yichao Zhang, Yifan Ding, Yifan Shi, Yijia Wu, Yiliang Xiong, Ying He, Ying Zhou, Yingjia Luo, Yinmin Zhong, Yishi Piao, Yisong Wang, Yixiang Zhang, Yixiao Chen, Yixuan Tan, Yixuan Wei, Yiyang Ma, Yiyuan Liu, Yonglun Yang, Yongqiang Guo, Yongtong Wu, Yu Wu, Yuan Cheng, Yuan Ou, Yuanfan Xu, Yuanhao Li, Yuduan Wang, Yuhan Wu, Yuhao Meng, Yuheng Zou, YuKun Li, Yunfan Xiong, Yupeng Chen, Yuqian Cao, Yuqian Wang, Yushun Zhang, Yutong Lin, Yuxian Gu, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuxuan Zhou, Yuyang Zhou, Yuzhen Huang, Z.F Wu, Zehao Wang, Zehua Zhao, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhibin Gou, Zhicheng Ma, Zhigang Yan, Zhihong Shao, Zhixian Huang, Zhixuan Chen, Zhiyu Wu, Zhizhou Ren, Zhuoshu Li, Zhuping Zhang, Zian Xu, Zihao Wang, Zihui Gu, Zijia Zhu, Zilin Li, Zipeng Zhang*, Ziwei Xie, Ziyi Gao, Zizheng Pan, Zongqing Yao.

**业务与合规团队**：

Chenchen Ling, Chengyu Hou, Dongjie Ji, Fang Wei, Hengqing Zhang, Jia Luo, Jia Song, Jialu Cai, Jian Liang, Jiangting Zhou, Jieyu Yang, Jin Chen, Jingzi Zhou, Junmin Zheng, Leyi Xia, Linyan Zhu, Miaojun Wang, Mingming Li, Minmin Han, Ning Wang, Panpan Wang, Peng Zhang, Ruyi Chen, Shangmian Sun, Shaoqing Wu, W.L Xiao, Wei An, Wenqing Hou, Xianzu Wang, Xiaowen Sun, Xiaoxiang Wang, Xinyu Zhang, Xueyin Chen, Yao Xu, Yi Shao, Yiling Ma, Ying Tang, Yuehan Yang, Yuer Xu, Yukun Zha, Yuping Lin, Yuting Yan, Zekai Zhang, Zhe Ju, Zheren Gao, Zhongyu Wu, Zihua Qu, Ziyi Wan.

#### A.2 致谢

我们要感谢 Dolly Deng 和其他测试者对 DeepSeek-V4 系列模型能力提供的宝贵建议和反馈。

---

*本文档翻译自 DeepSeek-V4 技术报告原文*

