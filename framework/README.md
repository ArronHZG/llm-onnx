# 通讯原语与分布式训练并行策略详解

## 目录

1. [NCCL 通讯原语](#1-nccl-通讯原语)
2. [DeepSpeed ZeRO 并行](#2-deepspeed-zero-并行)
3. [PyTorch FSDP](#3-pytorch-fsdp)
4. [Megatron-LM 并行策略](#4-megatron-lm-并行策略)

## 1. NCCL 通讯原语

NCCL (NVIDIA Collective Communications Library) 是 NVIDIA 提供的高性能 GPU 间通讯库，提供了一系列集合通讯原语。

### 1.1 点对点通讯 (Point-to-Point)

#### Send / Recv

最基础的点对点通讯方式，用于两个 GPU 之间的数据传输。

```python
# 伪代码示例
if rank == 0:
    ncclSend(buffer, count, datatype, 1, comm, stream)  # 发送给 rank 1
elif rank == 1:
    ncclRecv(buffer, count, datatype, 0, comm, stream)  # 从 rank 0 接收
```

**通讯量**: 发送/接收的数据量 = `count × sizeof(datatype)`

### 1.2 广播 (Broadcast)

将一个 GPU 上的数据广播到所有 GPU。

```
Before:  [A]  [ ]  [ ]  [ ]
After:   [A]  [A]  [A]  [A]
          ↑
       root rank
```

**通讯量**: `(n-1) × data_size`，其中 n 为 GPU 数量

### 1.3 归约 (Reduce)

将所有 GPU 的数据归约（如求和、求平均）到一个目标 GPU。

```
Before:  [A1] [A2] [A3] [A4]
After:   [A1+A2+A3+A4] [ ] [ ] [ ]
                ↑
           root rank
```

**操作类型**: `ncclSum`, `ncclProd`, `ncclMax`, `ncclMin`, `ncclAvg`

**通讯量**: `(n-1) × data_size`

### 1.4 全归约 (AllReduce) ⭐

最常用原语，将所有 GPU 的数据归约后分发给所有 GPU。

```
Before:  [A1] [A2] [A3] [A4]
After:   [A]  [A]  [A]  [A]
         A = A1+A2+A3+A4
```

**通讯量**: `2 × (n-1)/n × data_size ≈ 2 × data_size` (使用 Ring 算法)

**应用场景**: 梯度同步（数据并行中最核心操作）

### 1.5 规约发散 (ReduceScatter)

先将所有数据归约，然后将结果分散到各个 GPU（每个 GPU 获得一部分）。

```
Before:  [A1] [A2] [A3] [A4]
         [B1] [B2] [B3] [B4]
         [C1] [C2] [C3] [C4]
         [D1] [D2] [D3] [D4]

After:   [A]  [B]  [C]  [D]
         A = A1+A2+A3+A4
         B = B1+B2+B3+B4
         C = C1+C2+C3+C4
         D = D1+D2+D3+D4
```

**通讯量**: `(n-1)/n × data_size`

### 1.6 全收集 (AllGather)

将所有 GPU 的数据收集到所有 GPU（每个 GPU 获得完整数据）。

```
Before:  [A]  [B]  [C]  [D]
After:   [ABCD] [ABCD] [ABCD] [ABCD]
```

**通讯量**: `(n-1)/n × total_data_size`

### 1.7 All-to-All

每个 GPU 将自己的数据分块发送给其他所有 GPU。

```
Before:  [A1|A2|A3|A4]  [B1|B2|B3|B4]  [C1|C2|C3|C4]  [D1|D2|D3|D4]
            ↓              ↓              ↓              ↓
After:   [A1|B1|C1|D1]  [A2|B2|C2|D2]  [A3|B3|C3|D3]  [A4|B4|C4|D4]
```

**通讯量**: `(n-1)/n × total_data_size`

**应用场景**: MoE 专家并行中的 token 分发

---

## 2. DeepSpeed ZeRO 并行

ZeRO (Zero Redundancy Optimizer) 通过分片优化器状态、梯度和参数来减少内存冗余。

### 2.1 ZeRO Stage 1：优化器状态分片

**原理**: 将优化器状态分片到不同 GPU，每个 GPU 只存储 1/N 的优化器状态。

```
Data Parallel Group: 4 GPUs

GPU0: [Param][Grad][OS0      ]  GPU1: [Param][Grad][OS1      ]
GPU2: [Param][Grad][OS2      ]  GPU3: [Param][Grad][OS3      ]
                           ↑
                    OS = Optimizer States
```

**通讯模式**:

1. 前向/反向传播：各 GPU 独立计算梯度
2. 梯度同步: `AllReduce`（与普通数据并行相同）
3. 参数更新: 每个 GPU 只更新自己负责的参数分片
4. 参数广播: `Broadcast`（可选，或者下一轮自己计算）

**通讯量**:

- 梯度同步: `2Ψ` (与 DP 相同，Ψ 为参数总量)
- 内存节省: 4Ψ → 4Ψ (无节省，但优化器状态从 2Ψ 降到 2Ψ/N)

### 2.2 ZeRO Stage 2：梯度 + 优化器状态分片

**原理**: 在 Stage 1 基础上，梯度也进行分片，每个 GPU 只存储自己负责参数对应的梯度。

```
GPU0: [Param][Grad0][OS0]  GPU1: [Param][Grad1][OS1]
GPU2: [Param][Grad2][OS2]  GPU3: [Param][Grad3][OS3]
```

**通讯模式**:

1. 反向传播时，使用 `ReduceScatter` 替代 `AllReduce` 进行梯度同步
2. 每个 GPU 只保留自己负责的梯度分片
3. 参数更新后使用 `AllGather` 广播完整参数

**通讯量**:

- 梯度同步: `Ψ`（使用 ReduceScatter，相比 AllReduce 减半）
- 参数广播: `Ψ`
- **总通讯量**: `2Ψ`（与 Stage 1 相同，但通讯模式不同）
- 内存节省: 2Ψ (梯度存储)

### 2.3 ZeRO Stage 3：参数 + 梯度 + 优化器状态分片

**原理**: 所有状态都分片，每个 GPU 只存储 1/N 的参数。

```
GPU0: [Param0][Grad0][OS0]  GPU1: [Param1][Grad1][OS1]
GPU2: [Param2][Grad2][OS2]  GPU3: [Param3][Grad3][OS3]
```

**通讯模式**:

1. **前向传播**:
    - 需要某层参数时，使用 `AllGather` 收集完整参数
    - 计算完成后释放非自己负责的参数
2. **反向传播**:
    - 同样需要 `AllGather` 收集参数
    - 计算梯度后使用 `ReduceScatter` 分发梯度
3. **参数更新**:
    - 各 GPU 更新自己的参数分片

**通讯量**（假设模型有 L 层）:

- 前向: L × `AllGather(Params)` ≈ L × Ψ
- 反向: L × `AllGather(Params)` + L × `ReduceScatter(Grad)` ≈ 2L × Ψ
- **总通讯量**: `3L × Ψ`（远大于 Stage 1/2）

**优化**: 使用参数预取 (prefetching) 和梯度累积来减少通讯次数

### 2.4 ZeRO 通讯量对比表

| Stage  | 优化器状态 | 梯度 | 参数 | 通讯量/Step | 单卡显存占用          |
|--------|-------|----|----|----------|-----------------|
| 0 (DP) | N×    | N× | N× | 2Ψ       | 16Ψ             |
| 1      | N×    | N× | 1× | 2Ψ       | 16Ψ - 2(N-1)Ψ/N |
| 2      | N×    | N× | 1× | 2Ψ       | 16Ψ - 4(N-1)Ψ/N |
| 3      | N×    | N× | N× | 3Ψ (或更多) | 16Ψ/N           |

---

## 3. PyTorch FSDP

### 3.1 FSDP 原理

Fully Sharded Data Parallel (FSDP) 是 PyTorch 官方实现的类 ZeRO-3 方案。

**核心思想**:

1. 将模型参数、梯度、优化器状态分片到所有 GPU
2. 使用 `all_gather` 临时收集参数进行计算
3. 计算完成后释放非本地参数

### 3.2 FSDP 通讯模式

**前向传播**:

```python
# 对每一层
all_gather(shard_params) -> full_params  # 收集完整参数
forward_compute(full_params)
release(full_params - local_shard)  # 释放非本地参数
```

**反向传播**:

```python
# 对每一层
all_gather(shard_params) -> full_params  # 重新收集参数
backward_compute(full_params, grad_output)
reduce_scatter(full_grads) -> shard_grads  # 分散梯度到各 GPU
release(full_params)
```

### 3.3 FSDP 通讯量计算

假设模型有 M 个参数，N 个 GPU，分为 L 个 FSDP 单元：

| 阶段 | 操作            | 通讯量             | 说明      |
|----|---------------|-----------------|---------|
| 前向 | AllGather     | L × M × (N-1)/N | 每层都需要收集 |
| 反向 | AllGather     | L × M × (N-1)/N | 重新收集参数  |
| 反向 | ReduceScatter | L × M × (N-1)/N | 梯度分散    |

**优化策略**:

- `backward_prefetch`: 预取下一层参数
- `forward_prefetch`: 预取前向参数
- `limit_all_gathers`: 限制并发 all_gather 数量

---

## 4. Megatron-LM 并行策略

符号定义

- Ψ: 模型参数量
- N: GPU 总数
- D: DP size
- EDP: EP 对应的 DP size
- T: TP size
- P: PP size
- E: EP size
- C: CP size
- b: batch size per DP rank
- s: sequence length
- h: hidden dimension
- L: 层数
- k: tok_k 数量

N = PP x DP x TP x CP
N = PP x EDP x ETP x EP

### 4.1 数据并行 (DP)

**原理**: 每个 GPU 存储完整模型，处理不同数据批次，通过通信同步梯度。

**通讯实现**（`megatron/core/distributed/param_and_grad_buffer.py`）:

Megatron 提供两种 DP 通讯模式：

#### 模式一：标准 DP（AllReduce）

```
Backward → AllReduce(梯度) → optimizer.step() → 下一轮 Forward
```

- 反向传播完成后，对梯度 buffer 执行 `AllReduce` 同步
- 所有 rank 持有完整梯度，各 rank 独立更新完整模型参数
- **通讯量**: `2Ψ`（一次 Ring AllReduce = ReduceScatter + AllGather，各占 Ψ）

#### 模式二：DistributedOptimizer（ReduceScatter + AllGather）

**实现位置**: `megatron/core/optimizer/distrib_optimizer.py`

```
Backward → ReduceScatter(梯度分片) → optimizer.step(仅更新本地分片)
        → AllGather(参数广播) → 下一轮 Forward
```

| 步骤   | 通讯原语          | 说明                  |
|------|---------------|---------------------|
| 梯度同步 | ReduceScatter | 各 rank 只保留自己负责的梯度分片 |
| 参数更新 | 无通讯           | 各 rank 独立更新本地参数分片   |
| 参数广播 | AllGather     | 下一轮 Forward 前广播完整参数 |

**两种模式的通讯流程对比**:

```
模式一：标准 DP（AllReduce）
─────────────────────────────────────────────────────────────────
 GPU 0        GPU 1        GPU 2        GPU 3
┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐
│完整模型│    │完整模型│    │完整模型│    │完整模型│
│完整梯度│    │完整梯度│    │完整梯度│    │完整梯度│
└──┬───┘    └──┬───┘    └──┬───┘    └──┬───┘
   │           │           │           │
   └──────┬────┴─────┬─────┴──────┬────┘
          │          │          │
          │    AllReduce(梯度)  │    ← backward 后立即执行
          │          │          │
   ┌──────┴────┬─────┴────┬─────┴──────┐
   ↓           ↓          ↓           ↓
 optimizer   optimizer  optimizer  optimizer  ← 各 rank 独立更新完整参数
   │           │          │           │
   ↓           ↓          ↓           ↓
 下一个 micro-batch 的 Forward

通讯：AllReduce(2Ψ)    内存：每卡完整参数 + 完整梯度 + 完整优化器状态


模式二：DistributedOptimizer（ReduceScatter + AllGather）
─────────────────────────────────────────────────────────────────
 GPU 0        GPU 1        GPU 2        GPU 3
┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐
│完整模型│    │完整模型│    │完整模型│    │完整模型│
└──┬───┘    └──┬───┘    └──┬───┘    └──┬───┘
   │           │           │           │
   └──────┬────┴─────┬─────┴──────┬────┘
          │          │          │
          │  ReduceScatter(梯度)│    ← backward 后，各 rank 只保留 1/D 梯度
          │          │          │
   ┌──────┴────┬─────┴────┬─────┴──────┐
   ↓           ↓          ↓           ↓
 optimizer   optimizer  optimizer  optimizer  ← 只更新本地参数分片
 (分片0)     (分片1)     (分片2)     (分片3)
   │           │          │           │
   └──────┬────┴─────┬─────┴──────┬────┘
          │          │          │
          │  AllGather(参数)    │    ← Forward 前，组装完整参数
          │          │          │
   ┌──────┴────┬─────┴────┬─────┴──────┐
   ↓           ↓          ↓           ↓
┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐
│完整模型│    │完整模型│    │完整模型│    │完整模型│
└──────┘    └──────┘    └──────┘    └──────┘
   ↓           ↓          ↓           ↓
 下一个 micro-batch 的 Forward

通讯：ReduceScatter(Ψ) + AllGather(Ψ) = 2Ψ    内存：每卡仅 1/D 参数分片 + 1/D 梯度分片
```

**与标准 DP 的对比**:

| 特性          | 标准 DP (AllReduce) | DistributedOptimizer |
|-------------|-------------------|----------------------|
| backward 通讯 | AllReduce (2Ψ)    | ReduceScatter (Ψ)    |
| forward 前通讯 | 无                 | AllGather (Ψ)        |
| **总通讯量**    | **2Ψ**            | **2Ψ**（相同）           |
| 优化器状态内存     | 每卡完整 (2Ψ)         | 每卡分片 (2Ψ/D)          |
| 梯度内存        | 每卡完整 (Ψ)          | 每卡分片 (Ψ/D)           |

**代码实现要点**（`param_and_grad_buffer.py:340-420`）:

```python
if self.ddp_config.use_distributed_optimizer:
    # ReduceScatter: 梯度分片到各 rank
    grad_reduce_handle = dist_reduce_scatter_func(
        local_data_view, bucket.grad_data, op=reduce_op, group=communication_group
    )
else:
    # 标准 DP: AllReduce 同步完整梯度
    torch.distributed.all_reduce(
        bucket.grad_data, op=reduce_op, group=communication_group
    )
```

**内存节省**（bf16 参数 + fp32 梯度，DP=D）：

- 标准 DP: 每参数 18 字节
- DistributedOptimizer: 每参数 `6 + 12/D` 字节
- 当 D=8 时，节省约 58% 内存

**通讯量**: 无论哪种模式，DP 通讯量均为 `2Ψ/D`（每个 DP 组内独立通讯）

### 4.2 张量并行 (TP)

**原理**: 将单层计算（如 MLP、Attention）切分到多个 GPU，每个 GPU 持有权重的一个分片，通过通信组合部分结果。

**实现位置**:

- MLP 层: `megatron/core/tensor_parallel/layers.py` 中的 `ColumnParallelLinear`、`RowParallelLinear`
- 通信原语: `megatron/core/tensor_parallel/mappings.py`
- Attention 层: 通过 Transformer Engine 实现

#### MLP 切分示例（TP=2）

**第一层：列并行 (Column Parallel)**

```
              Input X [s, b, h]  (完整，所有 rank 共享)
                      │
          ┌───────────┴───────────┐
          ↓                       ↓
    ┌───────────┐           ┌───────────┐
    │  GPU 0    │           │  GPU 1    │
    │           │           │           │
    │ W₁: 列分片│           │ W₁: 列分片│
    │ [h, h/2]  │           │ [h, h/2]  │
    │           │           │           │
    │ Y₁=XW₁    │           │ Y₂=XW₂    │
    │ [s,b,h/2] │           │ [s,b,h/2] │
    └───────────┘           └───────────┘
          │                       │
          └───────────┬───────────┘
                      ↓
         Y₁, Y₂ 各自保留本地分片，直接传入行并行
```

**第二层：行并行 (Row Parallel)**

```
   Y_1 (输入分片)    Y_2 (输入分片)
   ↓                ↓
   GeLU             GeLU
   ↓                ↓
   Linear2 分片1    Linear2 分片2
   (输入维度分片)    (输入维度分片)
   ↓                ↓
   └────┬───AllReduce───┬────┘
        ↓              ↓
   Output [s, b, h] (完整，每个rank都有完整输出)
```

#### Attention 切分

Attention 的 TP 切分通过 Multi-Head Attention 的 head 数量来实现：

- **QKV 投影**: 使用 Column Parallel（列并行），按 head 数分片
- **Output 投影**: 使用 Row Parallel（行并行），AllReduce 汇总
- 每个 TP rank 负责处理 `num_heads / T` 个 attention head
- 通讯模式与 MLP 相同：每个 Transformer 层 2 次 AllReduce（Attention + MLP 各一次）

#### 通讯模式

| 层                    | 前向通讯                   | 反向通讯              | 代码实现                                                                             |
|----------------------|------------------------|-------------------|----------------------------------------------------------------------------------|
| ColumnParallelLinear | 无（各 rank 独立计算输出分片）     | AllReduce（汇总输入梯度） | `layers.py:946` + `LinearWithGradAccumulationAndAsyncCommunication.backward:534` |
| RowParallelLinear    | AllReduce（汇总部分和得到完整输出） | 无（梯度直接传递）         | `layers.py:1274` + `_ReduceFromModelParallelRegion.forward:226`                  |

**通讯量分析**:

根据 1.4 节的 AllReduce 分析，AllReduce 通讯量 = `2 × (T-1)/T × data_size`（Ring 算法）。

AllReduce 操作的数据形状为 `[s, b, h]`（完整 hidden 维度），因此 data_size = bsh。

**MLP 层前后向通讯**：

- Linear2 前向：1 次 AllReduce → 通讯量 = `2 × (T-1)/T × bsh`
- Linear1 反向：1 次 AllReduce → 通讯量 = `2 × (T-1)/T × bsh`

**单层 MLP 总通讯量**: `4 × (T-1)/T × bsh`

**TP 总通讯量（L 层）**:
$$\text{Comm}_{TP} = 4L \times \frac{T-1}{T} \times bsh$$

当 `T >> 1` 时，`(T-1)/T ≈ 1`，近似为 `4Lbsh`，与 T 近似无关。

#### Sequence Parallel 优化

当 `sequence_parallel=True` 时，TP 的 AllReduce 被替换为 ReduceScatter/AllGather。SP 模式下输入数据在序列维度被切分为
`[s/T, b, h]`，各 rank 只持有 1/T 的序列。

**完整通讯链路**（`layers.py` + `mappings.py`）：

| 位置                | 非 SP 模式   | SP 模式                       | 数据形状变化                                                                          |
|-------------------|-----------|-----------------------------|---------------------------------------------------------------------------------|
| ColumnParallel 前向 | 无通信       | **AllGather** (s/T → s)     | `LinearWithGradAccumulationAndAsyncCommunication.forward:475`                   |
| ColumnParallel 反向 | AllReduce | **ReduceScatter** (s → s/T) | `LinearWithGradAccumulationAndAsyncCommunication.backward:547`                  |
| RowParallel 前向    | AllReduce | **ReduceScatter** (s → s/T) | `RowParallelLinear.forward:1270` → `reduce_scatter_to_sequence_parallel_region` |
| RowParallel 反向    | 无通信       | **AllGather** (s/T → s)     | `_GatherFromSequenceParallelRegion.backward`（autograd 自动处理）                     |

**通讯量对比**（每层 MLP）：

- 非 SP 模式：2 次 AllReduce = `2 × 2 × (T-1)/T × bsh = 4bsh × (T-1)/T`
- SP 模式：2 次 AllGather + 2 次 ReduceScatter = `4 × (T-1)/T × (s/T)bh` = `4bsh × (T-1)/T²`

**关键优势**：SP 模式的总通讯量略低于非 SP（因子 1/T），但更重要的一点是 SP 的 ReduceScatter 输出可以**与 DP 的
ReduceScatter 合并**（fuse），减少通信启动次数。

#### 通信与计算重叠

Megatron 使用异步通信（`async_op=True`）实现通信与计算的重叠（`layers.py:536-549`）：

```python
# 反向传播中，先发起 AllReduce/ReduceScatter（异步）
handle = torch.distributed.all_reduce(grad_input, group=tp_group, async_op=True)
# 然后立即开始计算 weight gradient（与通信并行执行）
grad_weight = grad_output.t().matmul(total_input)
# 最后等待通信完成
handle.wait()
```

需要设置 `CUDA_DEVICE_MAX_CONNECTIONS=1` 以确保通信调度在计算内核之前。

### 4.3 流水线并行 (PP)

**原理**: 将模型不同层分配到不同 GPU（stage），通过 P2P 通信在 stage 间传递激活值和梯度。

```
GPU0: Layer 1-4   GPU1: Layer 5-8   GPU2: Layer 9-12   GPU3: Layer 13-16
   ↓                    ↓                    ↓                    ↓
   └────────────────────┴────────────────────┴────────────────────┘
                    P2P Send/Recv (isend/irecv)
```

**通讯实现**（`megatron/core/pipeline_parallel/p2p_communication.py`）:

- 核心类 `P2PCommunicator`，使用 `torch.distributed.isend/irecv` 异步 P2P 通信
- 每个 stage 只与相邻 stage（prev_rank, next_rank）通信
- 支持组合操作：`send_forward_recv_backward`、`send_backward_recv_forward` 等

**Activation 数据形状**（`megatron/core/pipeline_parallel/schedules.py`）:

```python
tensor_shape = (seq_length / (cp_size * tp_size), micro_batch_size, hidden_size)
```

- 启用 Sequence Parallel 时，序列维度会被 TP 进一步切分
- 启用 Context Parallel 时，序列维度会被 CP 切分

**通讯量**:

- 每个 micro-batch 在相邻 stage 间传递 1 次前向 activation + 1 次反向 gradient
- P 个 stage 间共有 `P-1` 个边界，每个 micro-batch 需要跨越所有边界
- 注意：`b` 为 micro_batch_size（单个 micro-batch 的样本数），每个 DP rank 处理 `m` 个 micro-batch

**单个 micro-batch 的通讯量**：

$$\text{Comm}_{PP}^{\text{per-microbatch}} = 2(P-1) \times bsh$$

**每个 DP rank 的总通讯量**（m 个 micro-batch）：

$$\text{Comm}_{PP} = m \times 2(P-1) \times bsh$$

其中 $m = \frac{b_{\text{global}}}{D \times b}$，$b_{\text{global}}$ 为全局 batch size，$D$ 为 DP size，$b$ 为单个 micro-batch size。

**调度策略**:

- **1F1B**（One Forward One Backward）：稳态阶段每步 1 次前向 + 1 次反向，减少 bubble
- **Interleaved 1F1B**：虚拟 pipeline 并行，进一步减少 bubble
- 代码入口：`megatron/core/pipeline_parallel/schedules.py` 中的 `forward_backward_pipelining_without_interleaving` 和
  `forward_backward_pipelining_with_interleaving`

### 4.4 序列并行 / 上下文并行 (CP)

**原理**: 将长序列切分到不同 GPU，每个 GPU 只处理序列的一个子集，通过通信让所有 GPU 获得完整的注意力上下文。

**Megatron 中的两种序列并行机制**:

1. **Sequence Parallel (SP)**：通过 ReduceScatter/AllGather 在 Linear 层切分序列维度，减少 TP 的 AllReduce 通讯量
    - 实现位置：`megatron/core/tensor_parallel/mappings.py` 中的 `scatter_to_sequence_parallel_region`、
      `gather_from_sequence_parallel_region`
    - 与 TP 紧密耦合，当 `sequence_parallel=True` 时，ColumnParallel 的反向使用 ReduceScatter 替代 AllReduce，RowParallel
      的前向使用 ReduceScatter 替代 AllReduce

2. **Context Parallel (CP)**：在 Attention 层通过 Ring Attention 等机制处理超长序列
    - 实现位置：`megatron/core/extensions/transformer_engine.py` 中传递 `cp_group` 给 Transformer Engine
    - 核心实现在 Transformer Engine 库中

**CP 通讯模式**（`megatron/core/transformer/transformer_config.py`）:

| 模式           | 说明                                           | 可重叠 | 适用场景   |
|--------------|----------------------------------------------|-----|--------|
| `p2p`（默认）    | Ring Attention，异步 P2P 传递 KV Cache 边界 token   | ✅   | 通用，推荐  |
| `all_gather` | AllGather 完整 KV Cache 再计算                    | ❌   | 简单但效率低 |
| `a2a`        | DeepSpeed Ulysses 风格，scatter attention heads | ❌   | 短序列多卡  |
| `a2a+p2p`    | 分层实现：NVLink 域内 A2A，IBLink 域间 P2P             | 部分  | 大规模集群  |

**P2P 模式 (Ring Attention) 通讯量**:

- Ring 拓扑中，每个 rank 需要经过 `C-1` 个 step 才能获取完整的 KV Cache
- 每个 step 传递 1 个 chunk 的 KV Cache，形状为 `[s/C, b, h_kv]`，包含 Key 和 Value 两个张量
- 前向和反向各执行一轮完整的 Ring，因此通信次数翻倍

$$\text{Comm}_{CP} = \underbrace{2}_{\text{前向+反向}} \times \underbrace{(C-1)}_{\text{Ring steps}} \times \underbrace{2}_{\text{K+V}} \times \frac{s}{C} \times b \times h_{kv} = 4(C-1) \times \frac{s b h_{kv}}{C}$$

其中 $h_{kv} = N_{kv} \times d_h$, $N_{kv}$ 为 KV head 数，$d_h$ 为 head 维度，通常 $h_{kv} = h$ 或使用 GQA 时 $h_{kv} < h$。

当 $h_{kv} = h$ 时，近似为：

$$\text{Comm}_{CP} \approx 4 \times \frac{C-1}{C} \times s b h$$

### 4.5 专家并行 (EP)

**原理**: 将不同专家分配到不同 GPU（主要用于 MoE 模型），通过 All-to-All 通信将 token 路由到对应专家所在 GPU。

**通讯实现**（`megatron/core/transformer/moe/token_dispatcher.py`）:

```
Gate decisions: token -> expert mapping

All-to-All(dispatch):
GPU0: [t1->E0, t2->E1]  GPU1: [t3->E2, t4->E3]
   ↓                        ↓
GPU0: [t1, t4] (E0,E3)  GPU1: [t2, t3] (E1,E2)

Expert Computation

All-to-All(combine):
GPU0: [t1_result, t4_result]  GPU1: [t2_result, t3_result]
   ↓                              ↓
GPU0: [t1, t2]  GPU1: [t3, t4]
```

**完整的 Token 路由流程**（`MoEAlltoAllTokenDispatcher`）:

1. **Permutation**: 本地 token 按 expert 路由结果重排
2. **All-to-All (EP)**: 将 token 发送到目标 EP rank（dispatch）
3. **AllGather (TP)**: 收集其他 TP rank 的 token（如果 TP > 1）
4. **Expert Computation**: 各 rank 计算本地 expert 的结果
5. **ReduceScatter (TP)**: 将结果分散到各 TP rank（如果 TP > 1）
6. **All-to-All (EP)**: 将 token 返回原始 EP rank（combine）
7. **Unpermutation**: 恢复原始 token 顺序

**通讯量**:

- All-to-All 通讯传递的数据形状为 `[num_tokens, hidden_size]`，**包含完整的 hidden 维度 H**
- 每次 All-to-All（dispatch + combine）：E 个 rank 间的数据交换
- All-to-All 通讯量 = `2 × (E-1)/E × total_tokens × h`（Ring 算法等价）

**通讯量公式**（包含 top-k 参数）：

设 $k$ 为 top-k 数值（每个 token 被路由到 k 个 expert），token 被路由的总数为 $k \times s \times b$：

$$\text{Comm}_{EP} = 2 \times \frac{E-1}{E} \times k \times sbh = 2k \times \frac{E-1}{E} \times sbh$$

其中：
- All-to-All dispatch：$\frac{E-1}{E} \times k \times sbh$
- All-to-All combine：$\frac{E-1}{E} \times k \times sbh$
- 系数 2 表示前向和反向都需要通讯

**特殊情况**：
- 当 $k=1$ 时：$\text{Comm}_{EP} = 2 \times \frac{E-1}{E} \times sbh$
- 当 $k=2$ 时：$\text{Comm}_{EP} = 4 \times \frac{E-1}{E} \times sbh$

**优化**: Megatron 使用 DeepEP（`megatron/core/transformer/moe/fused_a2a.py`）融合 permute 和 all-to-all 操作，减少内存带宽开销。

### 4.6 并行策略组合通讯量

#### DP + TP

```
┌─────────────────────────────────────┐
│          DP Group 1 (2 GPUs)        │
│  ┌──────────────┬──────────────┐    │
│  │    TP-0      │     TP-1     │    │
│  │  (shard 1)   │   (shard 2)  │    │
│  └──────────────┴──────────────┘    │
├─────────────────────────────────────┤
│          DP Group 2 (2 GPUs)        │
│  ┌──────────────┬──────────────┐    │
│  │    TP-0      │     TP-1     │    │
│  │  (shard 1)   │   (shard 2)  │    │
│  └──────────────┴──────────────┘    │
└─────────────────────────────────────┘
```

**通讯量**:

**TP 内通讯（每个 DP 组内）**:
$$\text{Comm}_{TP} = 4L \times \frac{T-1}{T} \times bsh$$

**DP 间通讯（不同 DP 组间）**:
$$\text{Comm}_{DP} = 2 \times \frac{\Psi}{D}$$

**总通讯量**:
$$\text{Comm}_{DP+TP} = 4L \times \frac{T-1}{T} \times bsh + 2\frac{\Psi}{D}$$

#### DP + EP (MoE 场景)

**通讯量**:

**EP 内通讯**（MoE 层的 All-to-All，top-k=1 时）:
$$\text{Comm}_{EP} = 2k \times \frac{E-1}{E} \times sbh$$

其中 k 为 top-k 值（每个 token 路由到 k 个 expert）。top-k=2 时为 $4 \times \frac{E-1}{E} \times sbh$，top-k=1
时为 $2 \times \frac{E-1}{E} \times sbh$。详见 4.5 节。

**DP 间通讯**（梯度同步，其中 $\Psi_{\text{eff}} = $ dense params + expert params$/E$）:
$$\text{Comm}_{DP} = 2\frac{\Psi_{\text{eff}}}{D}$$

**总通讯量**:
$$\text{Comm}_{DP+EP} = 2k\frac{E-1}{E} \times sbh + 2\frac{\Psi_{\text{eff}}}{D}$$

#### DP + TP + EP

**通讯量**:

**TP 内通讯**（每个 DP 组内）:
$$\text{Comm}_{TP} = 4L \times \frac{T-1}{T} \times bsh$$

**EP 内通讯**（MoE 层，top-k=k 时）:
$$\text{Comm}_{EP} = 2k \times \frac{E-1}{E} \times sbh$$

**DP 间通讯**（梯度同步）:
$$\text{Comm}_{DP} = 2\frac{\Psi_{\text{eff}}}{D \times T \times E}$$

**总通讯量**:
$$\text{Comm}_{DP+TP+EP} = 4L \times \frac{T-1}{T} \times bsh + 2k\frac{E-1}{E} \times sbh + 2\frac{\Psi_{\text{eff}}}{D \times T \times E}$$

#### 3D 并行 (DP + TP + PP)

**通讯量**:

**TP 内通讯**（每个 DP 组内，分摊到 P 个 stage）:
$$\text{Comm}_{TP} = \frac{4L}{P} \times \frac{T-1}{T} \times bsh$$

**PP 间通讯**（stage 间的激活值和梯度传递，每 micro-batch）：
$$\text{Comm}_{PP} = m \times 2(P-1) \times bsh$$

**DP 间通讯**（梯度同步）：
$$\text{Comm}_{DP} = 2\frac{\Psi}{D \times T}$$

**总通讯量**:
$$\text{Comm}_{DP+TP+PP} = \frac{4L}{P} \times \frac{T-1}{T} \times bsh + m \times 2(P-1) \times bsh + 2\frac{\Psi}{D \times T}$$

#### 4D 并行 (DP + TP + PP + EP)

**通讯量**:

**TP 内通讯**（仅 dense 层，分摊到 P 个 stage）:
$$\text{Comm}_{TP} = \frac{4L_{\text{dense}}}{P} \times \frac{T-1}{T} \times bsh$$

**EP 内通讯**（MoE 层，top-k=k 时）:
$$\text{Comm}_{EP} = 2k \times \frac{E-1}{E} \times sbh$$

**PP 间通讯**（stage 间的激活值和梯度传递，每 micro-batch）：
$$\text{Comm}_{PP} = m \times 2(P-1) \times bsh$$

**DP 间通讯**（梯度同步）：
$$\text{Comm}_{DP} = 2\frac{\Psi_{\text{eff}}}{D \times T \times E \times P}$$

**总通讯量**:
$$\text{Comm}_{DP+TP+PP+EP} = \frac{4L_{\text{dense}}}{P} \times \frac{T-1}{T} \times bsh + 2k\frac{E-1}{E} \times sbh + m \times 2(P-1) \times bsh + 2\frac{\Psi_{\text{eff}}}{D \times T \times E \times P}$$

---

## 分布式训练并行策略测试

本项目提供了分布式训练并行策略的简化实现，用于学习和验证 NCCL 通讯原语和并行策略。

### 测试文件说明

| 文件                 | 内容                      | 运行方式          |
|--------------------|-------------------------|---------------|
| `zero_impl.py`     | ZeRO-1/2/3 实现           | 单机多卡 torchrun |
| `fsdp_impl.py`     | PyTorch FSDP 实现         | 单机多卡 torchrun |
| `megatron_impl.py` | Megatron DP/TP/PP/EP 实现 | 单机多卡 torchrun |

### Ray 集群提交

在 8 卡 GPU 集群上执行测试：

```bash
# ZeRO 测试 (1-3阶段)
ray job submit --address='http://10.148.11.18:8420' \
  --working-dir=. \
  --entrypoint-num-gpus 8 \
  -- torchrun --nproc_per_node=8 --master-port=29610 zero_impl.py

# FSDP 测试
ray job submit --address='http://10.148.11.18:8420' \
  --working-dir=. \
  --entrypoint-num-gpus 8 \
  -- torchrun --nproc_per_node=8 --master-port=29611 fsdp_impl.py

# Megatron 并行测试 (DP+TP+PP+EP)
ray job submit --address='http://10.148.11.18:8420' \
  --working-dir=. \
  --entrypoint-num-gpus 8 \
  -- torchrun --nproc_per_node=8 --master-port=29612 megatron_impl.py
```

### 本地测试 (4卡)

```bash
# ZeRO
torchrun --nproc_per_node=8 zero_impl.py

# FSDP
torchrun --nproc_per_node=8 fsdp_impl.py

# Megatron (需要8卡)
torchrun --nproc_per_node=8 megatron_impl.py
```

# Reference

- https://docs.pytorch.org/docs/stable/distributed.html
- https://zhuanlan.zhihu.com/p/485208899
