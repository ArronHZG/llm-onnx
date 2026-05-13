# verl Packing 逻辑完整分析

## 一、核心概念

verl 中的 **packing** 指的是 **sequence packing（序列打包）**，也称为 **remove padding** 或 **THD 格式**。其核心思想是将多个变长序列拼接成一个连续张量（无 padding），避免 padding token 带来的计算浪费。

### 两种数据格式对比

| 格式 | 全称 | 说明 |
|------|------|------|
| **BSHD 格式** | Batch-Sequence-Head-Dimension | 传统格式，所有样本 padding 到相同长度，存在大量无效计算 |
| **THD 格式** | Time-Head-Dimension | 打包格式，将所有有效 token 拼接成一维连续张量，通过 `cu_seqlens` 标记每个样本的边界 |

---

## 二、配置项

核心配置项分布在以下位置：

### 2.1 Megatron 引擎配置

**文件：** `verl/trainer/config/engine/megatron.yaml:96-97`

```yaml
# 是否使用 THD 格式（序列打包），否则使用 BSHD 格式
use_remove_padding: True
```

### 2.2 Actor 配置

**文件：** `verl/trainer/config/actor/actor.yaml:27-33`

```yaml
# 启用动态 batch size（基于 token 数而非样本数划分 micro-batch）
use_dynamic_bsz: false
# 每个 GPU 一次 PPO 前向的最大 token 数
ppo_max_token_len_per_gpu: 16384
```

### 2.3 Rollout 配置

**文件：** `verl/trainer/config/rollout/rollout.yaml:103-109`

```yaml
# log_prob 计算时启用动态批处理
log_prob_use_dynamic_bsz: ${oc.select:actor_rollout_ref.actor.use_dynamic_bsz,false}
log_prob_max_token_len_per_gpu: ${oc.select:actor_rollout_ref.actor.ppo_max_token_len_per_gpu,16384}
```

---

## 三、核心实现逻辑

### 3.1 Packing 预处理（BSHD → THD）

核心实现在 `verl/models/mcore/util.py:44-131` 的 `preprocess_packed_seqs` 函数。

**关键步骤：**

1. 从 `attention_mask` 计算每个样本的实际长度：`seqlens_in_batch = attention_mask.sum(dim=-1)`
2. 构建累积序列长度：`cu_seqlens = [0, L₁, L₁+L₂, ..., ΣLᵢ]`
3. 调用 `unpad_input` 将 BSHD 格式转为 THD 格式（去掉 padding token）
4. 创建 `PackedSeqParams` 对象，包含 `cu_seqlens_q`、`max_seqlen_q` 等元信息
5. 对于 Context Parallelism (CP > 1)，按 CP 分片做负载均衡

### 3.2 Unpad 操作（FSDP Actor）

**文件：** `verl/workers/actor/dp_actor.py:166-183`

```python
if self.use_remove_padding:
    input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
        input_ids.unsqueeze(-1), attention_mask
    )
    # input_ids_rmpad 形状从 (bsz, seqlen) 变为 (total_nnz,)
    input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

    # position_ids 也需要同步 unpad
    position_ids_rmpad = index_first_axis(
        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
    ).transpose(0, 1)
```

其中 `indices` 记录了每个有效 token 在原始 batch 中的位置，用于后续恢复。

### 3.3 Pad 恢复（THD → BSHD）

**文件：** `verl/workers/actor/dp_actor.py:333-338`

```python
full_log_probs = pad_input(
    hidden_states=log_probs.unsqueeze(-1),
    indices=indices,
    batch=batch_size,
    seqlen=seqlen,
)
```

将 THD 格式的输出恢复为 BSHD 格式，填充回原始位置。

### 3.4 Attention Mask 处理

THD 格式下不再需要传统的 attention mask。模型通过 `PackedSeqParams.cu_seqlens` 和 `flash_attn_varlen` 实现可变长度注意力。

**文件：** `verl/models/mcore/util.py:55-68`

```python
seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=input_ids.device)
cu_seqlens[1:] = torch.cumsum(seqlens_in_batch, dim=0)
```

### 3.5 Position IDs 处理

Position IDs 同样被 unpad，保持与 token 的对应关系。

**文件：** `verl/models/mcore/util.py:389-393`

```python
position_ids_rmpad[start_idx : start_idx + seqlen] = torch.arange(
    seqlen, dtype=torch.long, device=input_ids.device
)
```

---

## 四、动态批处理（Dynamic Batch Size）

当 `use_dynamic_bsz=True` 时，micro-batch 按 token 数而非样本数划分。

**核心实现：** `verl/utils/seqlen_balancing.py:489-525` 的 `prepare_dynamic_batch` 函数

**流程：**

1. 计算每个样本的 token 数（workload）
2. 使用 Karmarkar-Karp 分区算法将样本分配到各 micro-batch，使每组的总 token 数接近 `ppo_max_token_len_per_gpu`
3. 返回分区后的 micro-batch 列表

**调用位置：** `verl/workers/actor/dp_actor.py:468-474`

```python
if use_dynamic_bsz:
    max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
    micro_batches, batch_idx_list = prepare_dynamic_batch(
        data, max_token_len=max_token_len, dp_group=torch.distributed.group.WORLD
    )
else:
    micro_batches = data.split(micro_batch_size)
```

---

## 五、训练阶段 vs Rollout 阶段的使用差异

| 阶段 | 组件 | 配置项 | 用途 |
|------|------|--------|------|
| 训练 | Actor (FSDP) | `use_remove_padding` | 前向/反向传播时去除 padding |
| 训练 | Actor (Megatron) | `use_remove_padding` → THD 格式 | Megatron 引擎的序列打包 |
| 训练 | Critic | 与 Actor 一致 | Value 模型的 log_prob 计算 |
| Rollout | vLLM/SGLang | `log_prob_use_dynamic_bsz` | rollout 后计算 log_prob 时使用动态批处理 |
| Rollout | Megatron Engine | `use_remove_padding` → THD 格式 | 引擎层的序列打包 |

---

## 六、关键文件路径汇总

| 文件 | 功能 |
|------|------|
| `verl/models/mcore/util.py` | Packing 核心工具函数（preprocess/postprocess） |
| `verl/models/mcore/model_forward.py` | 带 packing 的模型前向传播入口 |
| `verl/utils/attention_utils.py` | unpad/pad 输入的统一封装 |
| `verl/utils/seqlen_balancing.py` | 动态批处理和负载均衡算法 |
| `verl/workers/actor/dp_actor.py` | FSDP Actor 的 unpad/pad 实现 |
| `verl/workers/actor/megatron_actor.py` | Megatron Actor 的 THD 格式支持 |
| `verl/workers/engine/megatron/transformer_impl.py` | Megatron 引擎的 THD 配置 |

---

## 七、数据流示意

### 原始数据（BSHD 格式）

```
input_ids: [[1, 2, 3, 0, 0], [4, 5, 6, 7, 0]]   (bsz=2, seqlen=5)
attn_mask: [[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]]
```

### Unpack 后（THD 格式）

```
input_ids_rmpad: [1, 2, 3, 4, 5, 6, 7]   (total_nnz=7)
cu_seqlens:      [0, 3, 7]                (累积长度)
indices:         [0, 1, 2, 3, 4, 5, 6]     (原始位置)
```

### 模型计算后 Pad 恢复

```
output: [[a, b, c, 0, 0], [d, e, f, g, 0]]   (恢复为 BSHD)
```