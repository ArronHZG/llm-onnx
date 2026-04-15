**离策略（off-policy）诊断指标**，用于衡量 rollout 策略和训练策略之间的分布偏差

## 背景

在 RLHF/PPO 训练中：
- **rollout 策略**（vLLM/SGLang）：生成样本
- **训练策略**（FSDP/Megatron）：做梯度更新

两者之间存在"off-policy gap"，可能导致训练不稳定。这些指标帮助诊断问题严重程度。

## 各指标解释

### 1. 训练策略 PPL（始终可用）

| 指标 | 公式 | 含义 |
|------|------|------|
| [training_ppl](file:///Users/arron/Projects/verl_elastic_scheduling/verl/trainer/ppo/rollout_corr_helper.py#916#6) | `exp(-mean(log π_train))` | 训练策略的困惑度 |
| [training_log_ppl](file:///Users/arron/Projects/verl_elastic_scheduling/verl/trainer/ppo/rollout_corr_helper.py#944#13) | `-mean(log π_train)` | 对数困惑度（避免指数爆炸） |

PPL 越低 = 策略越"确信"自己的输出。

### 2. KL 散度（核心指标）

| 指标 | 公式 | 含义 |
|------|------|------|
| [kl](file:///Users/arron/Projects/verl_elastic_scheduling/verl/trainer/ppo/rollout_corr_helper.py#914#6) | `E[log π_rollout - log π_train]` | 直接估计器 `KL(π_rollout \|\| π_train)` |
| [k3_kl](file:///Users/arron/Projects/verl_elastic_scheduling/verl/trainer/ppo/rollout_corr_helper.py#915#6) | `E[exp(r) - log(r) - 1]`，`r = π_train/π_rollout` | K3 估计器，对**小 KL 值更稳定** |

- `kl > 0`：rollout 比训练策略更确信
- [kl](file:///Users/arron/Projects/verl_elastic_scheduling/verl/trainer/ppo/rollout_corr_helper.py#914#6) 过大 → 重要性采样方差大，训练可能不稳定

**为什么需要 K3 估计器？**

直接 KL 在 log_ratio 接近 0 时不精确（浮点误差），K3 利用 `e^x ≈ 1 + x + x²/2` 的性质，数值更稳定。

### 3. Rollout 策略 PPL

| 指标 | 含义 |
|------|------|
| [rollout_ppl](file:///Users/arron/Projects/verl_elastic_scheduling/verl/trainer/ppo/rollout_corr_helper.py#917#6) | rollout 策略的困惑度 |
| [rollout_log_ppl](file:///Users/arron/Projects/verl_elastic_scheduling/verl/trainer/ppo/rollout_corr_helper.py#964#17) | 对数困惑度 |

### 4. PPL 差异与比率

| 指标 | 公式 | 含义 |
|------|------|------|
| [log_ppl_diff](file:///Users/arron/Projects/verl_elastic_scheduling/verl/trainer/ppo/rollout_corr_helper.py#918#6) | `mean(log π_rollout) - mean(log π_train)` | 正值 = 训练 PPL 更高（训练策略更不确定）|
| [log_ppl_abs_diff](file:///Users/arron/Projects/verl_elastic_scheduling/verl/trainer/ppo/rollout_corr_helper.py#973#17) | `\|log_ppl_diff\|` | 绝对差异 |
| `log_ppl_diff_max/min` | 最大/最小差异 | 检测极端离群值 |
| [ppl_ratio](file:///Users/arron/Projects/verl_elastic_scheduling/verl/trainer/ppo/rollout_corr_helper.py#919#6) | [exp(log_ppl_diff)](file:///Users/arron/Projects/verl_elastic_scheduling/verl/trainer/ppo/rollout_corr_helper.py#980#62) | 训练 PPL / Rollout PPL |

### 5. 卡方散度（重要性采样方差）

| 指标 | 公式 | 含义 |
|------|------|------|
| [chi2_token](file:///Users/arron/Projects/verl_elastic_scheduling/verl/trainer/ppo/rollout_corr_helper.py#920#6) | [E_token[ρ²] - 1](file:///Users/arron/Projects/verl_elastic_scheduling/verl/trainer/ppo/rollout_corr_helper.py#988#23)，`ρ = π_train/π_rollout` | **token 级** IS 方差 |
| [chi2_seq](file:///Users/arron/Projects/verl_elastic_scheduling/verl/trainer/ppo/rollout_corr_helper.py#921#6) | `E_seq[(Πρ_t)²] - 1` | **序列级** IS 方差 |

这是最关键的警告指标：

```
χ² = 0      → 完美 on-policy，无方差
χ² < 1      → 方差可控，训练稳定
χ² ≈ 1      → 方差与信号等大，边界情况
χ² > 1      → 方差远超信号，训练可能发散 ⚠️
```

### 实际使用场景

```
指标正常 ✅:  kl=0.02, k3_kl=0.02, chi2_token=0.05, chi2_seq=0.1
  → rollout 和训练策略接近，off-policy gap 小

精度问题 ⚠️:  kl=0.5, log_ppl_diff=0.3, ppl_ratio=1.35
  → BF16 vs FP32 精度差异导致，需检查 rollout 配置

严重偏离 ❌:  kl=2.0, chi2_seq=5.0
  → 样本来自非常旧的 checkpoint，需要降低 staleness 阈值
```