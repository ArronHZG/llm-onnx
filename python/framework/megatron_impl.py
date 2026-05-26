"""
Megatron-LM 并行策略最简化实现

包含:
- 数据并行 (DP): AllReduce梯度同步
- 张量并行 (TP): 列并行 + 行并行 Linear
- 流水线并行 (PP): 层间分配到不同GPU
- 专家并行 (EP): MoE All-to-All
- 组合并行: DP+TP, DP+EP, DP+TP+PP, 3D并行等

运行方式:
    torchrun --nproc_per_node=8 megatron_impl.py
"""

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def setup_distributed():
    """初始化分布式环境"""
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    return rank, local_rank, world_size


@dataclass
class ParallelConfig:
    """并行配置"""

    dp_size: int = 1  # 数据并行大小
    tp_size: int = 1  # 张量并行大小
    pp_size: int = 1  # 流水线并行大小
    ep_size: int = 1  # 专家并行大小
    cp_size: int = 1  # 序列并行大小（本实现中暂不涉及）

    @property
    def world_size(self):
        return self.dp_size * self.tp_size * self.pp_size * self.ep_size


def create_process_groups(config: ParallelConfig, rank: int):
    """
    创建所有需要的进程组

    并行策略组合时的进程组划分:
    - TP组: 同一DP/PP/EP组内的张量并行GPU
    - DP组: 同一TP/PP/EP组内的数据并行GPU
    - PP组: 同一DP/TP/EP组内的流水线GPU
    - EP组: 同一DP/TP/PP组内的专家并行GPU
    """
    world_size = config.world_size

    # 计算各维度坐标
    # 布局: [DP, PP, EP, TP] 或根据具体硬件拓扑调整
    ep_stride = config.tp_size
    pp_stride = config.tp_size * config.ep_size
    dp_stride = config.tp_size * config.ep_size * config.pp_size

    tp_rank = rank % config.tp_size
    ep_rank = (rank // config.tp_size) % config.ep_size
    pp_rank = (rank // (config.tp_size * config.ep_size)) % config.pp_size
    dp_rank = rank // (config.tp_size * config.ep_size * config.pp_size)

    # 创建TP组
    tp_groups = []
    for dp in range(config.dp_size):
        for pp in range(config.pp_size):
            for ep in range(config.ep_size):
                ranks = [
                    dp * dp_stride + pp * pp_stride + ep * ep_stride + tp
                    for tp in range(config.tp_size)
                ]
                grp = dist.new_group(ranks)
                tp_groups.append(grp)
                if rank in ranks:
                    my_tp_group = grp
                    my_tp_ranks = ranks

    # 创建DP组
    dp_groups = []
    for pp in range(config.pp_size):
        for ep in range(config.ep_size):
            for tp in range(config.tp_size):
                ranks = [
                    d * dp_stride + pp * pp_stride + ep * ep_stride + tp
                    for d in range(config.dp_size)
                ]
                grp = dist.new_group(ranks)
                dp_groups.append(grp)
                if rank in ranks:
                    my_dp_group = grp
                    my_dp_ranks = ranks

    # 创建PP组
    pp_groups = []
    for dp in range(config.dp_size):
        for ep in range(config.ep_size):
            for tp in range(config.tp_size):
                ranks = [
                    dp * dp_stride + p * pp_stride + ep * ep_stride + tp
                    for p in range(config.pp_size)
                ]
                grp = dist.new_group(ranks)
                pp_groups.append(grp)
                if rank in ranks:
                    my_pp_group = grp
                    my_pp_ranks = ranks

    # 创建EP组
    ep_groups = []
    for dp in range(config.dp_size):
        for pp in range(config.pp_size):
            for tp in range(config.tp_size):
                ranks = [
                    dp * dp_stride + pp * pp_stride + e * ep_stride + tp
                    for e in range(config.ep_size)
                ]
                grp = dist.new_group(ranks)
                ep_groups.append(grp)
                if rank in ranks:
                    my_ep_group = grp
                    my_ep_ranks = ranks

    return {
        "tp": (my_tp_group, my_tp_ranks),
        "dp": (my_dp_group, my_dp_ranks),
        "pp": (my_pp_group, my_pp_ranks),
        "ep": (my_ep_group, my_ep_ranks),
    }


# ========================
# 1. 数据并行 (Data Parallel)
# ========================


class DataParallelLayer(nn.Module):
    """
    数据并行层
    原理: 每个GPU存储完整模型副本，反向传播后AllReduce同步梯度

    通讯: AllReduce (梯度同步)
    通讯量: 2Ψ per step (Ψ为参数总量)
    """

    def __init__(self, module: nn.Module, dp_group):
        super().__init__()
        self.module = module
        self.dp_group = dp_group

    def forward(self, x):
        return self.module(x)

    def allreduce_gradients(self):
        """AllReduce梯度"""
        for param in self.module.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, group=self.dp_group, op=dist.ReduceOp.SUM)
                param.grad /= dist.get_world_size(self.dp_group)


# ========================
# 2. 张量并行 (Tensor Parallel)
# ========================


class ColumnParallelLinear(nn.Module):
    """
    列并行线性层: 每个rank只计算输出的一部分

    Y = X @ W^T + b，其中 W = [W_0, W_1, ..., W_{tp_size-1}]
    - 每个rank计算 Y_i = X @ W_i^T
    - 各rank的输出 Y_i 沿输出维度进行 AllGather 得到完整输出
    - 反向时输出梯度被 ReduceScatter 切分到各rank

    前向通讯: AllGather (接收其他rank的 Y_j，但这里简化为不做AllGather)
    反向通讯: ReduceScatter (分散梯度到各rank)
    """

    def __init__(self, in_features: int, out_features: int, tp_group):
        super().__init__()
        self.tp_group = tp_group
        self.tp_size = dist.get_world_size(tp_group)
        self.tp_rank = dist.get_rank(tp_group)

        # 切分输出维度：每个rank负责输出的 1/tp_size
        assert out_features % self.tp_size == 0
        self.output_size_per_partition = out_features // self.tp_size

        # weight shape: [out_features/tp_size, in_features]
        self.weight = nn.Parameter(
            torch.randn(self.output_size_per_partition, in_features)
        )
        self.bias = nn.Parameter(torch.zeros(self.output_size_per_partition))

    def forward(self, x):
        # 每个rank独立计算自己负责的输出分片
        # 如果输入是完整的，则输出是该rank负责的输出切片
        output_parallel = F.linear(x, self.weight, self.bias)
        return output_parallel

    def get_comm_volume(self, batch_size, seq_len):
        """计算通讯量 (bytes) - AllGather"""
        # 前向: 每个rank发送自己的输出分片给其他rank: (tp_size-1) / tp_size * 输出大小
        return (
            batch_size
            * seq_len
            * self.output_size_per_partition
            * (self.tp_size - 1)
            * 4
        )


class RowParallelLinear(nn.Module):
    """
    行并行线性层: 每个rank只处理输入的一部分

    Y = X @ W^T + b，其中 X = [X_0, X_1, ..., X_{tp_size-1}]（按最后一维切分）
    - 权重 W 按行切分: W = [W_0; W_1; ...; W_{tp_size-1}]
    - 每个rank计算 Y_i = X_i @ W_i^T（部分和）
    - AllReduce 汇总得到 Y = sum(Y_i)
    - 反向时梯度被 AllReduce 同步给所有rank

    前向通讯: AllReduce (累加各rank的部分和)
    反向通讯: AllReduce (同步梯度到各rank)
    """

    def __init__(self, in_features: int, out_features: int, tp_group):
        super().__init__()
        self.tp_group = tp_group
        self.tp_size = dist.get_world_size(tp_group)
        self.tp_rank = dist.get_rank(tp_group)

        # 切分输入维度：每个rank负责输入的 1/tp_size
        assert in_features % self.tp_size == 0
        self.input_size_per_partition = in_features // self.tp_size

        # weight shape: [out_features, in_features/tp_size]
        self.weight = nn.Parameter(
            torch.randn(out_features, self.input_size_per_partition)
        )

        # 只有rank0有bias（避免在AllReduce中重复加）
        if self.tp_rank == 0:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        # 输入x已经是该rank负责的输入分片（大小为[batch, seq_len, in_features/tp_size]）
        # 计算该rank的部分和
        output_parallel = F.linear(x, self.weight)

        # AllReduce: 累加所有rank的部分和得到完整输出
        dist.all_reduce(output_parallel, group=self.tp_group, op=dist.ReduceOp.SUM)

        # 只在rank0加bias（其他rank的bias为None）
        if self.bias is not None:
            output_parallel = output_parallel + self.bias

        return output_parallel

    def get_comm_volume(self, batch_size, seq_len):
        """计算通讯量 (bytes) - AllReduce"""
        # 前向: AllReduce 累加各rank的部分和
        return batch_size * seq_len * self.weight.size(0) * 4


class TensorParallelMLP(nn.Module):
    """
    张量并行MLP
    Linear1 (列并行) -> GeLU -> Linear2 (行并行)

    流程:
    1. Linear1 (ColumnParallel): X (完整) -> Y_i (输出分片)
    2. GeLU: 各rank独立处理Y_i
    3. Linear2 (RowParallel): X_i (输入分片) -> AllReduce -> Y (完整)

    通讯: Linear2的AllReduce (将各rank的部分和累加)
    通讯量: batch_size * seq_len * hidden_dim * 4 (bytes)
    """

    def __init__(self, hidden_size: int, tp_group):
        super().__init__()
        self.fc = ColumnParallelLinear(hidden_size, hidden_size * 4, tp_group)
        self.gelu = nn.GELU()
        self.proj = RowParallelLinear(hidden_size * 4, hidden_size, tp_group)

    def forward(self, x):
        h = self.fc(x)
        h = self.gelu(h)
        return self.proj(h)

    def get_comm_volume(self, batch_size, seq_len):
        return self.fc.get_comm_volume(batch_size, seq_len) + self.proj.get_comm_volume(
            batch_size, seq_len
        )


# ========================
# 3. 流水线并行 (Pipeline Parallel)
# ========================


class PipelineParallelStage(nn.Module):
    """
    流水线并行的一个阶段
    只包含模型的一部分层

    通讯: P2P Send/Recv (激活值和梯度)
    通讯量: 2 * batch_size * seq_len * hidden_dim per stage boundary
    """

    def __init__(self, layers: nn.ModuleList, pp_group, pp_rank, pp_size):
        super().__init__()
        self.layers = layers
        self.pp_group = pp_group
        self.pp_rank = pp_rank
        self.pp_size = pp_size

        # 获取相邻rank
        ranks = sorted([dist.get_rank(pp_group) for _ in range(pp_size)])
        self.prev_rank = ranks[pp_rank - 1] if pp_rank > 0 else None
        self.next_rank = ranks[pp_rank + 1] if pp_rank < pp_size - 1 else None

    def forward(self, x, recv_from_prev=True, send_to_next=True):
        """
        前向传播，包含P2P通讯
        """
        # 从上一个stage接收
        if recv_from_prev and self.prev_rank is not None:
            dist.recv(x, src=self.prev_rank)

        # 本地计算
        for layer in self.layers:
            x = layer(x)

        # 发送到下一个stage
        if send_to_next and self.next_rank is not None:
            dist.send(x, dst=self.next_rank)

        return x

    def backward(self, grad_output, recv_from_next=True, send_to_prev=True):
        """
        反向传播的P2P通讯
        """
        # 从下一个stage接收梯度
        if recv_from_next and self.next_rank is not None:
            dist.recv(grad_output, src=self.next_rank)

        # 本地反向传播（简化，实际会调用backward）

        # 发送到上一个stage
        if send_to_prev and self.prev_rank is not None:
            dist.send(grad_output, dst=self.prev_rank)

        return grad_output


# ========================
# 4. 专家并行 (Expert Parallel - MoE)
# ========================


class ExpertParallelMoE(nn.Module):
    """
    专家并行MoE层
    不同专家分布在不同GPU上，使用All-to-All进行token路由

    通讯: All-to-All (dispatch + combine)
    通讯量: 2 * (ep_size-1)/ep_size * num_tokens * hidden_dim
    """

    def __init__(self, hidden_size: int, num_experts: int, top_k: int, ep_group):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.ep_group = ep_group
        self.ep_size = dist.get_world_size(ep_group)
        self.ep_rank = dist.get_rank(ep_group)

        # 每个rank负责部分专家
        assert num_experts % self.ep_size == 0
        self.experts_per_rank = num_experts // self.ep_size

        # 创建本地专家
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 4),
                    nn.GELU(),
                    nn.Linear(hidden_size * 4, hidden_size),
                )
                for _ in range(self.experts_per_rank)
            ]
        )

        # 门控网络（所有rank都有）
        self.gate = nn.Linear(hidden_size, num_experts)

    def forward(self, x):
        """
        MoE前向传播流程:
        1. 门控网络计算每个token分配到哪些专家
        2. All-to-All: 将token发送到对应专家所在rank
        3. 本地专家计算
        4. All-to-All: 将结果返回原rank
        5. 加权聚合
        """
        batch_size, seq_len, hidden = x.shape
        x_flat = x.view(-1, hidden)  # [num_tokens, hidden]
        num_tokens = x_flat.size(0)

        # 1. 门控: [num_tokens, num_experts]
        logits = self.gate(x_flat)
        weights, indices = torch.topk(F.softmax(logits, dim=-1), self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)  # 归一化

        # 2. All-to-All Dispatch (简化实现)
        # 实际中需要按照expert分配组织数据
        dispatched = self.all_to_all(x_flat, indices)

        # 3. 本地专家计算
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            # 获取分配给这个专家的token
            mask = indices // self.experts_per_rank == self.ep_rank
            if mask.any():
                tokens = dispatched[mask[:, i]]
                out = expert(tokens)
                expert_outputs.append(out)

        # 4. All-to-All Combine
        combined = self.all_to_all_reverse(expert_outputs, indices, num_tokens)

        # 5. 加权聚合
        output = torch.zeros_like(x_flat)
        for i in range(self.top_k):
            expert_idx = indices[:, i]
            weight = weights[:, i : i + 1]
            # 这里简化处理，实际需要根据专家位置聚合
            output += combined * weight

        return output.view(batch_size, seq_len, hidden)

    def all_to_all(self, x, indices):
        """
        All-to-All通讯: 分发token到对应专家
        实际All-to-All会将数据分块并交换
        """
        # 模拟All-to-All: 每个rank都会收到其他rank的数据
        output = x.clone()
        dist.all_reduce(output, group=self.ep_group, op=dist.ReduceOp.SUM)
        return output / self.ep_size

    def all_to_all_reverse(self, x, indices, num_tokens):
        """All-to-All返回"""
        combined = torch.cat(x) if isinstance(x, list) else x
        output = combined.clone()
        dist.all_reduce(output, group=self.ep_group, op=dist.ReduceOp.SUM)
        return output / self.ep_size

    def get_comm_volume(self, num_tokens):
        """计算All-to-All通讯量"""
        # 两次All-to-All
        ep = self.ep_size
        vol_per_all2all = (ep - 1) / ep * num_tokens * self.hidden_size * 4
        return 2 * vol_per_all2all


# ========================
# 5. 组合并行测试
# ========================


class MegatronModel(nn.Module):
    """
    Megatron风格的组合并行模型
    支持 DP + TP + PP + EP 的任意组合
    """

    def __init__(
        self,
        config: ParallelConfig,
        pgroups: dict,
        hidden_size: int = 256,
        num_layers: int = 8,
    ):
        super().__init__()
        self.config = config
        self.pgroups = pgroups
        self.hidden_size = hidden_size

        tp_group = pgroups["tp"][0]
        pp_group = pgroups["pp"][0]
        ep_group = pgroups["ep"][0]
        dp_group = pgroups["dp"][0]

        tp_size = dist.get_world_size(tp_group)
        pp_size = dist.get_world_size(pp_group)
        pp_rank = dist.get_rank(pp_group)

        # 层分配到不同PP rank
        layers_per_stage = num_layers // pp_size
        start_layer = pp_rank * layers_per_stage
        end_layer = start_layer + layers_per_stage

        # 创建该PP阶段的层
        self.layers = nn.ModuleList()
        for i in range(start_layer, end_layer):
            if tp_size > 1:
                # 使用张量并行
                layer = TensorParallelMLP(hidden_size, tp_group)
            else:
                layer = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 4),
                    nn.GELU(),
                    nn.Linear(hidden_size * 4, hidden_size),
                )
            self.layers.append(layer)

        # 如果是MoE模型，某些层使用EP
        if dist.get_world_size(ep_group) > 1:
            self.use_moe = True
            self.moe_layer = ExpertParallelMoE(
                hidden_size, num_experts=8, top_k=2, ep_group=ep_group
            )
        else:
            self.use_moe = False

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def allreduce_dp_gradients(self):
        """数据并行的梯度同步"""
        if self.config.dp_size > 1:
            dp_group = self.pgroups["dp"][0]
            for param in self.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, group=dp_group)
                    param.grad /= self.config.dp_size


def test_dp():
    """测试数据并行"""
    rank, local_rank, world_size = setup_distributed()

    if rank == 0:
        print("\n" + "=" * 60)
        print("Testing Data Parallel (DP)")
        print("=" * 60)

    config = ParallelConfig(dp_size=world_size, tp_size=1, pp_size=1, ep_size=1)
    pgroups = create_process_groups(config, rank)

    model = nn.Sequential(nn.Linear(256, 512), nn.GELU(), nn.Linear(512, 256)).cuda()

    # 包装为DP
    dp_model = DataParallelLayer(model, pgroups["dp"][0])

    # 模拟训练
    x = torch.randn(4, 256, device="cuda")
    output = dp_model(x)
    loss = output.sum()
    loss.backward()

    # AllReduce梯度
    dp_model.allreduce_gradients()

    if rank == 0:
        print(f"[DP] Rank {rank}: loss={loss.item():.4f}")
        print(f"[DP] Communication: AllReduce per step, volume=2Ψ")

    dist.destroy_process_group()


def test_tp():
    """测试张量并行"""
    rank, local_rank, world_size = setup_distributed()

    if world_size < 2:
        print("TP test requires at least 2 GPUs")
        return

    if rank == 0:
        print("\n" + "=" * 60)
        print("Testing Tensor Parallel (TP)")
        print("=" * 60)

    config = ParallelConfig(dp_size=1, tp_size=world_size, pp_size=1, ep_size=1)
    pgroups = create_process_groups(config, rank)

    # 创建TP MLP
    tp_mlp = TensorParallelMLP(256, pgroups["tp"][0]).cuda()

    x = torch.randn(2, 10, 256, device="cuda")
    output = tp_mlp(x)

    comm_vol = tp_mlp.get_comm_volume(2, 10) / 1024**2  # MB

    if rank == 0:
        print(f"[TP] Input shape: {x.shape}, Output shape: {output.shape}")
        print(f"[TP] Communication per layer: {comm_vol:.2f} MB")
        print(f"[TP] Pattern: 2x AllReduce per MLP layer")

    dist.destroy_process_group()


def test_dp_tp():
    """测试 DP + TP 组合"""
    rank, local_rank, world_size = setup_distributed()

    if world_size < 4 or world_size % 2 != 0:
        print("DP+TP test requires 4+ GPUs with even number")
        return

    if rank == 0:
        print("\n" + "=" * 60)
        print("Testing Data Parallel + Tensor Parallel (DP+TP)")
        print("=" * 60)

    tp_size = 2
    dp_size = world_size // tp_size

    config = ParallelConfig(dp_size=dp_size, tp_size=tp_size, pp_size=1, ep_size=1)
    pgroups = create_process_groups(config, rank)

    # 创建模型
    tp_mlp = TensorParallelMLP(256, pgroups["tp"][0]).cuda()

    # 模拟训练
    x = torch.randn(4, 16, 256, device="cuda")
    output = tp_mlp(x)
    loss = output.sum()
    loss.backward()

    # DP AllReduce（在TP组外）
    dp_group = pgroups["dp"][0]
    for param in tp_mlp.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, group=dp_group)
            param.grad /= dp_size

    # 计算通讯量
    tp_comm = tp_mlp.get_comm_volume(4, 16) / 1024**2  # MB
    param_count = sum(p.numel() for p in tp_mlp.parameters())
    dp_comm = 2 * param_count * 4 / 1024**2  # MB

    if rank == 0:
        print(f"[DP+TP] Config: DP={dp_size}, TP={tp_size}")
        print(f"[DP+TP] TP Communication (AllReduce): {tp_comm:.2f} MB/layer")
        print(f"[DP+TP] DP Communication (AllReduce): {dp_comm:.2f} MB/step")
        print(f"[DP+TP] Total per step: {tp_comm + dp_comm:.2f} MB")

    dist.destroy_process_group()


def test_moe_ep():
    """测试专家并行 (EP)"""
    rank, local_rank, world_size = setup_distributed()

    if world_size < 2:
        print("EP test requires at least 2 GPUs")
        return

    if rank == 0:
        print("\n" + "=" * 60)
        print("Testing Expert Parallel (EP) - MoE")
        print("=" * 60)

    config = ParallelConfig(dp_size=1, tp_size=1, pp_size=1, ep_size=world_size)
    pgroups = create_process_groups(config, rank)

    moe = ExpertParallelMoE(
        256, num_experts=8, top_k=2, ep_group=pgroups["ep"][0]
    ).cuda()

    x = torch.randn(2, 16, 256, device="cuda")
    output = moe(x)

    comm_vol = moe.get_comm_volume(2 * 16) / 1024**2  # MB

    if rank == 0:
        print(f"[EP] Input shape: {x.shape}, Output shape: {output.shape}")
        print(f"[EP] Number of experts: 8, Experts per rank: {moe.experts_per_rank}")
        print(f"[EP] Communication (2x All-to-All): {comm_vol:.2f} MB")
        print(f"[EP] Pattern: All-to-All(dispatch) + All-to-All(combine)")

    dist.destroy_process_group()


def print_communication_summary():
    """打印通讯量总结"""
    print("\n" + "=" * 60)
    print("Megatron 并行策略通讯量总结")
    print("=" * 60)

    summary = """
    符号定义:
    - Ψ: 模型参数量
    - N: 总GPU数
    - D: DP size
    - T: TP size
    - P: PP size
    - E: EP size
    - b: batch size per DP rank
    - s: sequence length
    - h: hidden dimension
    - L: 层数

    ┌─────────────────────────────────────────────────────────────────┐
    │ 并行策略        │ 通讯原语        │ 通讯量                       │
    ├─────────────────────────────────────────────────────────────────┤
    │ DP             │ AllReduce       │ 2Ψ per step                 │
    ├─────────────────────────────────────────────────────────────────┤
    │ TP             │ AllReduce       │ 4L×b×s×h×(T-1)/T per step   │
    │                │ (每层2次)       │                             │
    ├─────────────────────────────────────────────────────────────────┤
    │ PP             │ P2P Send/Recv   │ 2(P-1)×b×s×h per step       │
    ├─────────────────────────────────────────────────────────────────┤
    │ EP (MoE)       │ All-to-All      │ 2×(E-1)/E×num_tokens×h×2    │
    │                │ (dispatch+      │ per MoE layer               │
    │                │  combine)       │                             │
    ├─────────────────────────────────────────────────────────────────┤
    │ DP+TP          │ AllReduce(DP)   │ 2Ψ/D + 4L×b×s×h×(T-1)/T     │
    │                │ AllReduce(TP)   │                             │
    ├─────────────────────────────────────────────────────────────────┤
    │ DP+TP+PP       │ AllReduce(DP)   │ 2Ψ/(D×T) +                  │
    │ (3D并行)       │ AllReduce(TP)   │ 4L×b×s×h×(T-1)/(T×P) +      │
    │                │ P2P(PP)         │ 2(P-1)×b×s×h                │
    ├─────────────────────────────────────────────────────────────────┤
    │ DP+TP+PP+EP    │ AllReduce(DP)   │ 2Ψ_eff/(D×T×E×P) +          │
    │ (4D并行)       │ AllReduce(TP)   │ 4L_dense×b×s×h×(T-1)/(T×P) +│
    │                │ P2P(PP)         │ 2(P-1)×b×s×h +              │
    │                │ All-to-All(EP)  │ 2×(E-1)/E×b×s×h×L_moe       │
    └─────────────────────────────────────────────────────────────────┘

    通讯优化建议:
    1. TP通讯最频繁（每层2次），建议放在同一节点内（NVLink）
    2. PP通讯量小但延迟敏感，建议高速互联
    3. DP通讯可以overlap计算，使用bucketized AllReduce
    4. EP All-to-All需要高效的网络拓扑支持
    """
    print(summary)


if __name__ == "__main__":
    # 根据可用GPU数选择测试
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    # 始终打印总结
    if rank == 0:
        print_communication_summary()

    # 根据GPU数量运行不同测试
    if world_size >= 1:
        test_dp()

    if world_size >= 2:
        test_tp()
        test_moe_ep()

    if world_size >= 4:
        test_dp_tp()

    print("\n所有测试完成!")
