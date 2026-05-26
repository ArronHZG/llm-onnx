"""
ZeRO (Zero Redundancy Optimizer) 最简化实现
包含 ZeRO-1, ZeRO-2, ZeRO-3 的通讯原理演示

运行方式:
    torchrun --nproc_per_node=4 zero_impl.py
"""

import os
from time import sleep

import torch
import torch.distributed as dist
import torch.nn as nn


def setup_distributed():
    """初始化分布式环境"""
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    return rank, local_rank, world_size


class SimpleModel(nn.Module):
    """简单模型用于测试"""

    def __init__(self, dim=128, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ZeRO1Optimizer:
    """
    ZeRO Stage 1: 优化器状态分片

    每个rank只存储1/world_size的优化器状态
    梯度使用AllReduce同步（与常规DP相同）
    """

    def __init__(self, params, lr=0.001, world_size=1, rank=0):
        self.world_size = world_size
        self.rank = rank
        self.lr = lr

        # 扁平化所有参数
        self.all_params = list(params)
        self.total_params = sum(p.numel() for p in self.all_params)

        # 每个rank负责的参数范围
        params_per_rank = (self.total_params + world_size - 1) // world_size
        self.start_idx = rank * params_per_rank
        self.end_idx = min((rank + 1) * params_per_rank, self.total_params)

        # 只存储自己负责的优化器状态 (Adam: m, v)
        shard_size = self.end_idx - self.start_idx
        self.m = torch.zeros(shard_size, device="cuda")  # 一阶矩
        self.v = torch.zeros(shard_size, device="cuda")  # 二阶矩
        self.t = 0

        print(
            f"[ZeRO-1] Rank {rank}: 负责参数索引 [{self.start_idx}, {self.end_idx}), "
            f"优化器状态分片大小: {shard_size}/{self.total_params}"
        )

    def flatten_params(self):
        """将所有参数扁平化为一个tensor"""
        return torch.cat([p.view(-1) for p in self.all_params])

    def flatten_grads(self):
        """将所有梯度扁平化为一个tensor"""
        return torch.cat([p.grad.view(-1) for p in self.all_params])

    def unflatten_to_params(self, flat_tensor):
        """将扁平化tensor还原到参数"""
        offset = 0
        for p in self.all_params:
            numel = p.numel()
            p.data.copy_(flat_tensor[offset : offset + numel].view_as(p))
            offset += numel

    def step(self):
        """单步优化"""
        self.t += 1

        # 1. AllReduce 梯度 (与常规DP相同)
        flat_grads = self.flatten_grads()
        dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)
        flat_grads /= self.world_size  # 平均梯度

        # 2. 获取当前rank负责的梯度分片
        grad_shard = flat_grads[self.start_idx : self.end_idx]

        # 3. Adam更新 (只更新自己负责的参数)
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        self.m = beta1 * self.m + (1 - beta1) * grad_shard
        self.v = beta2 * self.v + (1 - beta2) * grad_shard * grad_shard

        m_hat = self.m / (1 - beta1**self.t)
        v_hat = self.v / (1 - beta2**self.t)

        update_shard = -self.lr * m_hat / (v_hat.sqrt() + eps)

        # 4. AllGather 更新后的参数分片（使用高效的原生 all_gather）
        flat_params = self.flatten_params()

        # 收集所有rank的参数更新
        # 方法1: all_gather_into_tensor
        # update_shard = update_shard.cuda()
        gathered_updates = torch.empty(self.total_params, device="cuda")
        dist.all_gather_into_tensor(gathered_updates, update_shard)

        # 应用更新：flat_params + 所有rank的更新
        self.unflatten_to_params(flat_params + gathered_updates)

        # 清零梯度
        for p in self.all_params:
            p.grad = None

    def print_memory_stats(self):
        """打印内存统计"""
        param_mem = sum(p.numel() for p in self.all_params) * 4 / 1024**2  # MB
        grad_mem = param_mem  # 梯度与参数同大小
        opt_mem = (self.end_idx - self.start_idx) * 2 * 4 / 1024**2  # m, v

        print(
            f"[ZeRO-1] Rank {self.rank}: 参数={param_mem:.1f}MB, "
            f"梯度={grad_mem:.1f}MB, 优化器状态={opt_mem:.1f}MB "
            f"(节省: {(2 - 2 / self.world_size) * param_mem:.1f}MB)"
        )


class ZeRO2Optimizer:
    """
    ZeRO Stage 2: 梯度 + 优化器状态分片

    核心思想：backward 完成后，用 ReduceScatter 替代 AllReduce 同步梯度，
    每个 rank 只保留自己负责的梯度分片，其余立即释放。

    与 ZeRO-1 的区别：
    - ZeRO-1: backward 后 AllReduce 同步梯度 → 每个 rank 持有完整梯度 → step
    - ZeRO-2: backward 后 ReduceScatter 梯度 → 每个 rank 只持分片 → step
    通讯量: ReduceScatter 仅为 AllReduce 的一半

    与 ZeRO-3 的区别：
    - ZeRO-2: 每个 rank 持有完整的模型参数，只分片梯度和优化器状态
    - ZeRO-3: 参数也分片，forward/backward 时需要动态 all_gather
    """

    def __init__(self, params, lr=0.001, world_size=1, rank=0):
        self.world_size = world_size
        self.rank = rank
        self.lr = lr

        self.all_params = list(params)
        self.total_params = sum(p.numel() for p in self.all_params)

        # 每个rank负责的参数范围
        params_per_rank = (self.total_params + world_size - 1) // world_size
        self.start_idx = rank * params_per_rank
        self.end_idx = min((rank + 1) * params_per_rank, self.total_params)
        shard_size = self.end_idx - self.start_idx

        # 优化器状态（分片存储）
        self.m = torch.zeros(shard_size, device="cuda")
        self.v = torch.zeros(shard_size, device="cuda")
        self.t = 0

        # 梯度分片缓冲区
        self.grad_shard = torch.zeros(shard_size, device="cuda")

        print(
            f"[ZeRO-2] Rank {rank}: 负责参数索引 [{self.start_idx}, {self.end_idx}), "
            f"梯度+优化器状态分片大小: {shard_size}/{self.total_params}"
        )

    def step(self):
        """
        单步优化：
        1. 扁平化梯度
        2. ReduceScatter: 归约 + 分散，每个 rank 只拿到自己的梯度分片
        3. 释放完整梯度
        4. Adam 更新
        5. AllGather 参数更新
        """
        self.t += 1

        # 1. 扁平化梯度
        flat_grads = torch.cat([p.grad.view(-1) for p in self.all_params])

        # 2. ReduceScatter: 梯度归约并分散到各rank
        #    输入: 完整梯度 (total_params)
        #    输出: 每个rank只拿到 1/world_size 的梯度
        dist.reduce_scatter_tensor(self.grad_shard, flat_grads, op=dist.ReduceOp.AVG)

        # 3. 立即释放完整梯度内存
        del flat_grads
        for p in self.all_params:
            p.grad = None

        # 4. Adam更新（只更新自己的分片）
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        self.m = beta1 * self.m + (1 - beta1) * self.grad_shard
        self.v = beta2 * self.v + (1 - beta2) * self.grad_shard * self.grad_shard

        m_hat = self.m / (1 - beta1**self.t)
        v_hat = self.v / (1 - beta2**self.t)

        update_shard = -self.lr * m_hat / (v_hat.sqrt() + eps)

        # 5. AllGather: 收集更新后的完整参数
        flat_params = self.flatten_params()
        flat_params[self.start_idx : self.end_idx] += update_shard

        gathered_params = torch.empty(self.total_params, device="cuda")
        dist.all_gather_into_tensor(
            gathered_params, flat_params[self.start_idx : self.end_idx]
        )

        self.unflatten_to_params(gathered_params)

    def flatten_params(self):
        return torch.cat([p.view(-1) for p in self.all_params])

    def unflatten_to_params(self, flat_tensor):
        offset = 0
        for p in self.all_params:
            numel = p.numel()
            p.data.copy_(flat_tensor[offset : offset + numel].view_as(p))
            offset += numel

    def print_memory_stats(self):
        """打印内存统计 - 比ZeRO-1节省更多内存"""
        param_mem = sum(p.numel() for p in self.all_params) * 4 / 1024**2
        grad_mem = (self.end_idx - self.start_idx) * 4 / 1024**2  # 梯度分片
        opt_mem = (self.end_idx - self.start_idx) * 2 * 4 / 1024**2  # m, v

        total = param_mem + grad_mem + opt_mem
        baseline = param_mem * 4  # 原始DP的内存占用

        print(
            f"[ZeRO-2] Rank {self.rank}: 参数={param_mem:.1f}MB, "
            f"梯度(分片)={grad_mem:.1f}MB, 优化器状态(分片)={opt_mem:.1f}MB, "
            f"总计={total:.1f}MB (节省: {baseline - total:.1f}MB, "
            f"{100 * (baseline - total) / baseline:.1f}%)"
        )


class ZeRO3Optimizer:
    """
    ZeRO Stage 3: 参数 + 梯度 + 优化器状态分片

    每个rank只存储1/world_size的参数，前向/反向时需要AllGather参数
    通讯量最大，但内存效率最高
    """

    def __init__(self, params, lr=0.001, world_size=1, rank=0):
        self.world_size = world_size
        self.rank = rank
        self.lr = lr

        self.all_params = list(params)
        self.total_params = sum(p.numel() for p in self.all_params)

        params_per_rank = (self.total_params + world_size - 1) // world_size
        self.start_idx = rank * params_per_rank
        self.end_idx = min((rank + 1) * params_per_rank, self.total_params)

        shard_size = self.end_idx - self.start_idx
        self.m = torch.zeros(shard_size, device="cuda")
        self.v = torch.zeros(shard_size, device="cuda")
        self.grad_shard = torch.zeros(shard_size, device="cuda")

        # ZeRO3: 只保存参数分片
        self.param_shard = self._get_param_shard().cuda()
        self.t = 0

        # 释放完整参数，只保留分片
        # 注意：这里不能改变参数形状，否则模型 forward 会失败
        # 实际 ZeRO 会使用元数据来跟踪参数，这里简化处理

        print(
            f"[ZeRO-3] Rank {rank}: 参数分片大小: {shard_size}/{self.total_params} "
            f"({100 * shard_size / self.total_params:.1f}%)"
        )

    def _get_param_shard(self):
        """获取自己的参数分片"""
        flat_params = torch.cat([p.view(-1) for p in self.all_params])
        return flat_params[self.start_idx : self.end_idx].clone()

    def all_gather_params(self):
        """AllGather: 从所有rank收集完整参数（使用高效的原生 all_gather）"""
        # 使用 all_gather_into_tensor 高效收集所有rank的参数
        full_params = torch.empty(self.total_params, device="cuda")
        dist.all_gather_into_tensor(full_params, self.param_shard)
        return full_params

    def forward_backward(self, model, input_data):
        """
        前向+反向传播
        需要动态收集参数，计算完成后释放
        """
        # 保存原始形状
        self.param_shapes = [p.shape for p in self.all_params]

        # 1. AllGather参数
        full_params = self.all_gather_params()

        # 2. 还原到模型参数（临时）
        offset = 0
        for i, p in enumerate(self.all_params):
            shape = self.param_shapes[i]
            p.data = full_params[offset : offset + p.numel()].view(shape)
            offset += p.numel()

        # 3. 前向传播
        output = model(input_data)
        loss = output.sum()

        # 4. 反向传播
        loss.backward()

        # 5. 提取梯度
        flat_grads = torch.cat([p.grad.view(-1) for p in self.all_params])

        # 6. ReduceScatter: 梯度分散到各rank（使用原生 reduce_scatter）
        dist.reduce_scatter_tensor(self.grad_shard, flat_grads, op=dist.ReduceOp.AVG)

        # 7. 清零梯度
        for p in self.all_params:
            p.grad = None

        return loss

    def step(self):
        """单步优化 - 只更新自己的参数分片"""
        self.t += 1

        beta1, beta2, eps = 0.9, 0.999, 1e-8

        self.m = beta1 * self.m + (1 - beta1) * self.grad_shard
        self.v = beta2 * self.v + (1 - beta2) * self.grad_shard * self.grad_shard

        m_hat = self.m / (1 - beta1**self.t)
        v_hat = self.v / (1 - beta2**self.t)

        update = -self.lr * m_hat / (v_hat.sqrt() + eps)
        self.param_shard += update

    def print_memory_stats(self):
        """打印内存统计 - 最大内存节省"""
        param_shard_mem = (self.end_idx - self.start_idx) * 4 / 1024**2
        grad_mem = (self.end_idx - self.start_idx) * 4 / 1024**2
        opt_mem = (self.end_idx - self.start_idx) * 2 * 4 / 1024**2

        total = param_shard_mem + grad_mem + opt_mem
        baseline = self.total_params * 4 * 4 / 1024**2  # 原始DP

        print(
            f"[ZeRO-3] Rank {self.rank}: 参数分片={param_shard_mem:.1f}MB, "
            f"梯度={grad_mem:.1f}MB, 优化器状态={opt_mem:.1f}MB, "
            f"总计={total:.1f}MB (节省: {baseline - total:.1f}MB, "
            f"{100 * (baseline - total) / baseline:.1f}%)"
        )


def test_zero1_with_init(rank, local_rank, world_size):
    """测试ZeRO-1"""
    if rank == 0:
        print("\n" + "=" * 60)
        print("Testing ZeRO-1: Optimizer State Sharding")
        print("=" * 60)
    sleep(10)

    torch.manual_seed(42)
    model = SimpleModel(dim=256, num_layers=4).cuda()
    optimizer = ZeRO1Optimizer(
        model.parameters(), lr=0.001, world_size=world_size, rank=rank
    )

    optimizer.print_memory_stats()

    # 模拟训练
    for step in range(3):
        x = torch.randn(4, 256, device="cuda")
        output = model(x)
        loss = output.sum()
        loss.backward()

        optimizer.step()

        if rank == 0:
            print(f"Step {step}: loss = {loss.item():.4f}")

    dist.barrier()
    if rank == 0:
        print("ZeRO-1 测试完成\n")


def test_zero2_with_init(rank, local_rank, world_size):
    """测试ZeRO-2"""
    if rank == 0:
        print("\n" + "=" * 60)
        print("Testing ZeRO-2: Gradient + Optimizer State Sharding")
        print("=" * 60)
    sleep(10)

    torch.manual_seed(42)
    model = SimpleModel(dim=256, num_layers=4).cuda()
    optimizer = ZeRO2Optimizer(
        model.parameters(), lr=0.001, world_size=world_size, rank=rank
    )

    optimizer.print_memory_stats()

    for step in range(3):
        x = torch.randn(4, 256, device="cuda")
        output = model(x)
        loss = output.sum()
        loss.backward()

        optimizer.step()

        if rank == 0:
            print(f"Step {step}: loss = {loss.item():.4f}")

    dist.barrier()
    if rank == 0:
        print("ZeRO-2 测试完成\n")


def test_zero3_with_init(rank, local_rank, world_size):
    """测试ZeRO-3"""
    if rank == 0:
        print("\n" + "=" * 60)
        print("Testing ZeRO-3: Parameter + Gradient + Optimizer State Sharding")
        print("=" * 60)
    sleep(10)

    torch.manual_seed(42)
    model = SimpleModel(dim=256, num_layers=4).cuda()
    optimizer = ZeRO3Optimizer(
        model.parameters(), lr=0.001, world_size=world_size, rank=rank
    )

    optimizer.print_memory_stats()

    for step in range(3):
        x = torch.randn(4, 256, device="cuda")
        loss = optimizer.forward_backward(model, x)
        optimizer.step()

        if rank == 0:
            print(f"Step {step}: loss = {loss.item():.4f}")

    dist.barrier()
    if rank == 0:
        print("ZeRO-3 测试完成\n")


if __name__ == "__main__":
    # 只初始化一次分布式环境
    rank, local_rank, world_size = setup_distributed()

    # 测试不同的ZeRO阶段
    test_zero1_with_init(rank, local_rank, world_size)
    test_zero2_with_init(rank, local_rank, world_size)
    test_zero3_with_init(rank, local_rank, world_size)

    dist.destroy_process_group()
