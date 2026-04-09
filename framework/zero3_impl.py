"""
ZeRO Stage 3: 参数 + 梯度 + 优化器状态分片

  - 逐 tensor 分配 owner_rank，只有 owner_rank 持久存储该 tensor 的参数、梯度、优化器状态
  - forward 时逐 tensor 从 owner Broadcast 参数，用完立即释放
  - backward 时逐 tensor Reduce 梯度到 owner
  - 内存效率最高：持久存储 ≈ 1/N 的参数 + 1/N 的梯度 + 1/N 的优化器状态

  与 ZeRO-2 的区别：
    - ZeRO-2: 每个 rank 持有完整参数，只分片梯度和优化器状态
    - ZeRO-3: 参数也分片，forward/backward 时需要动态 Broadcast

  内存节省：
    - 参数: 分片存储（节省 1 - 1/N）
    - 梯度: 分片存储（节省 1 - 1/N）
    - 优化器状态: 分片存储（节省 1 - 1/N）
    - 总共节省约 1 - 1/N 的内存

运行方式:
    torchrun --nproc_per_node=4 zero3_impl.py
"""

import os

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


class ZeRO3Optimizer:
    """
    ZeRO Stage 3: 参数 + 梯度 + 优化器状态分片

    每个参数 tensor 分配给一个确定的 owner_rank。
    - owner_rank: 持久存储该 tensor 的参数副本、Adam 优化器状态 (m, v)
    - 非 owner_rank: 不持久存储参数，forward 时从 owner 接收，backward 后立即释放

    这样每个 rank 的持久内存只占 1/N 的参数 + 1/N 的梯度 + 1/N 的优化器状态。
    """

    def __init__(self, params, lr=0.001, world_size=1, rank=0):
        self.world_size = world_size
        self.rank = rank
        self.lr = lr

        self.all_params = list(params)
        self.t = 0

        # 逐 tensor 分配 owner: 按顺序轮询分配，确保均匀
        self.param_info = []  # [(param, owner_rank)]
        total_params = 0
        for i, p in enumerate(self.all_params):
            owner = i % world_size
            self.param_info.append((p, owner))
            total_params += p.numel()

        self.total_params = total_params

        # 只有 owner_rank 持久保存参数副本和优化器状态
        self.param_shards = {}  # param_id -> param_data (owner only)
        self.optim_states = {}  # param_id -> (m, v)
        self.grad_shards = {}  # param_id -> grad (owner only)
        owned_count = 0
        for i, (p, owner) in enumerate(self.param_info):
            if owner == rank:
                self.param_shards[id(p)] = p.data.clone()
                self.optim_states[id(p)] = (
                    torch.zeros_like(p.data),  # m: 一阶矩
                    torch.zeros_like(p.data),  # v: 二阶矩
                )
                self.grad_shards[id(p)] = torch.zeros_like(p.data)
                owned_count += 1

        # 非 owner: 释放参数内存（持久化只保留 owner 的分片）
        for p, owner in self.param_info:
            if owner != self.rank:
                p.data.zero_()

        print(
            f"[ZeRO-3] Rank {rank}: 负责其中 {owned_count}/{len(self.all_params)} 个 "
            f"参数 tensor, 持久存储: "
            f"{sum(p.numel() for p, o in self.param_info if o == rank)}"
            f"/{self.total_params}"
        )

    def forward_backward(self, model, input_data):
        """
        前向 + 反向传播 — 逐 tensor:

        forward 前（对每个参数 tensor）:
          1. owner Broadcast 完整参数给所有 rank
          2. 非 owner 接收到完整参数后，填充到模型参数（临时）

        forward:
          3. 所有 rank 持有完整参数，正常前向传播

        backward 后（对每个参数 tensor）:
          4. Reduce: 梯度归约到 owner_rank
          5. owner 保存梯度，非 owner 释放参数和梯度
        """
        # ========== forward 前: 逐 tensor Broadcast 参数 ==========
        for p, owner in self.param_info:
            # owner 广播完整参数，非 owner 接收
            dist.broadcast(p.data, src=owner)

        # ========== 前向传播 ==========
        output = model(input_data)
        loss = output.sum()

        # ========== 反向传播 ==========
        loss.backward()

        # ========== backward 后: 逐 tensor Reduce 梯度 + 释放参数 ==========
        for p, owner in self.param_info:
            # Reduce 梯度到 owner
            assert p.grad is not None, f"参数 {id(p)} 梯度为 None"
            dist.reduce(p.grad, dst=owner, op=dist.ReduceOp.SUM)

            if owner == self.rank:
                # owner: 保存平均梯度
                self.grad_shards[id(p)].copy_(p.grad.div_(self.world_size))
                # 释放梯度（已经保存到 grad_shards）
                p.grad = None
                # 注意：owner 仍然持有完整参数 p.data（上面 Broadcast 接收的）
                # 不需要做 anything，p.data 在 step() 更新 param_shards 后
                # 会在下次 forward_backward 的 Broadcast 中更新
            else:
                # 非 owner: 释放梯度
                p.grad = None
                # 释放参数（清零，下次 forward 前从 owner 接收）
                p.data.zero_()

        return loss

    def step(self):
        """
        单步优化 — 只更新 owner_rank 持久化的参数分片

        owner_rank 用 Reduce 得到的平均梯度和自己的优化器状态做 Adam，
        更新本地持久化的参数副本 param_shard。
        """
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for p, owner in self.param_info:
            if owner != self.rank:
                continue

            m, v = self.optim_states[id(p)]
            grad = self.grad_shards[id(p)]

            m.mul_(beta1).add_(grad, alpha=1 - beta1)
            v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            m_hat = m / (1 - beta1**self.t)
            v_hat = v / (1 - beta2**self.t)

            self.param_shards[id(p)].add_(-self.lr * m_hat / (v_hat + eps))

            # 同步更新 p.data，使其在下一次 forward 时 Broadcast 的是更新后的值
            p.data.copy_(self.param_shards[id(p)])

    def print_memory_stats(self):
        """打印内存统计"""
        owned_elements = sum(p.numel() for p, o in self.param_info if o == self.rank)

        # 参数分片（owner 持久存储）
        param_mem = owned_elements * 4 / 1024**2
        # 梯度分片（owner 持久存储）
        grad_mem = owned_elements * 4 / 1024**2
        # 优化器状态 (m, v)（owner 持久存储）
        opt_mem = owned_elements * 2 * 4 / 1024**2

        total = param_mem + grad_mem + opt_mem
        baseline = self.total_params * 4 * 4 / 1024**2  # 原始 DP

        print(
            f"[ZeRO-3] Rank {self.rank}: 参数分片={param_mem:.1f}MB, "
            f"梯度={grad_mem:.1f}MB, 优化器状态={opt_mem:.1f}MB, "
            f"总计={total:.1f}MB (节省: {baseline - total:.1f}MB, "
            f"{100 * (baseline - total) / baseline:.1f}%)"
        )


if __name__ == "__main__":
    rank, local_rank, world_size = setup_distributed()

    if rank == 0:
        print("\n" + "=" * 60)
        print("Testing ZeRO-3: Parameter + Gradient + Optimizer State Sharding")
        print("=" * 60)

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

    dist.destroy_process_group()
