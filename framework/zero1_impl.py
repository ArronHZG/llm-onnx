"""
ZeRO Stage 1: 优化器状态分片

  - 每个参数 tensor 独立分配给一个 owner_rank
  - 只有 owner_rank 存储该 tensor 的 Adam 优化器状态 (m, v)
  - 梯度仍然 AllReduce 同步（与常规 DP 相同），每个 rank 都持有完整梯度
  - step() 中: 逐 tensor AllReduce → owner 做 Adam → Broadcast 参数更新

  内存节省：
    - 完整梯度: 保留（与常规 DP 相同）
    - 优化器状态 (m, v): 分片存储，节省 1 - 1/N

运行方式:
    torchrun --nproc_per_node=4 zero1_impl.py
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


class ZeRO1Optimizer:
    """
    ZeRO Stage 1: 优化器状态分片

    每个参数 tensor 分配给一个确定的 owner_rank，
    只有 owner_rank 存储该 tensor 的 Adam 优化器状态 (m, v)。
    """

    def __init__(self, params, lr=0.001, world_size=1, rank=0):
        self.world_size = world_size
        self.rank = rank
        self.lr = lr

        self.all_params = list(params)
        self.t = 0

        # 逐 tensor 分配 owner: 按顺序轮询分配，确保均匀
        # 这与 DeepSpeed 的 partition_param_count 逻辑一致
        self.param_info = []  # [(param, owner_rank)]
        total_params = 0
        for i, p in enumerate(self.all_params):
            owner = i % world_size
            self.param_info.append((p, owner))
            total_params += p.numel()

        self.total_params = total_params

        # 只为当前 rank 负责的参数分配优化器状态
        self.optim_states = {}  # param_id -> (m, v)
        owned_count = 0
        for i, (p, owner) in enumerate(self.param_info):
            if owner == rank:
                self.optim_states[id(p)] = (
                    torch.zeros_like(p.data),  # m: 一阶矩
                    torch.zeros_like(p.data),  # v: 二阶矩
                )
                owned_count += 1

        print(
            f"[ZeRO-1] Rank {rank}: 负责其中 {owned_count}/{len(self.all_params)} 个 "
            f"参数 tensor, 优化器状态: {sum(p.numel() for p, o in self.param_info if o == rank)}"
            f"/{self.total_params}"
        )

    def step(self):
        """
        单步优化 — 逐 tensor:

        对每个参数 tensor:
          1. AllReduce 同步梯度（与常规 DP 相同）
          2. owner_rank 用 Adam 更新自己负责的 tensor
          3. AllGather 该 tensor 的更新结果
        """
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for p, owner in self.param_info:
            # 1. 逐 tensor AllReduce 同步梯度
            assert p.grad is not None, f"参数 {id(p)} 梯度为 None"
            grad = p.grad
            dist.all_reduce(grad, op=dist.ReduceOp.SUM)
            grad.div_(self.world_size)

            # 2. owner 做 Adam 更新
            if owner == self.rank:
                m, v = self.optim_states[id(p)]
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                m_hat = m / (1 - beta1**self.t)
                v_hat = v / (1 - beta2**self.t)

                p.data.add_(-self.lr * m_hat / (v_hat + eps))

            # 3. Broadcast: owner 广播更新后的参数给所有 rank
            dist.broadcast(p.data, src=owner)

            # 清零梯度
            p.grad = None

    def print_memory_stats(self):
        """打印内存统计"""
        # 参数内存：每个 rank 都持有完整参数（ZeRO-1 不分片参数）
        param_mem = self.total_params * 4 / 1024**2  # MB, 假设 float32

        # 梯度内存：与参数同大小（ZeRO-1 保留完整梯度）
        grad_mem = param_mem

        # 优化器状态内存：只存储自己负责的 tensor 的 m, v
        owned_elements = sum(p.numel() for p, o in self.param_info if o == self.rank)
        opt_mem = owned_elements * 2 * 4 / 1024**2  # m, v

        total = param_mem + grad_mem + opt_mem
        baseline = param_mem * 4  # 原始 DP: 参数 + 梯度 + m + v

        print(
            f"[ZeRO-1] Rank {self.rank}: 参数={param_mem:.1f}MB, "
            f"梯度={grad_mem:.1f}MB, 优化器状态={opt_mem:.1f}MB, "
            f"总计={total:.1f}MB (优化器状态节省: "
            f"{100 * (1 - opt_mem / (param_mem * 2)):.1f}%)"
        )


if __name__ == "__main__":
    rank, local_rank, world_size = setup_distributed()

    if rank == 0:
        print("\n" + "=" * 60)
        print("Testing ZeRO-1: Optimizer State Sharding")
        print("=" * 60)

    torch.manual_seed(42)
    model = SimpleModel(dim=256, num_layers=4).cuda()
    optimizer = ZeRO1Optimizer(
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
        print("ZeRO-1 测试完成\n")

    dist.destroy_process_group()
