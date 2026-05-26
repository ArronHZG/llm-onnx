"""
PyTorch FSDP (Fully Sharded Data Parallel) 最简化实现

核心原理:
1. 参数、梯度、优化器状态都分片到所有GPU
2. 前向传播时用 all_gather 临时收集参数
3. 反向传播时用 reduce_scatter 分散梯度

运行方式:
    torchrun --nproc_per_node=4 fsdp_impl.py
"""

import os
from typing import List

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


class SimpleLinear(nn.Module):
    """简单的线性层，用于演示"""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return nn.functional.linear(x, self.weight, self.bias)


class FSDPLayer:
    """
    FSDP 单层的实现
    将参数分片，按需收集和释放
    """

    def __init__(self, params: List[nn.Parameter], world_size: int, rank: int):
        self.world_size = world_size
        self.rank = rank
        self.params = params  # 原始参数引用

        # 保存每个参数的形状和元素数（在清零参数前记录）
        self.param_shapes = [p.shape for p in params]
        self.param_numels = [p.numel() for p in params]

        # 计算总参数量
        self.total_numel = sum(self.param_numels)

        # 计算分片范围
        shard_size = (self.total_numel + world_size - 1) // world_size
        self.shard_start = rank * shard_size
        self.shard_end = min((rank + 1) * shard_size, self.total_numel)
        self.shard_numel = self.shard_end - self.shard_start

        # 保存参数分片（ZeRO-3风格）
        with torch.no_grad():
            flat_params = torch.cat([p.view(-1).cuda() for p in params])
            self.param_shard = flat_params[self.shard_start : self.shard_end].clone()

        # 梯度分片
        self.grad_shard = torch.zeros(self.shard_numel, device="cuda")

        # 释放原始参数显存（FSDP关键）
        for p in params:
            p.data = torch.empty(0, device="cuda")

    def all_gather_params(self) -> torch.Tensor:
        """
        AllGather: 收集完整参数
        通讯量: (N-1)/N * total_params
        """
        full_params = torch.empty(self.total_numel, device="cuda")
        dist.all_gather_into_tensor(full_params, self.param_shard)
        return full_params

    def reduce_scatter_grads(self, flat_grads: torch.Tensor):
        """
        ReduceScatter: 梯度归约并分散
        通讯量: (N-1)/N * total_params
        """
        dist.reduce_scatter_tensor(self.grad_shard, flat_grads, op=dist.ReduceOp.AVG)

    def set_full_params(self, full_params: torch.Tensor):
        """将完整参数还原到模型"""
        offset = 0
        for i, p in enumerate(self.params):
            numel = self.param_numels[i]
            shape = self.param_shapes[i]
            p.data = full_params[offset : offset + numel].view(shape)
            offset += numel

    def clear_params(self):
        """释放完整参数，只保留分片"""
        for p in self.params:
            p.data = torch.empty(0, device="cuda")

    def get_flat_grads(self) -> torch.Tensor:
        """从模型参数中收集梯度并扁平化"""
        grad_list = []
        for i, p in enumerate(self.params):
            if p.grad is not None:
                grad_list.append(p.grad.view(-1))
            else:
                grad_list.append(torch.zeros(self.param_numels[i], device="cuda"))
        return torch.cat(grad_list)

    def update_shard(self, lr: float):
        """使用SGD更新参数分片"""
        self.param_shard -= lr * self.grad_shard


class FSDPModule(nn.Module):
    """
    FSDP 包装模块
    将模型的每一层包装为FSDP单元
    """

    def __init__(self, model: nn.Module, world_size: int, rank: int):
        super().__init__()
        self.model = model
        self.world_size = world_size
        self.rank = rank

        # 为每一层创建FSDP单元
        self.fsdp_units = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, SimpleLinear)):
                params = list(module.parameters())
                if params:
                    fsdp_unit = FSDPLayer(params, world_size, rank)
                    self.fsdp_units.append((name, module, fsdp_unit))

        if rank == 0:
            print(f"[FSDP] Created {len(self.fsdp_units)} FSDP units")
            for name, _, unit in self.fsdp_units:
                print(
                    f"  - {name}: {unit.total_numel} params, "
                    f"shard size: {unit.shard_numel}"
                )

    def forward(self, x):
        """
        前向传播流程:
        1. 对每一层执行AllGather获取参数
        2. 执行前向计算
        3. 释放参数
        """
        output = x

        for name, module, fsdp_unit in self.fsdp_units:
            # 1. AllGather参数
            full_params = fsdp_unit.all_gather_params()

            # 2. 设置到模块
            fsdp_unit.set_full_params(full_params)

            # 3. 前向计算
            output = module(output)

            # 4. 立即释放参数
            fsdp_unit.clear_params()

        return output

    def backward(self, loss):
        """
        反向传播流程:
        1. 重新AllGather参数（backward需要参数来计算梯度）
        2. 执行反向传播
        3. ReduceScatter梯度到各rank
        4. 释放参数
        """
        # backward前需要先收集所有参数（计算图需要）
        for name, module, fsdp_unit in self.fsdp_units:
            full_params = fsdp_unit.all_gather_params()
            fsdp_unit.set_full_params(full_params)

        # 执行反向传播
        loss.backward()

        # 梯度ReduceScatter + 释放参数
        for name, module, fsdp_unit in self.fsdp_units:
            flat_grads = fsdp_unit.get_flat_grads()
            fsdp_unit.reduce_scatter_grads(flat_grads)
            fsdp_unit.clear_params()
            for p in fsdp_unit.params:
                p.grad = None

    def step(self, lr: float = 0.001):
        """参数更新步骤"""
        for name, module, fsdp_unit in self.fsdp_units:
            fsdp_unit.update_shard(lr)

    def print_memory_stats(self):
        """打印内存统计"""
        total_shard_params = sum(unit.shard_numel for _, _, unit in self.fsdp_units)
        total_params = sum(unit.total_numel for _, _, unit in self.fsdp_units)

        param_mem = total_shard_params * 4 / 1024**2
        grad_mem = total_shard_params * 4 / 1024**2
        opt_mem = total_shard_params * 2 * 4 / 1024**2

        total_mem = param_mem + grad_mem + opt_mem
        baseline = total_params * 4 * 4 / 1024**2

        print(f"\n[FSDP] Rank {self.rank} Memory Stats:")
        print(f"  Total parameters: {total_params:,}")
        print(
            f"  Shard size: {total_shard_params:,} ({100 * total_shard_params / total_params:.1f}%)"
        )
        print(f"  Parameter memory: {param_mem:.2f} MB")
        print(f"  Gradient memory: {grad_mem:.2f} MB")
        print(f"  Optimizer memory: {opt_mem:.2f} MB")
        print(f"  Total per rank: {total_mem:.2f} MB")
        print(f"  Baseline (DP): {baseline:.2f} MB")
        print(
            f"  Memory saved: {baseline - total_mem:.2f} MB ({100 * (baseline - total_mem) / baseline:.1f}%)"
        )

    def get_comm_volume_per_step(self) -> dict:
        """
        计算每步通讯量

        前向: AllGather (每层)
        反向: AllGather (每层) + ReduceScatter (每层)
        """
        total_params = sum(unit.total_numel for _, _, unit in self.fsdp_units)
        n = self.world_size

        allgather_vol = (n - 1) / n * total_params * 4 / 1024**2
        reducescatter_vol = (n - 1) / n * total_params * 4 / 1024**2

        num_layers = len(self.fsdp_units)

        return {
            "num_layers": num_layers,
            "total_params": total_params,
            "forward_allgather_mb": allgather_vol * num_layers,
            "backward_allgather_mb": allgather_vol * num_layers,
            "backward_reducescatter_mb": reducescatter_vol * num_layers,
            "total_per_step_mb": (allgather_vol * 2 + reducescatter_vol) * num_layers,
        }


class SimpleModel(nn.Module):
    """测试用简单模型"""

    def __init__(self, dim=256, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([SimpleLinear(dim, dim) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def test_fsdp():
    """测试FSDP实现"""
    rank, local_rank, world_size = setup_distributed()

    if rank == 0:
        print("\n" + "=" * 60)
        print("Testing PyTorch FSDP-like Implementation")
        print("=" * 60)

    # 创建模型
    torch.manual_seed(42)
    base_model = SimpleModel(dim=256, num_layers=4)

    # 包装为FSDP
    fsdp_model = FSDPModule(base_model, world_size, rank)

    # 打印内存统计
    fsdp_model.print_memory_stats()

    # 通讯量分析
    if rank == 0:
        comm_stats = fsdp_model.get_comm_volume_per_step()
        print(f"\n[FSDP] Communication Volume per Step:")
        print(f"  Number of FSDP units: {comm_stats['num_layers']}")
        print(f"  Total parameters: {comm_stats['total_params']:,}")
        print(f"  Forward AllGather: {comm_stats['forward_allgather_mb']:.2f} MB")
        print(f"  Backward AllGather: {comm_stats['backward_allgather_mb']:.2f} MB")
        print(
            f"  Backward ReduceScatter: {comm_stats['backward_reducescatter_mb']:.2f} MB"
        )
        print(f"  Total communication: {comm_stats['total_per_step_mb']:.2f} MB")

    # 模拟训练
    if rank == 0:
        print("\n[FSDP] Training...")

    for step in range(5):
        # 所有 rank 使用相同的输入（通过 seed 保证），模拟 DP 行为
        torch.manual_seed(step + 100)
        x = torch.randn(8, 256, device="cuda")

        # 前向传播
        output = fsdp_model(x)
        loss = output.mean()

        # 反向传播
        fsdp_model.backward(loss)

        # 参数更新
        fsdp_model.step(lr=0.01)

        if rank == 0:
            print(f"  Step {step}: loss = {loss.item():.4f}")

    dist.barrier()
    if rank == 0:
        print("\nFSDP 测试完成!")

    dist.destroy_process_group()


if __name__ == "__main__":
    test_fsdp()
