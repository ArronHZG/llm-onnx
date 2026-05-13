# CUDA

### 21. CUDA实现vector add并绑定到Python

#### 实现思路

1. 用CUDA编写vector add核函数

2. 用PyBind11封装CUDA函数，编译为Python扩展模块

3. Python中调用扩展模块

#### 完整代码示例

**1. [vector_add.cu](vector_add.cu)**

```C++

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>

namespace py = pybind11;

// CUDA核函数
__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// 封装函数
py::array_t<float> vector_add(py::array_t<float> a_arr, py::array_t<float> b_arr) {
    // 获取数组信息
    auto a = a_arr.unchecked<1>();
    auto b = b_arr.unchecked<1>();
    int n = a.size();
    
    // 检查输入维度
    if (b.size() != n) {
        throw std::runtime_error("Input arrays must have the same size");
    }
    
    // 分配设备内存
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));
    
    // 拷贝数据到设备
    cudaMemcpy(d_a, a.data(0), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(0), n * sizeof(float), cudaMemcpyHostToDevice);
    
    // 启动核函数
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    vector_add_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
    
    // 创建输出数组
    auto c_arr = py::array_t<float>(n);
    auto c = c_arr.mutable_unchecked<1>();
    
    // 拷贝结果到主机
    cudaMemcpy(c.mutable_data(0), d_c, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 释放设备内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return c_arr;
}

PYBIND11_MODULE(vector_add, m) {
    m.def("vector_add", &vector_add, "CUDA vector addition",
          py::arg("a"), py::arg("b"));
}
```

**2. [setup.py](setup.py)**

```Python

from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import sys

ext_modules = [
    Pybind11Extension(
        "vector_add",
        ["vector_add.cu"],
        extra_compile_args={
            "nvcc": ["-O3", "-arch=sm_75"]  # 根据GPU架构调整
        },
        language="c++"
    ),
]

setup(
    name="vector_add",
    version="0.1",
    author="Your Name",
    description="CUDA vector add with Python binding",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
```

**3. 编译和使用**

```Bash

# 编译
python setup.py build_ext --inplace

# Python调用示例
import numpy as np
import vector_add

a = np.random.rand(10000).astype(np.float32)
b = np.random.rand(10000).astype(np.float32)
c = vector_add.vector_add(a, b)

# 验证结果
assert np.allclose(c, a + b)
```

### 22. Triton和CUDA的区别及适用场景

#### 核心区别

|特性|CUDA|Triton|
|---|---|---|
|抽象层级|底层，直接控制硬件|高层，自动优化|
|编程复杂度|高，需手动处理内存/同步|低，Pythonic接口|
|优化难度|需手动优化（coalescing/occupancy等）|编译器自动优化|
|灵活性|极高，可实现任意逻辑|中等，受限于抽象|
|性能调优|依赖开发者经验|自动调优，搜索最优配置|
#### Triton更合适的场景

1. **快速原型开发**：无需深入CUDA细节即可实现高性能算子

2. **矩阵运算为主**：Triton对GEMM类运算优化极佳

3. **动态shape场景**：自动适配不同输入尺寸

4. **减少优化成本**：避免手动处理memory coalescing等细节

5. **Python生态集成**：原生支持Python，无缝对接PyTorch

#### CUDA更合适的场景

1. **极致性能优化**：需要精细控制硬件行为

2. **非标准运算**：复杂控制流、特殊内存访问模式

3. **硬件特性深度利用**：如tensor core、原子操作等

4. **已有成熟CUDA代码**：无需迁移

### 23. 编写layernorm + residual + activation融合算子

#### 实现思路

1. 合并多个算子的计算逻辑，减少内存读写

2. 一次kernel调用完成所有计算

3. 优化内存访问模式

#### CUDA实现示例

```C++

#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename T>
__global__ void fused_layernorm_residual_activation_kernel(
    const T* input, const T* residual, T* output,
    const T* gamma, const T* beta,
    int batch_size, int hidden_size, float eps) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * hidden_size) return;
    
    int b = idx / hidden_size;
    int h = idx % hidden_size;
    
    // 1. Residual add (input + residual)
    T val = input[idx] + residual[idx];
    
    // 2. LayerNorm计算 (简化版，实际需优化)
    // 注：实际实现需要先计算均值和方差，这里简化展示融合逻辑
    __shared__ T s_mean, s_var;
    if (threadIdx.x == 0) {
        // 实际中需要高效计算均值方差
        s_mean = 0; s_var = 0;
    }
    __syncthreads();
    
    val = (val - s_mean) / sqrt(s_var + eps);
    val = val * gamma[h] + beta[h];
    
    // 3. Activation (GELU)
    val = val * 0.5 * (1.0 + tanh(sqrt(2.0/M_PI) * (val + 0.044715 * pow(val, 3))));
    
    output[idx] = val;
}

torch::Tensor fused_layernorm_residual_activation(
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps = 1e-5) {
    
    const int batch_size = input.size(0);
    const int hidden_size = input.size(1);
    auto output = torch::empty_like(input);
    
    dim3 block_size(256);
    dim3 grid_size((batch_size * hidden_size + block_size.x - 1) / block_size.x);
    
    AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_op", ([&] {
        fused_layernorm_residual_activation_kernel<scalar_t><<<grid_size, block_size>>>(
            input.data<scalar_t>(),
            residual.data<scalar_t>(),
            output.data<scalar_t>(),
            gamma.data<scalar_t>(),
            beta.data<scalar_t>(),
            batch_size,
            hidden_size,
            eps
        );
    }));
    
    return output;
}

PYBIND11_MODULE(fused_ops, m) {
    m.def("fused_layernorm_residual_activation", 
          &fused_layernorm_residual_activation,
          "Fused layernorm + residual + activation");
}
```

### 24. CUTLASS介绍及适用场景

#### CUTLASS是什么？

CUTLASS（CUDA Templates for Linear Algebra Subroutines）是NVIDIA开源的CUDA模板库，提供了高度优化的线性代数原语（如GEMM、卷积、attention等），封装了：

- 底层CUDA优化（memory coalescing、tensor core利用等）

- 不同数据类型（FP16/FP32/INT8等）的支持

- 不同GPU架构的适配

#### 何时使用CUTLASS而非手写CUDA？

1. **矩阵运算为主**：GEMM、卷积、矩阵乘加等操作

2. **利用Tensor Core**：CUTLASS原生支持Tensor Core，手写难度大

3. **多数据类型支持**：轻松切换FP16/FP8/INT4等

4. **快速开发**：避免重复造轮子，直接复用优化好的模板

5. **跨架构兼容**：自动适配不同GPU架构（SM_70/80/90等）

6. **性能要求高**：CUTLASS由NVIDIA官方优化，性能接近理论峰值

#### 何时仍需手写CUDA？

- 非矩阵运算的特殊算子

- 需要极致定制化的内存访问模式

- 复杂控制流的算子

- 融合多个非标准操作的场景

### 25. cuda算子优化核心概念

#### 1. Memory Coalescing（内存合并访问）

- **定义**：GPU线程束（warp）中的线程访问全局内存时，连续的内存地址被合并为更少的内存事务

- **重要性**：未合并访问会导致内存带宽利用率低（仅1/32）

- **优化目标**：让连续线程访问连续的内存地址

#### 2. Bank Conflict（存储体冲突）

- **定义**：共享内存被划分为32/48个bank，多个线程同时访问同一bank会导致串行访问

- **影响**：严重降低共享内存访问速度

- **解决方法**：内存填充（padding）、调整访问模式

#### 3. Occupancy（占用率）

- **定义**：GPU SM上同时驻留的活动线程束数 / 最大支持线程束数

- **影响**：高占用率有助于隐藏内存延迟

- **关键因素**：block大小、register使用量、shared memory使用量

### 26. Nsight Compute分析kernel性能

#### 使用步骤

1. **收集数据**：

```Bash

# 基础分析
ncu -o kernel_report --target-processes all python your_script.py

# 详细分析特定kernel
ncu -k vector_add_kernel -o detailed_report python your_script.py
```

1. **关键Metrics**：

    - **Memory Metrics**：

        - `Global Memory Throughput`：全局内存吞吐量

        - `Global Memory Load/Store Efficiency`：内存访问效率

        - `Shared Memory Bank Conflicts`：存储体冲突率

    - **Compute Metrics**：

        - `SM Active Cycles`：SM活跃周期

        - `Instruction Throughput`：指令吞吐量

        - `FLOP Count`：浮点运算数

    - **Occupancy Metrics**：

        - `Achieved Occupancy`：实际占用率

        - `Register/Thread`：每个线程使用的寄存器数

    - **Latency Metrics**：

        - `Memory Stall Cycles`：内存等待周期

        - `Compute to Memory Ratio`：计算内存比

### 27. 编写CUDA时如何平衡register使用和occupancy

### Register（寄存器）

- **定义**：GPU 核心（SM）上的高速片上存储单元，是 GPU 线程最快速的访问存储器（访问延迟远低于共享内存、全局内存）。

- **作用**：用于存储线程执行过程中需要频繁访问的临时数据、局部变量、函数参数等，避免频繁访问低速内存，提升执行效率。

- **关键特性**：每个 SM 的寄存器总量有限（如 Turing 架构 SM 约 65536 个 32 位寄存器），线程使用的寄存器越多，单个 SM 能同时驻留的线程 / 线程束数量越少。

1. Occupancy（占用率）

- **定义**：GPU 单个 SM（流式多处理器）上**实际驻留的活动线程束数**与该 SM **最大支持的活动线程束数**的比值（范围 0~1）。

- **作用**：高占用率可帮助隐藏内存访问延迟（当部分线程等待内存数据时，SM 可调度其他就绪线程执行），充分利用 SM 计算资源；低占用率可能导致 SM 空闲，浪费计算能力。

- **核心关联**：寄存器使用量是影响 Occupancy 的关键因素 —— 线程使用的寄存器越多，SM 能分配给线程的资源越少，驻留的线程束数越少，Occupancy 越低（二者呈负相关，需平衡）。

#### 核心策略

1. **Register限制**：

    ```C++
    
    // 手动限制寄存器使用
    __global__ void __launch_bounds__(256, 4) // block大小256，最小占用率4/8
    kernel(...) {
        // 使用volatile减少寄存器优化
        volatile float tmp = 0;
    }
    ```

2. **编译选项控制**：

    ```Bash
    
    # 限制最大寄存器数
    nvcc -maxrregcount=64 your_kernel.cu
    ```

3. **平衡策略**：

    - **高寄存器使用**：计算密集型kernel，减少内存访问

    - **低寄存器使用**：内存密集型kernel，提高occupancy隐藏延迟

    - **逐步调整**：从默认开始，逐步限制寄存器，监控occupancy和总性能

4. **实用技巧**：

    - 使用shared memory替代寄存器存储临时数据

    - 将大数组拆分为小块计算

    - 使用Nsight Compute监控寄存器使用和occupancy Trade-off

### 28. 什么是warp divergence？如何检测和避免？ Warp Divergence（线程束分化）

#### 定义

同一个warp（32个线程）中的线程因分支语句（if/else）执行不同路径，导致部分线程等待，降低执行效率。

#### 检测方法

1. **Nsight Compute**：查看`Control Flow`中的`Warp Divergence`指标

2. **CUDA Profiler**：`warp_execution_efficiency`指标

3. **代码审查**：检查kernel中的分支语句

#### 避免方法

1. **消除不必要的分支**：

    ```C++
    
    // 差：分支导致分化
    if (idx % 2 == 0) { a[idx] *= 2; }
    else { a[idx] += 1; }
    
    // 好：无分支实现
    a[idx] = (idx % 2 == 0) ? a[idx] * 2 : a[idx] + 1;
    ```

2. **数据重排**：将相同分支的线程数据放在一起

3. **分支归并**：合并相似的分支逻辑

4. **线程束对齐**：确保分支条件按warp粒度对齐

### 29. 动态Shape算子优化 & PagedAttention

#### 动态Shape优化策略

1. **提前编译多组配置**：为常见shape预编译优化kernel

2. **即时编译（JIT）**：根据实际shape动态生成优化代码

3. **内存池化**：预分配内存池，避免频繁malloc/free

4. **分块计算**：将动态shape拆分为固定大小的块

5. **模板化设计**：使用模板参数适配不同shape

#### PagedAttention核心原理（vLLM）

1. **问题**：传统Attention中，序列长度动态变化导致内存浪费和访存低效

2. **解决方案**：

    - 将KV缓存分页存储（类似操作系统虚拟内存）

    - 每个page固定大小，按需加载到GPU

    - 只计算需要的token的attention，避免冗余计算

3. **优势**：

    - 内存利用率提升：按需分配，避免预分配大数组

    - 访存效率高：连续的page访问，更好的coalescing

    - 动态适配：支持任意长度的序列

### 30. 算子融合收益评估 & 反效果场景

#### 收益评估方法

1. **量化指标**：

    - 内存访问次数：融合后减少的global memory读写

    - 计算时间：融合前后的kernel执行时间

    - 显存占用：减少中间结果存储

    - FLOPS/Byte：计算访存比，越高越好

2. **评估步骤**：

    - 基准测试：单独执行各个算子的总时间

    - 融合测试：执行融合算子的时间

    - 计算加速比 = 基准时间 / 融合时间

    - 验证精度：确保融合后结果无精度损失

#### 融合反而变慢的场景

1. **计算访存比降低**：融合后内存访问成为瓶颈

2. **寄存器压力过大**：导致occupancy下降，无法隐藏延迟

3. **控制流复杂化**：引入大量warp divergence

4. **shared memory不足**：无法有效缓存数据

5. **kernel过长**：无法充分利用stream并行

6. **动态shape场景**：融合后无法适配不同shape的优化

### 总结

1. **CUDA算子开发**：核心是优化内存访问（coalescing）、控制占用率、避免warp分化，PyBind11/TorchScript可实现Python绑定

2. **工具选择**：简单算子用Triton（开发效率高），复杂/极致性能用CUDA，矩阵运算优先CUTLASS

3. **算子融合**：收益在于减少内存访问，但需评估计算访存比，避免因寄存器/占用率问题导致性能下降

4. **动态Shape优化**：PagedAttention通过分页机制解决Attention的动态shape问题，是典型的空间换时间策略

5. **性能分析**：Nsight Compute是核心工具，重点关注内存效率、占用率、warp执行效率等指标
> （注：文档部分内容可能由 AI 生成）