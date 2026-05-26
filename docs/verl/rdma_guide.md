# RDMA 协议指南

## 目录

1. [RDMA 概述](#1-rdma-概述)
2. [RDMA 与传统 TCP/IP 的区别](#2-rdma-与传统-tcpip-的区别)
3. [RDMA 硬件实现方案](#3-rdma-硬件实现方案)
4. [RDMA 工作原理](#4-rdma-工作原理)
5. [RDMA 数据传输方式](#5-rdma-数据传输方式)
6. [RDMA 网络配置与调优](#6-rdma-网络配置与调优)
7. [PyTorch 中使用 RDMA](#7-pytorch-中使用-rdma)
8. [Ray 中使用 RDMA](#8-ray-中使用-rdma)
9. [性能对比与最佳实践](#9-性能对比与最佳实践)

---

## 1. RDMA 概述

### 1.1 定义

**RDMA（Remote Direct Memory Access，远程直接内存访问）** 是一种为了解决网络传输中服务器端数据处理延迟而产生的技术。通过
RDMA，本端节点可以"直接"访问远端节点的内存，通过网络把数据直接传入计算机的存储区，将数据从一个系统快速移动到远程系统存储器中，
**而不对操作系统造成任何影响**。

### 1.2 核心特点

| 特性                      | 说明                               |
|-------------------------|----------------------------------|
| **零拷贝（Zero Copy）**      | 数据直接从网卡到应用内存，无需在内核空间和用户空间之间多次拷贝  |
| **内核旁路（Kernel Bypass）** | 绕过操作系统内核协议栈，直接由网卡硬件处理通信          |
| **协议栈卸载（Offload）**      | 将 TCP/IP 协议栈的实现下沉至 RDMA 网卡（RNIC） |
| **低延迟**                 | 微秒级（μs）延迟，相比 TCP 的毫秒级（ms）提升显著    |
| **高吞吐**                 | 解放 CPU 资源，释放内存带宽用于应用计算           |

### 1.3 为什么需要 RDMA？

在大规模分布式训练场景中：

```
┌─────────────────────────────────────────────────────────────┐
│                    传统 TCP/IP 通信瓶颈                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   应用程序 → 内核拷贝 → 协议栈处理 → 网卡驱动 → 物理网络      │
│       ↑         ↑           ↑          ↑                   │
│    用户态    内核态切换    CPU消耗     中断处理               │
│                                                             │
│   问题: 每次通信涉及多次数据拷贝 + 上下文切换 + 内核中断        │
│   结果: CPU 占用高、延迟大、吞吐受限                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     RDMA 通信优势                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   应用程序 ←─────────────────────────→ RNIC ←→ 物理网络      │
│       ↑                                   ↑                │
│    直接内存访问                        硬件协议栈处理          │
│                                                             │
│   优势: 零拷贝、内核旁路、CPU 几乎不参与                      │
│   结果: 低延迟、高吞吐、CPU 释放给计算任务                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. RDMA 与传统 TCP/IP 的区别

### 2.1 架构对比

```
┌──────────────────────────────────────────────────────────────────┐
│                    传统 TCP/IP 协议栈                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐       │
│  │ 应用层   │ →  │ Socket  │ →  │ TCP/IP  │ →  │  网卡驱动 │ → 网络 │
│  │ (用户态) │    │ (内核)  │    │ (内核)  │    │  (内核)  │       │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘       │
│       ↓              ↓             ↓             ↓              │
│   用户缓冲区    内核缓冲区     协议栈处理     驱动队列            │
│                                                                  │
│  ❌ 多次内存拷贝 (3-4 次)                                         │
│  ❌ 多次上下文切换                                                 │
│  ❌ CPU 参与协议处理                                               │
│  ❌ 内核中断开销                                                   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                      RDMA 协议栈                                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────┐                              ┌─────────┐            │
│  │ 应用层   │ ───────────────────────────→ │  RNIC   │ → 网络     │
│  │ (用户态) │     直接内存访问 (Verbs API)  │ (硬件)  │            │
│  └─────────┘                              └─────────┘            │
│       ↓                                        ↓                 │
│   用户缓冲区                               网卡硬件处理            │
│                                                                  │
│  ✅ 零拷贝                                                         │
│  ✅ 内核旁路                                                       │
│  ✅ CPU 不参与数据传输                                              │
│  ✅ 硬件卸载协议处理                                                │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 性能对比

| 指标          | TCP/IP   | RDMA     | 提升倍数      |
|-------------|----------|----------|-----------|
| **延迟**      | 10-50 μs | 1-5 μs   | **10x+**  |
| **带宽利用率**   | 60-80%   | 90-95%+  | **~1.2x** |
| **CPU 占用率** | 高（内核参与）  | 极低（硬件处理） | **大幅降低**  |
| **吞吐量**     | 受限于 CPU  | 接近线速     | **显著提升**  |

---

## 3. RDMA 硬件实现方案

RDMA 有三种主要的硬件实现方式：

### 3.1 方案对比

| 特性           | InfiniBand (IB) | iWARP    | RoCE v2           |
|--------------|-----------------|----------|-------------------|
| **底层协议**     | 专用 IB 协议        | 基于 TCP   | 基于 UDP/IP         |
| **性能**       | ⭐⭐⭐ 最高          | ⭐⭐ 一般    | ⭐⭐⭐ 较高            |
| **成本**       | 💰💰💰 高        | 💰 较低    | 💰💰 较低           |
| **是否需要专用设备** | ✅ 需要            | ❌ 不需要    | ❌ 不需要（只需 RoCE 网卡） |
| **部署复杂度**    | 高               | 低        | 中等                |
| **路由能力**     | 有限              | 成熟 IP 路由 | 支持 L3 路由          |

### 3.2 InfiniBand (IB)

- 从一开始就支持 RDMA 的新一代网络协议
- 广泛应用于高性能计算（HPC）领域
- 需要专用的 IB 网卡和交换机
- 价格昂贵，但性能最优

### 3.3 RoCE (RDMA over Converged Ethernet)

- 允许通过以太网执行 RDMA 的网络协议
- 可在标准以太网交换机上使用（需支持 PFC/ECN）
- **RoCE v1**：基于链路层，不可路由
- **RoCE v2**（推荐）：基于 UDP/IP，支持 L3 路由和拥塞控制
- **目前业界最主流的方案**，性价比最高

### 3.4 iWARP

- 允许通过 TCP 执行 RDMA
- 利用成熟的 IP 网络基础设施
- 性能相对较低，但兼容性最好

---

## 4. RDMA 工作原理

### 4.1 核心组件

RDMA 提供了基于消息队列的点对点通信机制，核心组件包括：

```
┌─────────────────────────────────────────────────────────────────┐
│                      RDMA 核心架构                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   本地节点 (Local)                        远程节点 (Remote)      │
│   ┌──────────────┐                      ┌──────────────┐        │
│   │   Application│                      │   Application│        │
│   └──────┬───────┘                      └──────┬───────┘        │
│          │                                     │                │
│          ▼                                     ▼                │
│   ┌──────────────┐    Network     ┌──────────────────────┐      │
│   │ Queue Pair   │ ────────────→  │     Queue Pair       │      │
│   │ ┌──────────┐ │               │  ┌────────────────┐  │      │
│   │ │Send Queue │ │               │  │  Receive Queue  │  │      │
│   │ │   (SQ)    │ │               │  │     (RQ)        │  │      │
│   │ ├──────────┤ │               │  ├────────────────┤  │      │
│   │ │Recv Queue │ │               │  │  Send Queue     │  │      │
│   │ │   (RQ)    │ │               │  │     (SQ)        │  │      │
│   │ └──────────┘ │               │  └────────────────┘  │      │
│   └──────────────┘               └──────────────────────┘      │
│          │                                     │                │
│          ▼                                     ▼                │
│   ┌──────────────┐                      ┌──────────────┐        │
│   │Complete Queue│                      │Complete Queue│        │
│   │     (CQ)     │                      │     (CQ)     │        │
│   └──────────────┘                      └──────────────┘        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 关键概念说明

| 组件                         | 说明                                                     |
|----------------------------|--------------------------------------------------------|
| **Queue Pair (QP)**        | 一对 Send Queue (SQ) 和 Receive Queue (RQ)，是 RDMA 通信的基本单元 |
| **Completion Queue (CQ)**  | 完成队列，通知用户消息处理结果（成功/失败）                                 |
| **Work Request (WR)**      | 工作请求，用户提交的操作描述                                         |
| **Work Queue Entry (WQE)** | 工作队列中的元素，对应一个 WR                                       |
| **Memory Region (MR)**     | 注册的内存区域，用于 RDMA 操作                                     |
| **rkey/lkey**              | 远程/本地内存操作的密钥，保护内存安全                                    |

### 4.2 通信流程

```
步骤 1: 建立连接
─────────────────────────────────────────────────────────────────
  本地节点                          远程节点
     │                                 │
     │ ──── 创建 Channel/QP/CQ ────→   │
     │ ←── 交换 QP 信息 ────────────   │
     │                                 │

步骤 2: 内存注册 (Memory Registration)
─────────────────────────────────────────────────────────────────
  应用程序                           RNIC (网卡)
     │                                 │
     │ ──── 注册虚拟内存地址 ──────→    │
     │ ←── 返回 rkey/lkey 密钥 ────    │
     │                                 │
     │ 注: 注册后的内存被锁定，不会被换出

步骤 3: 提交工作请求
─────────────────────────────────────────────────────────────────
  应用程序                           RNIC (网卡)
     │                                 │
     │ ──── 提交 WR 到 SQ/RQ ─────→    │
     │                                 │
     │    WR 包含:
     │    - 本地内存地址 + lkey
     │    - 远程内存地址 + rkey (对于 Read/Write)
     │    - 数据长度
     │                                 │

步骤 4: 硬件异步处理
─────────────────────────────────────────────────────────────────
  RNIC                             远程 RNIC / 内存
     │                                 │
     │ ──── 硬件调度执行操作 ──────→    │
     │    (无需 CPU 参与)               │
     │                                 │

步骤 5: 完成通知
─────────────────────────────────────────────────────────────────
  RNIC                             应用程序
     │                                 │
     │ ──── 写入 CQE 到 CQ ────────→   │
     │    (完成队列元素)                │
     │                                 │
     │    应用可通过轮询或中断获取结果    │
```

---

## 5. RDMA 数据传输方式

RDMA 提供三种基本的数据传输操作：

### 5.1 Send / Recv（双边操作）

```
┌─────────────────────────────────────────────────────────────┐
│                  RDMA Send/Recv 流程                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Sender (发送方)              Receiver (接收方)             │
│   ┌─────────┐                  ┌─────────┐                  │
│   │  App    │                  │  App    │                  │
│   └────┬────┘                  └────┬────┘                  │
│        │  Post Send                │  Post Recv (先于Send)    │
│        ▼                            ▼                         │
│   ┌────┴────┐                  ┌────┴────┐                  │
│   │   SQ    │ ──── 数据 ────→  │   RQ    │                  │
│   └─────────┘                  └─────────┘                  │
│                                                             │
│   特点:                                                      │
│   - 类似 TCP 的 send/recv 接口                               │
│   - 双边操作：发送方和接收方都需要参与                         │
│   - 接收方必须先 Post Recv，否则 Send 会失败                  │
│   - 数据包组装在硬件网卡上完成                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Read（单边操作 - Pull）

```
┌─────────────────────────────────────────────────────────────┐
│                   RDMA Read 流程                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Initiator (发起方)           Target (目标方)               │
│   ┌─────────┐                  ┌─────────┐                  │
│   │  App    │                  │  App    │                  │
│   └────┬────┘                  └────┬────┘                  │
│        │                            │                        │
│        │  1. 目标方提供:             │                        │
│        │     - 远程内存地址          │                        │
│        │     - rkey 密钥            │                        │
│        │ ←───────────────────────── │                        │
│        │                            │                        │
│        │  2. 发起 Read 请求          │  (目标方无感知!)        │
│        │    (本地addr, remote addr, rkey)                    │
│        ▼                            │                        │
│   ┌────┴────┐    ┌──────────┐   ┌───┴────┐                  │
│   │   SQ    │ →  │  Remote  │ → │ Memory │                  │
│   └─────────┘    │  Memory  │   └────────┘                  │
│                   └──────────┘      ↑                        │
│                   (Pull 拉取数据)    │                        │
│                                  数据到达本地内存             │
│                                                             │
│   特点:                                                      │
│   - 单边操作：只有发起方参与，目标方完全不感知                  │
│   - Pull 模式：主动从远程拉取数据                             │
│   - 目标方 CPU 不参与                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.3 Write（单边操作 - Push）

```
┌─────────────────────────────────────────────────────────────┐
│                   RDMA Write 流程                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Initiator (发起方)           Target (目标方)               │
│   ┌─────────┐                  ┌─────────┐                  │
│   │  App    │                  │  App    │                  │
│   └────┬────┘                  └────┬────┘                  │
│        │                            │                        │
│        │  1. 目标方提供:             │                        │
│        │     - 远程内存地址          │                        │
│        │     - rkey 密钥            │                        │
│        │ ←───────────────────────── │                        │
│        │                            │                        │
│        │  2. 发起 Write 请求         │  (目标方无感知!)        │
│        │    (本地addr, remote addr, rkey)                    │
│        ▼                            │                        │
│   ┌────┴────┐                  ┌────┴────┐                  │
│   │  Local  │ ──── 数据 ────→  │ Remote  │                  │
│   │ Memory  │    (Push 推送)   │ Memory  │                  │
│   └─────────┘                  └─────────┘                  │
│                                                             │
│   特点:                                                      │
│   - 单边操作：只有发起方参与，目标方完全不感知                  │
│   - Push 模式：主动推送数据到远程                              │
│   - 目标方 CPU 不参与                                        │
│   - 支持 With Immediate: 写入同时通知对端(带1字节立即数)       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.4 三种操作对比

| 操作类型          | 参与方       | 方向    | 使用场景        |
|---------------|-----------|-------|-------------|
| **Send/Recv** | 双边        | 显式控制  | 消息传递、控制信令   |
| **Read**      | 单边 (Pull) | 远程→本地 | 远程数据读取      |
| **Write**     | 单边 (Push) | 本地→远程 | 远程数据写入、参数同步 |

---

## 6. RDMA 网络配置与调优

### 6.1 PFC（Priority-based Flow Control）

PFC 是一种**点到点**的流控技术，可精确控制某条物理链路上某个优先级的所有流量。

```bash
# 查看 PFC 配置（以 eth0 为例）
mlnx_qos -i eth0
```

**原理**：

- 根据 IP 头部 DSCP 字段将报文映射到 8 个优先级
- 当交换机入口 buffer 不足时，向上游设备发送 **PAUSE 帧**
- 基于水线机制：**X-OFF**（暂停）/ **X-ON**（恢复）

```
交换机 Ingress Buffer
┌────────────────────────────────┐
│███████████████████████████████│ ← X-OFF 水线: 发送 PAUSE
│░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
│░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│ ← X-ON 水线:  恢复传输
│                                │
└────────────────────────────────┘
```

### 6.2 ECN（Explicit Congestion Notification）

ECN 是一种**端到端**的流控技术，可针对具体数据流进行拥塞控制。

```bash
# 查看 ECN 配置
cat /sys/class/net/eth0/ecn/roce_np/enable    # ECN 是否开启
cat /sys/class/net/eth0/ecn/roce_rp/enable    # 发送方向 ECN
cat /sys/class/net/eth0/ecn/roce_rn/enable    # 接收方向 ECN

# 查看流量 class 配置
cat /sys/class/infiniband/mlx5_0/tc/1/traffic_class
```

**DCQCN（Data Center Quantized Congestion Notification）流程**：

```
发送端 (RP)                    交换机 (CP)              接收端 (NP)
    │                              │                        │
    │ ─────── 数据报文 ──────────→  │                        │
    │                              │                        │
    │                              ├── 标记 ECN 位 ──────→   │
    │                              │   (检测到拥塞)          │
    │                              │                        │
    │ ←──── CNP 报文 (降速通知) ───┼─────────────────────────│
    │                              │                        │
    │ ──── 按 DCQCN 算法降速 ────→  │                        │
```

### 6.3 Lossy 配置

```bash
# 查看 lossy 配置（以 mlx5_bond_0 为例）
sudo mlxreg -d mlx5_bond_0 --reg_name ROCE_ACCL --get
```

> ⚠️ 如果输出报错，说明 OFED 和固件版本过低。

### 6.4 常用监控命令

```bash
# RDMA 流量监控（一次只能查看一个网卡）
mlnx_perf -i eth0 | grep tx_prio3_bytes   # RDMA 发送流量
mlnx_perf -i eth0 | grep rx_prio3_bytes   # RDMA 接收流量

# 查看 RDMA 设备信息
ibv_devices                                # 列出 RDMA 设备
ibv_devinfo -v                             # 详细设备信息

# 查看 RDMA 链路状态
ibstat                                     # IB 链路状态

# 性能测试
ib_write_bw  # RDMA Write 带宽测试
ib_read_bw   # RDMA Read 带宽测试
ib_send_bw   # RDMA Send 带宽测试
```

### 6.5 联通性测试

```bash
# 基础连通性测试
ibping  # RDMA 层面的 ping 测试

# 带宽和延迟测试
ib_write_bw -d mlx5_0 -i 1 -s 1g --run_infinitely
ib_send_lat -d mlx5_0 -o write --run_infinitely
```

---

## 7. PyTorch 中使用 RDMA

### 7.1 原理概述

PyTorch 分布式训练本身**不需要直接操作 RDMA Verbs API**。PyTorch 通过以下调用链间接使用 RDMA：

```
PyTorch Distributed
    ↓
torch.distributed.init_process_group(backend="nccl")
    ↓
NCCL (NVIDIA Collective Communications Library)
    ↓
自动检测并使用可用的网络接口:
  - 若有 IB/RoCE 设备 → 自动使用 RDMA (via IB verbs)
  - 否则 → 回退到 TCP/IP Socket (via Socket)
```

**关键点**：NCCL 会**自动检测**并优先使用 RDMA 网络。只要环境中有 RDMA 设备且 NCCL 编译时包含了 IB verbs 支持，就会自动启用。

### 7.2 环境准备

#### 7.2.1 硬件要求

- 支持 RDMA 的网卡（Mellanox ConnectX 系列、Broadcom 等）
- RDMA 交换机（或支持 PFC/ECN 的以太网交换机 for RoCE）
- 安装了正确的驱动（MLNX_OFED / RDMA Core 等）

#### 7.2.2 软件安装

```bash
# Ubuntu/Debian
sudo apt-get install libibverbs1 libibverbs-dev librdmacm1 librdmacm-dev

# CentOS/RHEL
sudo yum install rdma-core-devel libibverbs-utils librdmacm-utils

# 验证安装
ibv_devinfo
rdma link
```

#### 7.2.3 NCCL 配置

NCCL 提供了一组环境变量来控制 RDMA 行为：

```bash
# ========== 核心 RDMA 配置 ==========

# 强制 NCCL 使用 RDMA (IB) 接口
export NCCL_SOCKET_IFNAME=eth0          # 指定网络接口
export NCCL_IB_DISABLE=0                # 启用 IB/RDMA (默认为 0，即启用)
export NCCL_IB_GID_INDEX=3              # RoCE v2 GID index (通常为 3)

# 禁用 RDMA（回退到 TCP，用于调试对比）
# export NCCL_IB_DISABLE=1

# ========== 网络选择策略 ==========
# NCCL_IFACE 优先级高于 NCCL_SOCKET_IFNAME
export NCCL_IFNAME=eth0                 # 指定使用的网络接口名

# ========== 性能调优参数 ==========
export NCCL_IB_TIMEOUT=23               # IB 超时 (单位: 4.096μs 的 2^n 倍)
                                       # 23 ≈ ~34秒，适用于大规模训练
export NCCL_IB_RETRY_CNT=7              # IB 重试次数 (默认 7)

# 缓冲区大小配置
export NCCL_BUFFSIZE=8388608            # 8MB ring buffer (默认值)

# P2P 通信配置
export NCCL_P2P_LEVEL=NVL               # P2P 级别: NVL/BUS/PIX/SYS

# ========== 调试诊断 ==========
export NCCL_DEBUG=INFO                  # 日志级别: VERSION/INFO/WARN/ERROR
export NCCL_DEBUG_SUBSYS=ALL            # 调试子系统
```

### 7.3 PyTorch 分布式训练代码示例

#### 7.3.1 基础 DDP 训练（使用 RDMA）

```python
"""
PyTorch DDP + RDMA 示例
当 NCCL 检测到 RDMA 设备时，AllReduce 等集合通信会自动走 RDMA 路径

运行方式:
    torchrun --nproc_per_node=4 --master-port=29500 ddp_rdma_example.py
"""

import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_distributed():
    """初始化分布式环境 - NCCL backend 自动使用 RDMA"""
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # 设置 CUDA 设备
    torch.cuda.set_device(local_rank)

    # 初始化进程组 - backend="nccl" 会自动检测并使用 RDMA
    # NCCL 内部调用顺序:
    #   1. ibv_get_device_list() 检测 IB 设备
    #   2. 若有可用 IB 设备 → 创建 IB QP 进行 RDMA 通信
    #   3. 否则 → 回退到 Socket 通信
    dist.init_process_group(
        backend="nccl",  # NCCL backend
        init_method="env://",  # 从环境变量初始化
        world_size=world_size,
        rank=rank,
    )

    print(f"[Rank {rank}] Initialized on GPU {local_rank}, "
          f"using NCCL (RDMA auto-detected)")

    return rank, local_rank, world_size


class SimpleModel(nn.Module):
    """示例模型"""

    def __init__(self, hidden_dim=512, num_layers=6):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x  # 残差连接
        return x


def train_one_epoch(model, optimizer, rank, epoch):
    """训练一个 epoch"""
    model.train()

    total_loss = 0
    num_steps = 100

    for step in range(num_steps):
        # 模拟输入数据（各 rank 使用相同数据模拟 DDP）
        torch.manual_seed(step + epoch * 1000)
        x = torch.randn(32, 128, 512, device=f"cuda:{rank}")

        # 前向传播
        output = model(x)
        loss = output.mean()

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度同步 - 这里 DDP 会自动调用 AllReduce
        # AllReduce 通过 NCCL → IB Verbs → RDMA 网卡 执行
        optimizer.step()

        total_loss += loss.item()

        if step % 20 == 0 and rank == 0:
            print(f"  Epoch {epoch}, Step {step}: loss={loss.item():.4f}")

    return total_loss / num_steps


def main():
    rank, local_rank, world_size = setup_distributed()

    # 创建模型并包装为 DDP
    torch.manual_seed(42)
    model = SimpleModel(hidden_dim=512, num_layers=6).cuda()
    ddp_model = DDP(model, device_ids=[local_rank])

    # 优化器
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-3)

    if rank == 0:
        print(f"\n{'=' * 60}")
        print(f"Training with DDP + NCCL (RDMA)")
        print(f"World Size: {world_size}")
        print(f"{'=' * 60}\n")

    # 训练
    for epoch in range(3):
        avg_loss = train_one_epoch(ddp_model, optimizer, rank, epoch)

        if rank == 0:
            print(f"Epoch {epoch} completed, avg_loss={avg_loss:.4f}\n")

    # 清理
    dist.destroy_process_group()
    if rank == 0:
        print("Training completed!")


if __name__ == "__main__":
    main()
```

#### 7.3.2 FSDP + RDMA（大模型训练）

```python
"""
PyTorch FSDP + RDMA 示例
FSDP 的 AllGather/ReduceScatter 同样通过 NCCL 自动使用 RDMA

运行方式:
    torchrun --nproc_per_node=4 fsdp_rdma_example.py
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecisionPolicy,
    ShardingStrategy,
)


def setup_distributed():
    """初始化分布式环境"""
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")  # NCCL 自动使用 RDMA

    return rank, local_rank, world_size


class TransformerBlock(nn.Module):
    """简化的 Transformer Block"""

    def __init__(self, dim=1024):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads=16, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # Self-Attention (含 AllReduce via TP 或 DDP)
        attn_out, _ = self.attention(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        # FFN
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerModel(nn.Module):
    """Transformer 模型"""

    def __init__(self, dim=1024, num_layers=12, vocab_size=32000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)


def main():
    rank, local_rank, world_size = setup_distributed()

    if rank == 0:
        print("\n" + "=" * 60)
        print("FSDP Training with RDMA")
        print("=" * 60)

    # 创建模型
    model = TransformerModel(dim=1024, num_layers=12, vocab_size=32000)

    # FSDP 配置 - sharding 策略决定通信模式
    # FULL_SHARD: 参数+梯度+优化器状态都分片 (类似 ZeRO-3)
    #   Forward:  AllGather (参数收集) → NCCL → RDMA
    #   Backward: ReduceScatter (梯度分散) → NCCL → RDMA
    # SHARD_GRAD_OP: 只分片梯度和优化器状态 (类似 ZeRO-2)
    # 包装模型 - 使用关键字参数配置 FSDP
    fsdp_model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision_policy=MixedPrecisionPolicy(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
        ),
        device_id=local_rank,
    )

    if rank == 0:
        print(f"FSDP initialized with FULL_SHARD strategy")
        print(f"AllGather/ReduceScatter will use NCCL → RDMA\n")

    # ... 训练循环 ...

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

### 7.4 验证 RDMA 是否生效

```python
"""验证 NCCL 是否正在使用 RDMA"""

import torch
import torch.distributed as dist
import os


def verify_rdma_usage():
    """检查 NCCL 是否使用了 RDMA"""
    rank = int(os.environ.get("RANK", 0))

    dist.init_process_group(backend="nccl")

    if rank == 0:
        print("\n" + "=" * 60)
        print("RDMA Usage Verification")
        print("=" * 60)

        # 方法 1: 检查 NCCL 日志输出
        print("\n[方法 1] 设置 NCCL_DEBUG=INFO 查看日志:")
        print("  export NCCL_DEBUG=INFO")
        print("  日志中出现 'NCCL_INFO IB:' 表示使用了 RDMA")
        print("  日志中出现 'NCCL_INFO Socket:' 表示回退到了 TCP")

        # 方法 2: 检查 IB 设备
        print("\n[方法 2] 检查系统中是否有 IB 设备:")
        try:
            import subprocess
            result = subprocess.run(["ibv_devinfo"], capture_output=True, text=True)
            if result.returncode == 0 and "mlx5" in result.stdout.lower():
                print("  ✅ 检测到 Mellanox IB/RDMA 设备")
            else:
                print("  ⚠️ 未检测到标准 IB 设备")
        except FileNotFoundError:
            print("  ⚠️ ibv_devinfo 未安装，无法检测")

        # 方法 3: 性能对比测试
        print("\n[方法 3] AllReduce 带宽测试:")
        data = torch.randn(100_000_000, device="cuda")  # ~400MB

        import time
        # Warmup
        for _ in range(3):
            dist.all_reduce(data)

        # Benchmark
        times = []
        for _ in range(10):
            torch.cuda.synchronize()
            start = time.perf_counter()
            dist.all_reduce(data)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        avg_time = sum(times) / len(times)
        bandwidth_gb = (data.numel() * 4 / 1e9) / avg_time  # float32

        print(f"  平均 AllReduce 时间: {avg_time * 1000:.2f} ms")
        print(f"  估算带宽: {bandwidth_gb:.2f} GB/s")
        print(f"  {'✅ 可能使用 RDMA' if bandwidth_gb > 10 else '⚠️ 可能使用 TCP'}")

    dist.destroy_process_group()


if __name__ == "__main__":
    verify_rdma_usage()
```

### 7.5 启动脚本示例

```bash
#!/bin/bash
# launch_rdma_training.sh
# 使用 RDMA 启动 PyTorch 分布式训练

set -e

# ==================== RDMA 环境配置 ====================
export NCCL_IB_DISABLE=0              # 启用 IB/RDMA
export NCCL_IB_GID_INDEX=3            # RoCE v2
export NCCL_IB_TIMEOUT=23             # 超时时间
export NCCL_IB_RETRY_CNT=7            # 重试次数
export NCCL_SOCKET_IFNAME=eth0        # 网络接口
export NCCL_DEBUG=INFO                # 开启日志（调试时可开启）

# GPU 相关
export CUDA_DEVICE_MAX_CONNECTIONS=1  # 通信与计算重叠优化

# ==================== 启动训练 ====================
echo "Starting training with RDMA..."
echo "NCCL_IB_DISABLE=$NCCL_IB_DISABLE"
echo "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"

torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    ddp_rdma_example.py

echo "Training completed!"
```

---

## 8. Ray 中使用 RDMA

### 8.1 Ray 与 RDMA 的关系

Ray 是一个分布式计算框架，其底层的节点间通信可以通过 RDMA 加速：

```
┌─────────────────────────────────────────────────────────────┐
│                    Ray 架构中的 RDMA                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Head Node                    Worker Nodes                  │
│  ┌──────────┐                                           │
│  │  Ray     │ ◄─────────────────────────────────────┐    │
│  │  Head    │     Object Store / Control Messages    │    │
│  └──────────┘                                      │    │
│      │                                            │    │
│      │  gRPC / Plasma Store                       │    │
│      │  (可配置使用 RDMA)                          │    │
│      │                                            │    │
│  ┌────────────────────────────────────────────────┴──┐ │
│  │              底层训练进程 (torchrun)               │ │
│  │  ┌────────────────────────────────────────────┐  │ │
│  │  │  PyTorch DDP/FSDP                          │  │ │
│  │  │    ↓                                       │  │ │
│  │  │  NCCL Backend                              │  │ │
│  │  │    ↓                                       │  │ │
│  │  │  IB Verbs ──────────────────→ RDMA Network │  │ │
│  │  └────────────────────────────────────────────┘  │ │
│  └───────────────────────────────────────────────────┘ │
│                                                             │
│  关键: Ray 任务内部的 PyTorch 训练进程                       │
│        通过 NCCL 自动使用 RDMA                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 Ray 集群配置 RDMA

#### 8.2.1 Ray 集群提交命令

```bash
# 在具有 RDMA 网络的集群上提交 Ray 训练任务
ray job submit \
    --address='http://<ray-head>:8420' \
    --working-dir=. \
    --entrypoint-num-gpus 8 \
    --environment='{"NCCL_IB_DISABLE": "0", "NCCL_IB_GID_INDEX": "3"}' \
    -- torchrun --nproc_per_node=8 --master_port=29500 your_training_script.py
```

#### 8.2.2 Ray 自定义资源配置（支持 RDMA）

```python
"""
Ray 集群配置 - 启用 RDMA 资源调度
确保训练任务被调度到具有 RDMA 能力的节点上
"""

import ray
from ray import tune
from ray.train.torch import TorchTrainer

# 定义包含 RDMA 资源的自定义资源
# 假设集群已注册 "rdma" 自定义资源
RDMA_RESOURCES = {
    "GPU": 8,  # 8 张 GPU
    "rdma": 1,  # 要求 RDMA 能力
}


def train_func(config):
    """Ray 中的训练函数"""
    import os
    import torch
    import torch.distributed as dist

    # Ray TorchTrainer 会自动设置分布式环境变量
    # 但我们需要确保 NCCL 使用 RDMA
    os.environ["NCCL_IB_DISABLE"] = "0"
    os.environ["NCCL_IB_GID_INDEX"] = "3"

    # 初始化分布式
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"[Rank {rank}] Training with RDMA, world_size={world_size}")

    # ... 你的训练代码 ...

    dist.destroy_process_group()


# 使用 TorchTrainer 启动训练
trainer = TorchTrainer(
    train_func=train_func,
    train_loop_config={
        "lr": 1e-3,
        "batch_size": 32,
        "epochs": 10,
    },
    scaling_config={
        # 使用自定义资源（包含 RDMA）
        "use_gpu": True,
        "num_workers": 1,  # 单节点多卡
        "resources_per_worker": {
            "GPU": 8,
            "rdma": 1,  # 要求 RDMA 资源
        },
    },
    # 设置环境变量
    run_config={
        "env": {
            "NCCL_IB_DISABLE": "0",
            "NCCL_IB_GID_INDEX": "3",
            "NCCL_DEBUG": "INFO",
        },
    },
)

result = trainer.fit()
print(f"Training completed: {result}")
```

#### 8.2.3 Ray + PyTorch 完整示例

```python
"""
Ray + PyTorch DDP + RDMA 完整示例
展示如何在 Ray 集群上启动使用 RDMA 的分布式训练

运行方式:
    1. 确保 Ray 集群的网络支持 RDMA
    2. python ray_rdma_training.py
"""

import ray
from ray import train
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
import torch
import torch.nn as nn
import torch.distributed as dist


class SimpleNet(nn.Module):
    """简单神经网络"""

    def __init__(self, dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return self.net(x)


def train_epoch(model, optimizer, rank, epoch, config):
    """训练一个 epoch"""
    model.train()
    batch_size = config.get("batch_size", 64)
    dim = config.get("dim", 256)
    steps_per_epoch = config.get("steps_per_epoch", 100)

    total_loss = 0
    for step in range(steps_per_epoch):
        # 模拟数据
        torch.manual_seed(epoch * 1000 + step)
        x = torch.randn(batch_size, dim, device=f"cuda:{rank % torch.cuda.device_count()}")
        y = torch.randn(batch_size, dim, device=f"cuda:{rank % torch.cuda.device_count()}")

        # 前向 + 反向
        output = model(x)
        loss = nn.functional.mse_loss(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 上报指标到 Ray
        if step % 20 == 0:
            train.report({"loss": loss.item(), "step": epoch * steps_per_epoch + step})

    return total_loss / steps_per_epoch


def train_func(config):
    """
    Ray TorchTrainer 的训练函数
    Ray 会自动初始化分布式环境并设置 RANK/WORLD_SIZE 等变量
    """
    # ====== RDMA 配置 ======
    import os
    os.environ.setdefault("NCCL_IB_DISABLE", "0")
    os.environ.setdefault("NCCL_IB_GID_INDEX", "3")
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "eth0")

    # 初始化 NCCL (backend="nccl" 自动使用 RDMA)
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    local_rank = rank % torch.cuda.device_count()
    world_size = dist.get_world_size()

    torch.cuda.set_device(local_rank)

    if rank == 0:
        print(f"\n{'=' * 60}")
        print(f"Ray + PyTorch DDP + RDMA Training")
        print(f"World Size: {world_size}")
        print(f"NCCL_IB_DISABLE: {os.environ['NCCL_IB_DISABLE']}")
        print(f"{'=' * 60}\n")

    # 创建模型
    torch.manual_seed(42)
    model = SimpleNet(dim=config.get("dim", 256)).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr", 1e-3))

    # 训练循环
    for epoch in range(config.get("epochs", 5)):
        avg_loss = train_epoch(model, optimizer, rank, epoch, config)

        if rank == 0:
            print(f"Epoch {epoch}: avg_loss = {avg_loss:.4f}")

    dist.destroy_process_group()

    if rank == 0:
        print("Training completed!")


def main():
    # 连接或初始化 Ray
    if not ray.is_initialized():
        ray.init(address="auto")  # 连接现有集群，或 ray.init() 启动新的

    # 配置训练
    scaler = ScalingConfig(
        num_workers=1,  # 单节点
        use_gpu=True,
        resources_per_worker={
            "GPU": 8,  # 8 卡 GPU
            # "rdma": 1,            # 如果集群注册了 RDMA 资源
        },
    )

    # 创建 Trainer
    trainer = TorchTrainer(
        train_func=train_func,
        train_loop_config={
            "lr": 1e-3,
            "batch_size": 64,
            "dim": 256,
            "epochs": 5,
            "steps_per_epoch": 100,
        },
        scaling_config=scaler,
        # 通过 run_config 传递环境变量
        run_config=train.RunConfig(
            name="rdma-training-experiment",
            storage_path="./ray_results",
        ),
    )

    # 运行训练
    result = trainer.fit()
    print(f"\nFinal result: {result.metrics}")


if __name__ == "__main__":
    main()
```

### 8.3 Ray Job CLI 提交 RDMA 训练

```bash
#!/bin/bash
# submit_rdma_job.sh
# 通过 Ray Job CLI 提交 RDMA 训练任务

RAY_ADDRESS="http://your-ray-head:8420"

# 提交 ZeRO 训练任务
ray job submit \
    --address="$RAY_ADDRESS" \
    --working-dir="$(pwd)" \
    --entrypoint-num-gpus 8 \
    -- \
    torchrun \
        --nproc_per_node=8 \
        --master_port=29500 \
        zero_impl.py

# 提交 FSDP 训练任务
ray job submit \
    --address="$RAY_ADDRESS" \
    --working-dir="$(pwd)" \
    --entrypoint-num-gpus 8 \
    -- \
    torchrun \
        --nproc_per_node=8 \
        --master_port=29501 \
        fsdp_impl.py

# 提交 Megatron 并行训练任务
ray job submit \
    --address="$RAY_ADDRESS" \
    --working-dir="$(pwd)" \
    --entrypoint-num-gpus 8 \
    -- \
    torchrun \
        --nproc_per_node=8 \
        --master_port=29502 \
        megatron_impl.py
```

### 8.4 Ray 集群配置文件（YAML）

```yaml
# cluster_rdma.yaml
# Ray 集群配置 - 支持 RDMA 的节点池

cluster_name: llm-training-cluster

provider:
  type: aws  # 或 kubernetes / local
  region: us-west-2
  availability_zone: us-west-2a

auth:
  ssh_user: ubuntu

# 主节点配置
head_node:
  InstanceType: p4d.24xlarge  # 8x A100 + RDMA
  ImageId: ami-xxxxxxxx
  BlockDeviceMappings:
    - DeviceName: /dev/sda
      Ebs:
        VolumeSize: 1000
        VolumeType: gp3

# 工作节点配置 - 使用支持 RDMA 的实例类型
worker_nodes:
  InstanceType:
    - p4d.24xlarge   # AWS: 8x A100 + EFA (RDMA)
    - p5.48xlarge    # AWS: 8x H100 + EFA (RDMA)
  ImageId: ami-xxxxxxxx
  MinWorkers: 2
  MaxWorkers: 10
  InitialWorkers: 2

# 节点启动时的设置
setup_commands:
  # 安装 RDMA 驱动和工具
  - sudo apt-get update
  - sudo apt-get install -y rdma-core libibverbs1 libibverbs-dev
  # 验证 RDMA 设备
  - ibv_devinfo || echo "No IB devices found"

# 头节点启动命令
head_start_ray_commands:
  - ray stop
  - ulimit -n 65535
  - ray start --head --port=6379 --object-manager-port=8076 --dashboard-host=0.0.0.0

# 工作节点启动命令
worker_start_ray_commands:
  - ray stop
  - ray start --address=$RAY_HEAD_ADDRESS --object-manager-port=8076
```

---

## 9. 性能对比与最佳实践

### 9.1 TCP vs RDMA 性能对比

根据实际生产环境的测试数据：

| 场景            | TCP (global_step/sec) | RDMA (global_step/sec) | 提升       | 训练时长节省 |
|---------------|-----------------------|------------------------|----------|--------|
| 大规模稀疏模型 (83天) | 20.31                 | 25.21                  | **+24%** | ~5 小时  |
| 中等规模模型 (7天)   | 7.2                   | 8.0                    | **+11%** | ~40 分钟 |

**效果指标一致性**：RDMA 切换后，模型评估指标（AUC/GAUC/calibration）在波动范围内一致，不影响模型质量。

### 9.2 最佳实践清单

#### ✅ 必做项

- [ ] **确认硬件支持**：网卡支持 RoCE/IB，驱动正确安装
- [ ] **NCCL 版本匹配**：使用与 CUDA 版本匹配的 NCCL，且编译时包含 IB verbs 支持
- [ ] **网络配置正确**：PFC/ECN 已在交换机上正确配置
- [ ] **设置环境变量**：`NCCL_IB_DISABLE=0`, `NCCL_IB_GID_INDEX=3`
- [ ] **验证生效**：通过 `NCCL_DEBUG=INFO` 日志确认使用了 `IB:` 而非 `Socket:`

#### 🔧 推荐配置

```bash
# 推荐的 RDMA 环境变量组合
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3           # RoCE v2
export NCCL_IB_TIMEOUT=23            # 大规模训练建议增大超时
export NCCL_IB_RETRY_CNT=7
export NCCL_SOCKET_IFNAME=<你的RDMA网卡>
export NCCL_ALGO=Ring                # 或 Tree/RingTree
export NCCL_NVLS_ENABLE=0           # 非 NVLink 场景关闭

# GPU 通信优化
export CUDA_DEVICE_MAX_CONNECTIONS=1  # 通信与计算重叠
export NCCL_ASYNC_ERROR_HANDLING=1
```

#### ⚠️ 注意事项

1. **混合网络问题**：如果节点同时有多张网卡（一张 RDMA，一张普通以太网），务必通过 `NCCL_IFNAME` 或 `NCCL_SOCKET_IFNAME` 明确指定
   RDMA 网口
2. **MTU 配置**：RDMA 网络通常需要较大的 MTU（9000+），确保路径上所有设备的 MTU 一致
3. **防火墙**：RDMA 使用特定端口范围，确保防火墙不会阻断 RDMA 流量
4. **多节点场景**：跨节点 RDMA 需要交换机支持 PFC/ECN，否则可能出现丢包导致性能下降
5. **故障排查**：如果 RDMA 不工作，可以临时设置 `NCCL_IB_DISABLE=1` 回退到 TCP 进行对比测试

### 9.3 故障排查指南

| 症状                           | 可能原因               | 解决方案                                       |
|------------------------------|--------------------|--------------------------------------------|
| NCCL 日志显示 `Socket:` 而非 `IB:` | 未检测到 RDMA 设备       | 检查 `ibv_devinfo`，确认驱动安装                    |
| 训练卡住/超时                      | RDMA 网络拥塞或 MTU 不匹配 | 检查 PFC/ECN 配置，统一 MTU                       |
| `NCCL WARN` 重复重试             | 网络不稳定              | 增大 `NCCL_IB_RETRY_CNT` 和 `NCCL_IB_TIMEOUT` |
| 性能未提升                        | NCCL 编译时未包含 IB 支持  | 重新编译 NCCL 或更换预编译版本                         |
| 部分节点使用 TCP                   | 混合网络环境             | 显式指定 `NCCL_IFNAME`                         |

---

## 附录：快速参考

### A. 常用环境变量速查

```bash
# RDMA 核心
NCCL_IB_DISABLE=0          # 0=启用RDMA, 1=禁用(回退TCP)
NCCL_IB_GID_INDEX=3        # GID索引 (3=RoCEv2)
NCCL_IB_TIMEOUT=23         # 超时 (4.096μs × 2^23 ≈ 34s)
NCCL_IB_RETRY_CNT=7        # 重试次数

# 网络选择
NCCL_IFNAME=eth0           # 优先使用的网络接口
NCCL_SOCKET_IFNAME=eth0    # Socket fallback 接口

# 调试
NCCL_DEBUG=INFO            # 日志级别
NCCL_DEBUG_SUBSYS=ALL      # 调试子系统
```

### B. 常用命令速查

```bash
# RDMA 设备检查
ibv_devinfo                # 设备详情
ibstat                     # 链路状态
rdma link                  # RDMA 链路
rdma dev                   # RDMA 设备列表

# 性能测试
ib_write_bw -d mlx5_0      # Write 带宽
ib_read_bw -d mlx5_0       # Read 带宽
ib_send_bw -d mlx5_0       # Send 带宽
ib_send_lat -d mlx5_0      # Send 延迟

# 网络配置
mlnx_qos -i eth0           # PFC/QoS 配置
mlnx_perf -i eth0          # 流量监控
```

### C. 参考资料

- [PyTorch Distributed 文档](https://pytorch.org/docs/stable/distributed.html)
- [NCCL 官方文档](https://docs.nvidia.com/deeplearning/nccl/)
- [RDMA Verbs API 规范](https://docs.rdma.verbs.api/)

