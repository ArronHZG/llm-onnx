# Mooncake 全局 KV Cache 工作机制详解

## **Mooncake 全局 KV Cache 工作机制详解**

Mooncake 全局 KV Cache 是一个分布式键值缓存存储引擎，专为 LLM 推理场景设计。它将集群中所有节点的内存聚合为一个巨大的分布式内存池,实现 KV cache 的跨实例共享和高效访问。



### **整体架构**

![mooncake-store-preview](https://github.com/kvcache-ai/Mooncake/blob/main/image/mooncake-store-preview.png?raw=true")

graph TB
    subgraph "Inference Nodes"
        I1["Client 1<br/>(Prefill Node)<br/>Local: GPU + Host Memory<br/>Global: 0GB"]
        I2["Client 2<br/>(Decode Node)<br/>Local: GPU + Host Memory<br/>Global: 0GB"]
        I3["Client 3<br/>(Storage Node)<br/>Local: None<br/>Global: 60GB"]
    end
    

```mermaid
graph TB
subgraph "Control Plane"
    MS["Master Service<br/>- 空间分配<br/>- 元数据管理<br/>- 淘汰策略<br/>- 租约机制"]
    MDS["Metadata Service<br/>(etcd/Redis/HTTP)<br/>- 网络拓扑<br/>- 连接信息"]
end

subgraph "Data Plane"
    TE["Transfer Engine<br/>- RDMA/TCP 传输<br/>- 零拷贝<br/>- 多网卡聚合"]
    RDMA["RDMA Network<br/>- GPUDirect RDMA<br/>- 高速数据传输"]
end

subgraph "Storage Layer"
    MEM["Distributed Memory Pool<br/>- 聚合所有节点内存<br/>- 分段管理"]
    SSD["SSD/DFS<br/>(可选持久化层)<br/>- 冷数据卸载"]
end

I1 --"1. Get/Put 请求"--> MS
I2 --"1. Get/Put 请求"--> MS
I3 --"2. MountSegment 注册内存"--> MS
MS --"3. PutStart 分配空间<br/>GetReplicaList 查询"--> I1
MS --"3. PutStart 分配空间<br/>GetReplicaList 查询"--> I2
MS --"3. PutStart 分配空间<br/>GetReplicaList 查询"--> I3
I1 --"4. RDMA 读写"--> TE
I2 --"4. RDMA 读写"--> TE
I3 --"4. RDMA 读写"--> TE
TE --> RDMA
TE --"注册/查询 Segment"--> MDS
MS -."元数据管理".-> MDS
```



### **核心工作流程**

#### **1. 写入流程**

```mermaid
sequenceDiagram
    participant C as Client (Producer)
    participant M as Master Service
    participant S as Storage Client (Provider)
    participant T as Transfer Engine
    participant R as RDMA Network
    
    Note over C,R: Put 操作:存储 KV Cache
    
    C->>M: 1. PutStart(key, size, config)
    activate M
    
    Note right of M: 分配策略:<br/>- 根据配置选择副本数<br/>- 优先分配到 preferred_segment<br/>- 不同 slice 分布在不同 segment

    M->>M: AllocationStrategy.Allocate()
    M-->>C: 2. PutStartResponse<br/>返回 replica_list (包含 buffer 地址)
    deactivate M
    
    Note over C,R: 数据传输 (零拷贝)
    C->>T: 3. SubmitTransfer<br/>(remote buffer addresses)
    activate T
    loop 每个副本
        T->>R: 4. RDMA Write<br/>直接写入远程内存
        R->>S: 数据到达
    end
    T-->>C: 5. Transfer Complete
    deactivate T
    
    C->>M: 6. PutEnd(key)
    activate M
    M->>M: 更新副本状态为 COMPLETE
    M-->>C: 7. PutEndResponse
    deactivate M
```

#### **2. 读取流程**

```mermaid
sequenceDiagram
    participant C as Client (Consumer)
    participant L1 as L1 Cache (GPU)
    participant L2 as L2 Cache (Host)
    participant M as Master Service
    participant S as Storage Client (Provider)
    participant R as RDMA Network
    
    Note over C,R: Get 操作:读取 KV Cache
    
    C->>L1: 1. 查询本地 L1
    alt L1 命中
        L1-->>C: 直接返回
    else L1 未命中
        C->>L2: 2. 查询本地 L2
        alt L2 命中
            L2-->>C: 返回数据
        else L2 未命中
            C->>M: 3. GetReplicaList(key)
            activate M
            Note right of M: 选择最佳副本:<br>- 优先本地副本<br>- 选择引用计数低<br>- 考虑网络拓扑
            M-->>C: 4. 返回 replica_list
            deactivate M
            
            C->>M: 5. ExistKey(key)<br/>申请租约 (防止淘汰)
            activate M
            M->>M: 更新租约 TTL (默认 5s)
            M-->>C: 租约授权
            deactivate M
            
            Note over C,R: 零拷贝传输
            C->>S: 6. RDMA Read<br/>从远程内存直接读取
            S-->>C: 数据直达本地内存
            
            C->>C: 7. 写入 L2 缓存
        end
    end
```

### **元数据管理与一致性**

```mermaid
graph TB
    subgraph "Master Service - 1024 Shards"
        S1["Shard 0<br/>metadata[0:1023]"]
        S2["Shard 1<br/>metadata[1024:2047]"]
        S3["Shard N<br/>metadata[N*1024:(N+1)*1023]"]
        S1024["Shard 1023<br/>metadata[1047552:1048575]"]
    end
    
    subgraph "每个 Shard 包含"
        METADATA["std::unordered_map<string, ObjectMetadata>"]
        PROCESSING["std::unordered_set<string><br/>processing_keys"]
        REPLICATION["std::unordered_map<string, ReplicationTask>"]
    end
    
    subgraph "ObjectMetadata 结构"
        O_KEY["string key"]
        O_CLIENT["UUID client_id"]
        O_SIZE["uint64_t size"]
        O_REPLICAS["vector<Replica> replicas<br/>- segment_name<br/>- buffer 地址<br/>- 状态"]
        O_PIN["bool soft_pin<br/>软固定标记"]
        O_PIN_TTL["chrono::time_point pin_timeout"]
        O_LEASE["chrono::time_point lease_timeout"]
        O_LEASE_CNT["uint32_t refcnt<br/>引用计数"]
    end
    
    H["hash(key) % 1024"] --> S1
    
    METADATA --> O_KEY
    METADATA --> O_CLIENT
    METADATA --> O_SIZE
    METADATA --> O_REPLICAS
    METADATA --> O_PIN
    METADATA --> O_PIN_TTL
    METADATA --> O_LEASE
    METADATA --> O_LEASE_CNT
```

### **淘汰策略与租约机制**

```mermaid
stateDiagram-v2
    [*] --> INIT: PutStart 分配
    INIT --> PROCESSING: 正在写入
    PROCESSING --> COMPLETE: PutEnd 完成
    PROCESSING --> FAILED: PutRevoke 撤销<br/>或超时 (30s)
    FAILED --> [*]: 空间回收 (10m)
    
    COMPLETE --> [*]: Remove 删除
    COMPLETE --> [*]: 淘汰触发
    
    note right of COMPLETE
        淘汰条件:
        1. 内存使用率 > 高水位 (95%)
        2. PutStart 失败触发
        3. 租约过期
        4. 引用计数 = 0
        
        淘汰顺序:
        1. 优先淘汰未 soft_pin 对象
        2. 再淘汰 soft_pin 但 TTL 过期对象
        3. 最后淘汰 soft_pin 对象
    end note
    
    note left of PROCESSING
        租约保护:
        - GetReplicaList 成功授予租约 (5s)
        - 租约期间不会被淘汰/删除
        - 租约过期后可被淘汰
    end note
```

### **分层存储架构**

```mermaid
graph TB
    subgraph "三层缓存 (与 SGLang HiCache 集成)"
        L1["L1: GPU Memory<br/>- 最快访问<br/>- 容量最小<br/>- 每实例私有"]
        L2["L2: Host Memory<br/>- 快速访问<br/>- 容量中等<br/>- 每实例私有"]
        L3["L3: Mooncake Store<br/>- 分布式内存池<br/>- 跨实例共享<br/>- RDMA 加速"]
    end
    
    subgraph "L3 内部分层"
        L3_MEM["L3-Memory<br/>- 热数据<br/>- 高速内存<br/>- 零拷贝传输"]
        L3_SSD["L3-SSD/DFS<br/>- 冷数据<br/>- 异步持久化<br/>- 内存溢出时卸载"]
    end
    
    L1 --"未命中"--> L2
    L2 --"未命中"--> L3_MEM
    L3_MEM --"淘汰/老化"--> L3_SSD
    L3_SSD --"访问"--> L3_MEM
    
    subgraph "数据移动"
        PRE["Prefetch<br/>预取热点数据"]
        WB["Write-back<br/>异步写入"]
        EVICT["Evict<br/>淘汰冷数据"]
    end
    
    L2 --"需要"--> PRE
    PRE --> L2
    L1 --> WB
    WB --> L3_MEM
    L3_MEM --> EVICT
```

### **僵尸对象清理机制**

```mermaid
sequenceDiagram
    participant C as Client
    participant M as Master Service
    participant T as Task Cleanup Thread
    
    Note over C,T: 防止 PutStart 后崩溃导致僵尸对象
    
    C->>M: PutStart(key) - 时间戳 t0
    Note over C: 客户端崩溃/网络故障<br/>未发送 PutEnd/PutRevoke
    
    rect rgb(255, 200, 200)
        Note right of M: 时间区间 1: t0 + 30s (put_start_discard_timeout)
        M->>M: 检测到新的 PutStart(key)<br/>允许"顶替"旧 PutStart
        Note right of M: 重新分配空间,<br/>旧空间标记为可回收
    end
    
    rect rgb(255, 255, 200)
        Note right of M: 时间区间 2: t0 + 10min (put_start_release_timeout)
        M->>M: 空间仍未完成<br/>标记为 RELEASED
        Note right of M: 进入回收队列,<br/>等待淘汰线程回收
    end
    
    T->>M: 定期检查 (每 30s)
    M->>M: 释放过期空间<br/>更新可用容量
    M-->>T: 清理完成数量
```

### **关键特性总结**

1. 零拷贝传输:基于 Transfer Engine,通过 RDMA 直接在远程内存和本地内存间传输,完全绕过 CPU
2. 控制流与数据流分离:Master Service 仅管理元数据,实际数据传输直接在不同 Client 之间进行
3. 副本管理:支持多副本,同一对象的不同 slice 保证分布在不同 segment,尽力而为分配
4. 强一致性:Get 操作保证返回完整正确的数据,Put 后数据不可变
5. 容错能力:任意数量 Master 和 Client 故障都不会读到错误数据,只要有至少一个 Master 和 Client 正常运行
6. 软固定机制:重要对象(如 system prompt)可软固定,淘汰时优先保留,长时间未访问自动解除
7. 分级存储:支持内存到 SSD 的分层缓存,异步持久化,平衡性能与成本

这个架构使 Mooncake 能够在大规模 LLM 推理场景中,提供 virtually unlimited 的 KV cache 容量,同时保持高带宽、低延迟的数据访问性能。



```
mooncake_master \
  --enable_http_metadata_server=true \
  --http_metadata_server_host=0.0.0.0 \
  --http_metadata_server_port=44407
```



## Mooncake Master 功能讲解

Mooncake Master 是 Mooncake 分布式存储系统的核心协调服务，负责管理元数据和协调存储节点。

### 核心功能

| 功能             | 说明                                           |
| ---------------- | ---------------------------------------------- |
| **RPC 服务**     | 提供 gRPC 风格的服务，处理客户端的 KV 操作请求 |
| **元数据管理**   | 存储和管理键值对的元数据信息                   |
| **存储后端管理** | 支持多种内存分配器（cachelib、offset）         |
| **缓存策略**     | 支持软钉住（soft pin）对象的淘汰策略           |
| **任务管理**     | 管理 Put/Get 任务的执行和超时                  |

### 启动参数分类

#### 1. **RPC 服务配置**

```
DEFINE_int32(rpc_port, 0, "RPC 服务器端口");
DEFINE_int32(rpc_thread_num, 0, "RPC 线程数");
DEFINE_string(rpc_address, "0.0.0.0", "绑定地址");
DEFINE_bool(rpc_enable_tcp_no_delay, true, "禁用 Nagle's 算法");
```

#### 2. **元数据服务器配置**

```
DEFINE_bool(enable_http_metadata_server, false, "启用 HTTP 元数据服务器");
DEFINE_int32(http_metadata_server_port, 8080, "HTTP 元数据服务器端口");
DEFINE_string(http_metadata_server_host, "0.0.0.0", "HTTP 绑定地址");
```

#### 3. **高可用配置 (HA)**

```
DEFINE_bool(enable_ha, false, "启用高可用模式");
DEFINE_string(etcd_endpoints, "", "etcd 集群地址");
DEFINE_string(cluster_id, "", "集群 ID");
DEFINE_int64(client_ttl, ..., "客户端存活超时时间");
```

#### 4. **存储后端配置**

```
DEFINE_string(memory_allocator, "offset", "内存分配器: cachelib | offset");
DEFINE_bool(enable_cxl, false, "启用 CXL 内存支持");
DEFINE_string(cxl_path, ..., "CXL 设备路径");
DEFINE_uint64(cxl_size, ..., "CXL 内存大小");
DEFINE_bool(enable_disk_eviction, true, "启用磁盘淘汰");
DEFINE_double(eviction_ratio, ..., "淘汰比例");
```

#### 5. **KV 生命周期配置**

```
DEFINE_uint64(default_kv_lease_ttl, ..., "KV 租约默认 TTL");
DEFINE_uint64(default_kv_soft_pin_ttl, ..., "软钉住对象 TTL");
DEFINE_bool(allow_evict_soft_pinned_objects, ..., "是否允许淘汰软钉住对象");
```

#### 6. **监控指标**

```
DEFINE_bool(enable_metric_reporting, true, "启用指标上报");
DEFINE_int32(metrics_port, 9003, "指标 HTTP 服务器端口");
```

### 主流程 ([main](file:///Users/arron/Projects/Mooncake/mooncake-store/src/master.cpp#478#4) 函数)

```
int main(int argc, char* argv[]) {
    // 1. 解析命令行参数
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    // 2. 初始化配置（从文件 + 命令行）
    LoadConfigFromCmdline(master_config, ...);
    
    // 3. 启动 HTTP 元数据服务器（如果启用）
    if (master_config.enable_http_metadata_server) {
        StartHttpMetadataServer(...);
    }
    
    // 4. 启动 RPC 服务
    if (master_config.enable_ha) {
        // HA 模式：启动 MasterServiceSupervisor
        supervisor.Start();
    } else {
        // 非 HA 模式：直接启动 RPC 服务器
        coro_rpc::coro_rpc_server server(...);
        mooncake::RegisterRpcService(server, wrapped_master_service);
        server.start();
    }
}
```

### 架构定位

```
┌─────────────────────────────────────────────────────┐
│                    Client                            │
└───────────────────┬─────────────────────────────────┘
                    │ RPC 请求
                    ▼
┌─────────────────────────────────────────────────────┐
│              Mooncake Master                         │
│  ┌─────────────────────────────────────────────────┐│
│  │  • 元数据管理                                   ││
│  │  • 路由决策                                     ││
│  │  • 负载均衡                                     ││
│  │  • 故障检测                                     ││
│  └─────────────────────────────────────────────────┘│
└───────────────────┬─────────────────────────────────┘
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
┌───────────┐ ┌───────────┐ ┌───────────┐
│  Store 1  │ │ Store 2   │ │ Store N   │
│  (RDMA)   │ │  (TCP)    │ │  (NVLink) │
└───────────┘ └───────────┘ └───────────┘
```

Mooncake 是一个类似 Redis / Memcached 的分布式缓存系统，但针对 **RDMA / NVLink** 等高速互连进行了优化，支持更高效的远程内存访问。





https://kvcache-ai.github.io/Mooncake/design/mooncake-store.html