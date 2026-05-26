# 弹性Rollout与训练调度系统设计方案

## 1. 整体架构设计

### 1.1 资源划分

```mermaid
graph TB
    subgraph "Total GPU Resources"
        subgraph "Rollout Resources (固定)"
            R1[Rollout DP Group 1]
            R2[Rollout DP Group 2]
            R3[Rollout DP Group N]
        end
        
        subgraph "Train Resources (固定)"
            T1[Train DP Group 1]
            T2[Train DP Group 2]
            T3[Train DP Group M]
        end
        
        subgraph "Elastic Resources (可变)"
            E1[Elastic DP Group 1]
            E2[Elastic DP Group 2]
            E3[Elastic DP Group K]
        end
    end
    
    subgraph "3D-HybridEngine"
        direction TB
        HE[HybridEngine Controller]
        HE -->|切换为Rollout| E_R[Rollout Mode]
        HE -->|切换为Train| E_T[Train Mode]
    end
    
    E1 --> HE
    E2 --> HE
    E3 --> HE
    
    R1 --> RM[Rollout Manager]
    R2 --> RM
    R3 --> RM
    E_R --> RM
    
    T1 --> TM[Train Manager]
    T2 --> TM
    T3 --> TM
    E_T --> TM
    
    RM --> RC[Resource Coordinator]
    TM --> RC
    RC -->|监控| MQ[Message Queue]
    RC -->|调度| HE
```

### 1.2 核心数据流

```mermaid
sequenceDiagram
    participant RC as Resource Coordinator
    participant RM as Rollout Manager
    participant TM as Train Manager
    participant HE as 3D-HybridEngine
    participant MQ as Message Queue
    participant CM as Checkpoint Manager
    
    Note over RC: 初始化阶段
    RC->>HE: 初始化弹性资源池
    
    Note over RC: 监控循环
    loop 持续监控
        RC->>RM: 获取rollout生产速率
        RC->>TM: 获取train消费速率
        RC->>RC: 计算拥塞状态
        
        alt 拥塞检测
            RC->>HE: 请求资源切换
            HE->>HE: 角色切换
            HE->>CM: 同步新加入资源的参数
            CM->>RM: 分发参数到新rollout实例
            CM->>TM: 分发参数到新train实例
        end
    end
    
    Note over RC: 参数同步时机
    alt 参数同步前
        RC->>CM: 触发参数同步
        CM->>HE: 获取当前弹性角色
        HE-->>CM: 返回角色状态
    end
```

### 1.3 架构设计

```mermaid
graph TB
    subgraph "资源层 (GPU集群)"
        subgraph "Rollout资源池 (固定)"
            R_POOL["Rollout HybridEngine Pool<br/>(ActorRolloutRefWorker)"]
        end
        
        subgraph "Train资源池 (固定)"
            T_POOL["Train HybridEngine Pool<br/>(ActorRolloutRefWorker)"]
        end
        
        subgraph "Elastic资源池 (可变)"
            E_POOL["Elastic HybridEngine Pool<br/>(ActorRolloutRefWorker)"]
        end
    end
    
    subgraph "调度层"
        RC["Resource Coordinator<br/>(弹性调度器)"]
        QM["Queue Monitor<br/>(拥塞监控)"]
        SM["Switch Manager<br/>(角色切换管理)"]
    end
    
    subgraph "同步层"
        PS["Parameter Sync<br/>(参数同步引擎)"]
        CG["Checkpoint Manager<br/>(支持新增DP的同步)"]
    end
    
    RC <-->|监控拥塞| QM
    RC <-->|触发切换| SM
    SM <-->|切换模式| R_POOL
    SM <-->|切换模式| T_POOL
    SM <-->|切换模式| E_POOL
    
    PS <-->|参数同步| CG
    
    R_POOL -->|生产samples| MQ["Message Queue"]
    T_POOL -->|消费samples| MQ
    E_POOL -->|弹性samples| MQ
```

## 3. 实施计划

### Phase 1: 核心框架 (1-2周)

1. 创建 `verl/experimental/elastic_scheduling/` 目录结构
2. 实现 `ElasticResourceManager` - 资源管理
3. 实现 `CongestionMonitor` - 拥塞监控
4. 实现 `ResourceCoordinator` - 调度协调

### Phase 2: Rollouter/Trainer集成 (2-3周)

1. 实现 `ElasticRollouter` - 继承FullyAsyncRollouter
2. 实现 `ElasticTrainer` - 继承FullyAsyncTrainer
3. 实现 `ElasticCheckpointManager` - 参数同步

### Phase 3: 引擎集成 (2周)

1. 支持FSDP2的动态DP切换
2. 支持Megatron的动态DP切换
3. 集成3D-HybridEngine

### Phase 4: 测试与优化 (1-2周)

1. 单元测试
2. 集成测试
3. 性能优化

## 4. 关键设计决策

### 4.1 为什么基于ActorRolloutRefWorker?

- 它已经是成熟的HybridEngine实现
- 支持FSDP2和Megatron后端
- 有完整的参数同步机制
- 可以在rollout和train模式间切换

### 4.2 DP通讯组处理

对于FSDP2:

- 利用 `veomni.distributed.parallel_state` 动态调整DP大小
- 保持TP/EP不变

对于Megatron:

- 调用 `mpu.destroy_model_parallel()` 重建
- 需要协调多个PP stage

### 4.3 参数同步策略

- 使用现有的 `CheckpointEngine` 架构
- 支持增量同步（只同步变化部分）
- 对于新增DP，广播完整参数

