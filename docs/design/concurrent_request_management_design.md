# 并发请求管理方案设计

## 文档概述

本文档设计一个完整的并发请求管理系统，涵盖从HTTP Server接收请求到Inference引擎处理请求的完整生命周期，包括：
- HTTP层请求队列管理
- 最大并发请求数限制
- KV缓存管理和淘汰策略
- 序列ID（llama_seq_id）生命周期管理
- 请求状态流转

## 核心设计原则

1. **分层管理**：HTTP层、调度层、推理层各自管理自己的资源限制
2. **提前拒绝**：在HTTP层就拒绝超过最大并发数的请求，避免资源浪费（通过查询Scheduler状态）
3. **KV缓存淘汰**：主动淘汰长时间未使用的KV缓存，释放资源
4. **序列ID绑定**：序列ID与请求ID绑定，避免重用冲突
5. **统一管理**：所有请求状态在Scheduler层统一管理（runningRequests_），避免多套状态系统

## 系统架构概览

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                          HTTP Server Layer                                        │
│                                                                                   │
│  HTTP Request (requestId: size_t)                                                  │
│         │                                                                         │
│         ▼                                                                         │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ HTTP Request Handler                                                        │   │
│  │ - HTTP层接收请求，直接调用Scheduler::addRequest()                            │   │
│  │ - 请求进入Scheduler的RequestQueue（见Scheduler层）                            │   │
│  │ - 注意：当前实现中，HTTP层不检查并发数，直接加入队列                          │   │
│  │   未来优化：可在HTTP层通过Scheduler::getRunningCount()检查并发数              │   │
│  └────────┬─────────────────────────────────────────────────────────────────┘   │
│           │                                                                       │
│           │ [HTTP层调用Scheduler::addRequest()]                                   │
└───────────┼───────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌───────────┼───────────────────────────────────────────────────────────────────────┐
│           │                    Scheduler Layer                                     │
│           │                                                                       │
│           │  ┌────────────────────────────────────────────────────────────┐      │
│           │  │ RequestQueue (max: 256)                                     │      │
│           │  │ - HTTP层通过Scheduler::addRequest()将请求加入此队列          │      │
│           │  │ - 最外层队列，接收所有HTTP请求                                │      │
│           │  │ - Scheduler从此队列取出请求加入runningRequests_             │      │
│           │  └────────┬───────────────────────────────────────────────────┘      │
│           │           │                                                           │
│           │           │ [检查runningRequests_.size() < 8]                         │
│           │           │ 如果未满，从RequestQueue取出请求加入runningRequests_     │
│           │           │                                                           │
│           │           ▼                                                           │
│           │  ┌────────────────────────────────────────────────────────────┐      │
│           │  │ runningRequests_ (max: 8)                                  │      │
│           │  │ - 替代ConcurrentQueue，统一管理所有正在处理的请求            │      │
│           │  │ - 并发控制（限制同时处理的请求数量）                         │      │
│           │  │ - 状态管理（PENDING → PROCESSING → COMPLETED/TIMEOUT/FAILED）│     │
│           │  │ - 超时检测（requestTimeout: 60s，定期扫描）                  │      │
│           │  └────────┬───────────────────────────────────────────────────┘      │
│           │           │                                                           │
│           │           ▼                                                           │
│           │  ┌────────────────────────────────────────────────────────────┐      │
│           │  │ Batch Formation                                            │      │
│           │  │ - 从runningRequests_中获取请求（maxBatchSize: 4）           │      │
│           │  │ - maxBatchSize ≤ maxConcurrentRequests                     │      │
│           │  │ - 输入限制：maxInputContextLength: 128                      │      │
│           │  │ - 输出限制：maxOutContextLength: 2048                       │      │
│           │  │ - 批处理中的所有请求标记为PROCESSING                         │      │
│           │  └────────┬───────────────────────────────────────────────────┘      │
│           │           │                                                           │
│           │           │ [形成批处理]                                              │
└───────────┼───────────┼───────────────────────────────────────────────────────────┘
            │           │
            │           ▼
┌───────────┼───────────┼───────────────────────────────────────────────────────────┐
│           │           │              Inference Layer (llama.cpp)                  │
│           │           │                                                           │
│           │           │  ┌────────────────────────────────────────────────┐      │
│           │           │  │ Sequence ID Manager                             │      │
│           │           │  │ - 序列ID池（范围：0到n_seq_max-1，默认值1，可配置1-256）│      │
│           │           │  │ - 基于requestId分配序列ID                        │      │
│           │           │  │ - 请求完成时自动释放序列ID                       │      │
│           │           │  │ - 序列ID是llama.cpp内部使用，不直接对应KV缓存     │      │
│           │           │  │ - 注意：n_seq_max可配置，默认值1，最大值256        │      │
│           │           │  └────────┬───────────────────────────────────────┘      │
│           │           │           │                                               │
│           │           │           ▼                                               │
│           │           │  ┌────────────────────────────────────────────────┐      │
│           │           │  │ llama.cpp Backend                               │      │
│           │           │  │ - nBackendBatch: 512 (一次处理的最大token数)     │      │
│           │           │  │ - 使用分配的序列ID进行推理                        │      │
│           │           │  │ - 推理结果（KV缓存）基于requestId存储             │      │
│           │           │  └────────┬───────────────────────────────────────┘      │
│           │           │           │                                               │
│           │           │           ▼                                               │
│           │           │  ┌────────────────────────────────────────────────┐      │
│           │           │  │ KV Cache Manager (基于requestId, LRU策略)        │      │
│           │           │  │ - 基于requestId管理KV缓存（不是序列ID）           │      │
│           │           │  │ - 每个requestId对应一个KV缓存（包含该请求的所有token）│
│           │           │  │ - maxKVCachesItems: 4*1024*1024 (缓存条目总数限制)  │
│           │           │  │ - kvCacheMaxMemoryMb: 1024 (内存限制)           │      │
│           │           │  │ - kvCacheEvictionThreshold: 0.8 (80%触发淘汰)   │      │
│           │           │  │ - kvCacheCleanupInterval: 1s (清理间隔)          │      │
│           │           │  │ - LRU淘汰：按requestId淘汰整个请求的KV缓存        │      │
│           │           │  │ - 淘汰保护：正在处理的请求（PROCESSING）的KV缓存不淘汰│
│           │           │  └────────────────────────────────────────────────┘      │
│           │           │                                                           │
│           │           │ [处理完成/失败]                                           │
│           │           │                                                           │
│           └───────────┼───────────────────────────────────────────────────────────┘
│                       │
│                       ▼
│            [请求完成/超时/失败]                                                  │
│                       │                                                           │
│                       ▼                                                           │
│            [从runningRequests_释放请求]                                           │
│                       │                                                           │
│                       ├─ [释放序列ID]                                             │
│                       ├─ [清理KV缓存]                                             │
│                       └─ [通知HTTP层]                                             │
│                       │                                                           │
│                       ▼                                                           │
│            [Scheduler回调HTTP层]                                                  │
│                       │                                                           │
│                       ▼                                                           │
│            [HTTP Response (requestId: size_t)]                                   │
│                       │                                                           │
│                       ▼                                                           │
│            [从RequestQueue取出下一个请求加入runningRequests_]                 │
└───────────────────────────────────────────────────────────────────────────────────┘
```

### 数据流说明

1. **HTTP请求进入** → 加入 `RequestQueue` (max: 256)
   - HTTP层直接调用`Scheduler::addRequest()`，请求进入Scheduler的`RequestQueue`
   - 队列满 → 返回429（如果实现队列大小检查）
   - **注意**：当前实现中，HTTP层不检查并发数，直接加入Scheduler队列
   - **未来优化**：可在HTTP层通过`Scheduler::getRunningCount()`检查并发数，超过时直接返回429

2. **Scheduler获取请求** → 加入 `runningRequests_` (max: 8)
   - Scheduler从`RequestQueue`取出请求
   - 检查`runningRequests_.size() < maxConcurrentRequests (8)`
     - 如果满了 → 等待，稍后重试
     - 否则 → 加入runningRequests_，状态为PENDING

3. **Scheduler批处理** → 从runningRequests_获取请求（maxBatchSize: 4）
   - 批处理中的所有请求标记为PROCESSING
   - 输入限制：maxInputContextLength (128)
   - 输出限制：maxOutContextLength (2048)

4. **Inference处理** → llama.cpp Backend
   - **序列ID分配**：基于requestId从序列ID池分配（范围：0到n_seq_max-1，默认值1，可配置1-256）
     - 序列ID是llama.cpp内部使用的临时标识
     - 注意：n_seq_max可配置，默认值1（llama.cpp默认），最大值256
   - **KV缓存存储**：基于requestId存储KV缓存，每个requestId对应一个完整的KV缓存
     - 每个requestId的KV缓存包含该请求的所有token的KV数据
     - KV缓存条目数 = 该requestId的序列长度（token数量）
   - 批处理大小：nBackendBatch (512)
   - KV缓存管理（LRU策略）：
     - 基于requestId管理，按requestId淘汰整个请求的KV缓存
     - 只淘汰未使用的缓存（状态为PENDING或COMPLETED的requestId对应的KV缓存）
     - 条目总数限制：maxKVCachesItems (4*1024*1024)
     - 内存限制：kvCacheMaxMemoryMb (1024MB)

5. **请求完成/超时/失败** → 从runningRequests_释放
   - **完成**：标记为COMPLETED → 释放序列ID → 基于requestId清理KV缓存 → 通知HTTP层 → 返回HTTP Response
   - **超时**：超时检测（requestTimeout: 60s）→ 标记为TIMEOUT → 释放序列ID → 基于requestId清理KV缓存 → 通知HTTP层 → 返回HTTP 408
   - **失败**：标记为FAILED → 释放序列ID → 基于requestId清理KV缓存 → 通知HTTP层 → 返回HTTP 500
   - **KV缓存清理**：通过requestId找到对应的KV缓存，删除整个请求的KV缓存，并更新缓存条目总数和内存使用

6. **触发下一个请求** → 从HttpRequestQueue取出下一个请求
   - 请求释放后，Scheduler自动从HttpRequestQueue取出下一个请求
   - 如果runningRequests_未满，立即加入runningRequests_

### runningRequests_状态机定义

**注意**：采用方案B后，移除了ConcurrentQueue，所有状态管理统一在Scheduler的`runningRequests_`中。

`runningRequests_`中的每个请求都有明确的状态，状态转换如下：

```
┌─────────────────────────────────────────────────────────────────┐
│              runningRequests_ 请求状态机                         │
│                                                                  │
│  Scheduler从RequestQueue取出请求                              │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────┐                                                │
│  │  PENDING    │  ◄─────── 新请求从RequestQueue加入runningRequests_│
│  │ (等待批处理)│                                                │
│  └─────┬───────┘                                                │
│        │                                                        │
│        │ [Scheduler形成批处理]                                   │
│        ▼                                                        │
│  ┌─────────────┐                                                │
│  │ PROCESSING  │  ◄─────── Scheduler正在处理（Inference中）      │
│  │ (处理中)    │                                                │
│  └─────┬───────┘                                                │
│        │                                                        │
│        ├────────────────┬─────────────────┐                     │
│        │                │                 │                     │
│        │ [处理完成]     │ [超时]          │ [处理失败]           │
│        ▼                ▼                 ▼                     │
│  ┌───────────┐  ┌───────────┐   ┌───────────┐                │
│  │ COMPLETED │  │  TIMEOUT  │   │   FAILED  │                │
│  │ (已完成)  │  │  (超时)   │   │  (失败)   │                │
│  └─────┬─────┘  └─────┬─────┘   └─────┬─────┘                │
│        │              │               │                        │
│        └──────────────┴───────────────┘                        │
│                     │                                           │
│                     ├─ [释放序列ID]                              │
│                     ├─ [清理KV缓存]                              │
│                     ├─ [通知HTTP层]                              │
│                     ▼                                           │
│           [从runningRequests_释放]                               │
│                     │                                           │
│                     ▼                                           │
│           [从RequestQueue取下一个请求加入runningRequests_]   │
└─────────────────────────────────────────────────────────────────┘
```

#### 状态说明

| 状态 | 说明 | 转换条件 | 行为 |
|------|------|----------|------|
| **PENDING** | 等待批处理 | 从RequestQueue加入runningRequests_ | 请求在runningRequests_中，等待被Scheduler形成批处理 |
| **PROCESSING** | 正在处理 | Scheduler形成批处理，开始Inference | 请求正在被处理（Inference层），状态为"处理中" |
| **COMPLETED** | 处理完成 | Inference处理成功 | 请求处理完成，释放序列ID和KV缓存，通知HTTP层返回响应 |
| **TIMEOUT** | 请求超时 | 处理时间 > requestTimeout (60s) | 请求超时，释放序列ID和KV缓存，通知HTTP层返回HTTP 408 |
| **FAILED** | 处理失败 | Inference层处理失败 | 请求处理失败，释放序列ID和KV缓存，通知HTTP层返回HTTP 500 |

#### 状态管理要点

1. **PENDING → PROCESSING**：
   - Scheduler形成批处理时，批处理中的所有请求状态变为PROCESSING
   - 请求仍在runningRequests_中，标记为"处理中"

2. **PROCESSING → COMPLETED**：
   - Inference处理完成并返回结果时，状态变为COMPLETED
   - 释放序列ID → 清理KV缓存 → 通知HTTP层 → 从runningRequests_释放

3. **PROCESSING → TIMEOUT**：
   - 超时检测机制发现处理时间 > requestTimeout时，状态变为TIMEOUT
   - 通知Scheduler取消处理（如果可能）→ 释放序列ID → 清理KV缓存 → 通知HTTP层 → 从runningRequests_释放

4. **PROCESSING → FAILED**：
   - Inference层处理失败时，状态变为FAILED
   - 释放序列ID → 清理KV缓存 → 通知HTTP层 → 从runningRequests_释放

5. **COMPLETED/TIMEOUT/FAILED → 释放**：
   - 从runningRequests_释放请求
   - 触发从HttpRequestQueue取出下一个请求加入runningRequests_

### 超时检测机制

#### 检测位置
超时检测在**Scheduler层**实现，通过定期扫描`runningRequests_`中的请求。

#### 检测机制

```
┌─────────────────────────────────────────────────────────────────┐
│                   超时检测机制（Scheduler层）                      │
│                                                                  │
│  1. Scheduler维护一个定时器（周期：1秒或更短）                    │
│     │                                                            │
│     ▼                                                            │
│  2. 定时器触发时，扫描runningRequests_中所有请求                  │
│     │                                                            │
│     ▼                                                            │
│  3. 对于状态为PROCESSING的请求：                                  │
│     ├─ 检查处理时间 = 当前时间 - 请求加入runningRequests_的时间   │
│     │                                                            │
│     ├─ 如果处理时间 > requestTimeout (60s)：                     │
│     │   ├─ 将请求状态标记为TIMEOUT                               │
│     │   ├─ 尝试取消Inference层的处理（如果可能）                 │
│     │   ├─ 释放序列ID（如果已分配）                              │
│     │   ├─ 清理KV缓存（如果已分配）                              │
│     │   ├─ 通知HTTP层返回HTTP 408 (Request Timeout)              │
│     │   └─ 从runningRequests_释放该请求                          │
│     │                                                            │
│     └─ 否则，继续等待                                            │
│                                                                  │
│  4. 触发从RequestQueue取出下一个请求加入runningRequests_      │
└─────────────────────────────────────────────────────────────────┘
```

#### 检测频率

- **默认检测周期**：1秒
- **建议范围**：0.5秒 - 5秒
- **权衡**：
  - 检测周期过短：CPU开销大
  - 检测周期过长：超时检测不够及时（最多延迟1个周期）

#### 超时处理流程

```
请求状态：PROCESSING
    │
    ├─ [超时检测] Scheduler检测到处理时间 > requestTimeout
    │
    ▼
1. 标记状态为 TIMEOUT（在runningRequests_中）
    │
2. 尝试取消Inference层的处理（如果可能）
    │
3. 释放序列ID（通过Sequence ID Manager）
    │
4. 清理KV缓存（通过KV Cache Manager）
    │
5. 通知HTTP层（通过回调机制）
    │
6. 从runningRequests_释放请求
    │
7. 触发从RequestQueue取出下一个请求
```

#### 超时检测的关键点

1. **检测范围**：只检测状态为PROCESSING的请求
2. **时间记录**：每个请求加入runningRequests_时记录时间戳（startTime）
3. **取消机制**：超时后尝试取消Inference层的处理（如果可能）
4. **资源清理**：超时后立即释放序列ID、KV缓存等资源
5. **响应返回**：超时后通过回调通知HTTP层返回HTTP 408，不再等待处理结果
6. **统一管理**：所有超时检测和资源清理在Scheduler层统一管理，避免多套系统

### 关键设计点

- **统一状态管理**：所有请求状态在Scheduler的`runningRequests_`中统一管理，移除ConcurrentQueue，避免多套状态系统
- **两级队列设计**：RequestQueue(256) → Scheduler.runningRequests_(8)，实现更细粒度的流量控制
- **HTTP层请求处理**：HTTP层直接调用`Scheduler::addRequest()`，请求进入Scheduler的`RequestQueue`（当前实现）
  - **未来优化**：可在HTTP层通过`Scheduler::getRunningCount()`检查并发数，超过maxConcurrentRequests时直接返回429
- **requestId绑定**：使用`size_t`类型的requestId作为请求标识，贯穿整个处理流程
- **状态机管理**：明确的状态转换（PENDING → PROCESSING → COMPLETED/TIMEOUT/FAILED），确保请求状态清晰可追踪
- **主动超时检测**：Scheduler层定期检测PROCESSING状态的请求，超时立即处理，避免请求长时间挂起
- **序列ID统一管理**：序列ID在Inference层统一管理（基于requestId绑定），序列ID是llama.cpp内部使用的临时标识，KV缓存基于requestId管理（不直接对应序列ID）
- **响应回调机制**：请求完成/超时/失败后，Scheduler通过回调通知HTTP层返回响应

## 配置说明

### HTTP层配置

**maxHttpQueueSize**: 256（实际对应Scheduler的RequestQueue）
  - **说明**：Scheduler的RequestQueue最大长度
  - **队列名称**：RequestQueue（`requestQueue_`，位于Scheduler层）
  - **作用**：限制等待处理的请求数量，防止内存无限增长，超出此数量，就直接返回429，让外部知道当前系统负载过重，不再发送请求
  - **关系**：通常远大于maxConcurrentRequests

**maxConcurrentRequests**: 8
  - **说明**：HTTP Server最大并发请求数
  - **实现方式**：通过Scheduler.runningRequests_实现（max: 8）
  - **作用**：限制Scheduler层同时处理的活跃请求数量，超出此数量，Request就在RequestQueue中等待
  - **数据来源**：Scheduler的`RequestQueue`（`requestQueue_`）
  - **HTTP层检查**：
    - 当前实现中，HTTP层不检查并发数，直接调用`Scheduler::addRequest()`加入`RequestQueue`
    - 未来优化：可在HTTP层通过`Scheduler::getRunningCount()`检查并发数
      - 如果runningCount >= maxConcurrentRequests → 直接返回429（无需加入队列）
      - 否则 → 调用`Scheduler::addRequest()`加入`RequestQueue`等待处理
  - **请求处理流程**：
    1. HTTP层调用`Scheduler::addRequest()`，请求进入Scheduler的`RequestQueue`
    2. Scheduler从`RequestQueue`取出请求
    3. 检查runningRequests_.size() < maxConcurrentRequests
       - 如果满了 → 等待，稍后重试
       - 否则 → 加入runningRequests_，开始处理
    4. 请求完成/超时/失败后，从runningRequests_释放
    5. 触发从`RequestQueue`取出下一个请求

**requestTimeout**: 60s
  - **说明**：请求超时时间
  - **检测位置**：Scheduler层（定期扫描runningRequests_）
  - **检测方式**：检查请求的处理时间（当前时间 - startTime）
  - **超时处理**：如果处理时间 > requestTimeout
    - 标记状态为TIMEOUT
    - 释放序列ID和KV缓存
    - 通知HTTP层返回HTTP 408 (Request Timeout)
    - 从runningRequests_释放请求

**requestId**: size_t（请求ID类型）
  - **说明**：请求的唯一标识符（`size_t`类型，由Scheduler分配）
  - **作用**：用于跟踪和管理每个请求的状态和资源
  - **注意**：当前实现中requestId由Scheduler的`RequestTracker`自动生成，不是UUIDv4格式

### 调度器配置

**maxBatchSize**: 4
  - **说明**：最大批处理大小，一次批处理中允许的最大请求数
  - **约束**：maxBatchSize ≤ maxConcurrentRequests，因为批处理应该小于最大的并发数
  - **数据来源**：runningRequests_（从状态为PENDING的请求中获取）
  - **批处理形成**：Scheduler从runningRequests_中获取状态为PENDING的请求，形成批处理
  - **批处理状态**：批处理中的所有请求标记为PROCESSING
  - **部分完成**：支持批处理中部分请求完成、部分还在处理的场景

**maxInputContextLength**: 128
  - **说明**：最大输入上下文长度
  - **作用**：限制单个请求的输入上下文长度，控制内存使用

**maxOutContextLength**: 2048
  - **说明**：最大输出上下文长度
  - **作用**：限制单个请求的输出上下文长度，控制内存使用



### KV缓存配置

**maxKVCachesItems**: 4*1024*1024
  - **说明**：最大KV缓存条目总数（不是固定的槽位数）
  - **含义**：所有requestId的KV缓存条目总数限制（4*1024*1024 = 4,194,304条目）
  - **条目计算**：每个requestId的KV缓存条目数 = 该请求的序列长度（token数量）
  - **示例**：
    - 请求A：序列长度128 → 占用128个KV缓存条目
    - 请求B：序列长度256 → 占用256个KV缓存条目
    - 总共：128 + 256 = 384个条目（不超过maxKVCachesItems）
  - **作用**：控制KV缓存条目总数，防止内存无限增长
  - **淘汰触发**：当缓存条目总数 > `maxKVCachesItems × kvCacheEvictionThreshold` 时，触发LRU淘汰

**kvCacheMaxMemoryMb**: 1024
  - **说明**：KV缓存最大内存限制（MB）
  - **单位**：MB
  - **作用**：硬性内存限制，防止KV缓存耗尽系统内存
  - **计算**：约等于 `maxKVCaches × avg_sequence_length × model_size_per_token`
  - **淘汰触发**：当内存使用超过此限制时，触发LRU淘汰

**kvCacheEvictionPolicy**: LRU
  - **说明**：KV缓存淘汰策略
  - **可选值**：`LRU`（Least Recently Used，最久未使用）
  - **LRU策略**：淘汰最久未被访问的KV缓存
  - **作用**：当KV缓存数量或内存超过限制时，自动淘汰不常用的缓存

**kvCacheEvictionThreshold**: 0.8
  - **说明**：KV缓存淘汰触发阈值
  - **范围**：0.0 - 1.0
  - **示例**：0.8表示使用率达到80%时开始淘汰
  - **作用**：提前淘汰，避免缓存满时强制淘汰导致的延迟

**kvCacheCleanupInterval**: 1
  - **说明**：KV缓存清理检查间隔（秒）
  - **作用**：定期检查KV缓存是否需要清理的间隔时间，定期清理长时间未使用的KV缓存，释放内存
  - **注意**：过短的间隔会增加CPU开销，过长的间隔可能导致内存不及时释放


### 序列ID管理配置（Inference层）

**管理器位置**：Inference层（LlamaCppBackend）

**序列ID池大小**：可配置（默认值：llama.cpp为1，本项目已实现从配置读取）
  - **llama.cpp默认值**：`n_seq_max = 1`（根据`llama_context_default_params()`定义，源码位置：`llama-context.cpp:2901`）
    - llama.cpp的`llama_context_default_params()`返回的默认`n_seq_max`为1
    - 1表示默认不并行处理多个序列（单序列处理模式）
  - **llama.cpp最大值上限**：`LLAMA_MAX_SEQ = 256`（源码位置：`llama-cparams.h:7`）
    - `n_seq_max`不能超过`LLAMA_MAX_SEQ`（256），否则会抛出异常
  - **取值范围**：
    - 最小值：1（至少支持一个并发序列）
    - 最大值：256（`LLAMA_MAX_SEQ`）
    - 配置默认值：1（与llama.cpp默认值一致）
  - **配置路径**：`backend.llama_cpp.n_seq_max`（配置文件）
  - **当前状态**：已实现从配置读取，通过`Config::instance().backendLlamaCppNSeqMax()`获取
  - **设计目标**：应该可以通过配置文件设置，建议值等于`http.max_concurrent_requests`或略小
  - **关系说明**：
    - `n_seq_max`（序列ID池大小）= `backend.llama_cpp.n_seq_max`
    - `maxConcurrentRequests`（最大并发请求数）= `http.max_concurrent_requests`
    - 建议：`n_seq_max` ≤ `maxConcurrentRequests`（确保有足够的序列ID支持并发请求）
  - **为什么需要映射到0到n_seq_max-1**：这是llama.cpp API的限制，不是我们的设计选择
    - llama.cpp的`llama_seq_id`类型要求是整数，范围必须在`n_seq_max`以内（0到n_seq_max-1）
    - llama.cpp内部使用`seq_id`来索引KV cache的slot，用于管理并行序列的KV缓存
    - 调用`llama_decode`和`llama_memory_seq_rm`时，必须传入有效的`llama_seq_id`（0到n_seq_max-1范围）
  - **映射策略**：requestId（size_t类型）不能直接作为`llama_seq_id`使用
    - 使用动态分配：为每个requestId动态分配一个可用的`seq_id`（范围：0到n_seq_max-1）
    - 请求完成时释放：释放`seq_id`，使其可以被其他requestId重用
    - 不是固定槽位：`seq_id`是可重用的临时标识，不是固定分配给特定requestId的槽位
  - **实现方式**：requestId → seqId动态映射（`std::map<size_t requestId, int32_t seqId>`）
    - 分配时：从可用池中选取一个seq_id（范围：0到n_seq_max-1），建立requestId → seqId映射
    - 使用中：通过requestId查找对应的seq_id，传递给llama.cpp API
    - 释放时：删除映射，seq_id返回可用池供其他requestId使用
  - **限制说明**：
    - 同时最多n_seq_max个requestId可以拥有seq_id
    - 如果所有seq_id都被占用，新请求必须在runningRequests_中等待（PENDING状态）
    - 请求完成/超时/失败时，自动释放seq_id，触发下一个等待请求的分配

**序列ID与KV缓存的关系说明**：
  - **序列ID**：llama.cpp内部使用的标识符（范围：0到n_seq_max-1），用于并行推理
  - **KV缓存**：基于requestId管理，每个requestId对应一个完整的KV缓存
  - **关系**：序列ID是推理时的临时标识，KV缓存是持久化的请求数据
  - **分配**：分配序列ID时，不直接分配KV缓存；推理时，基于requestId存储KV缓存
  - **释放**：释放序列ID时，不直接释放KV缓存；请求完成时，基于requestId清理KV缓存

### llama.cpp后端配置（相关参数）

**nBackendBatch**: 512
  - **说明**：llama.cpp批处理大小
  - **作用**：一次处理的最大token数，控制llama.cpp内部批处理的效率

**nSeqMax**: 可配置（llama.cpp默认值：1，配置文件路径：`backend.llama_cpp.n_seq_max`）
  - **说明**：llama.cpp最大序列数（建议等于或略小于maxConcurrentRequests）
  - **llama.cpp源码定义**：
    - 默认值：1（`llama_context_default_params()`，源码：`llama-context.cpp:2901`）
    - 最大值上限：256（`LLAMA_MAX_SEQ`，源码：`llama-cparams.h:7`）
    - 验证规则：`1 ≤ n_seq_max ≤ LLAMA_MAX_SEQ`（源码：`llama-context.cpp:35-38`）
  - **作用**：限制llama.cpp内部可同时处理的序列数量，影响KV cache的分配
    - 每个序列最多可以使用 `n_ctx / n_seq_max` 的tokens
    - 例如：n_ctx=2048, n_seq_max=8，每个序列最多256个tokens
  - **当前状态**：已实现从配置读取，通过`Config::instance().backendLlamaCppNSeqMax()`获取（`src/inference/llama_cpp_backend.cpp`）
    - 配置读取：代码自动从`config.yaml`的`backend.llama_cpp.n_seq_max`读取，默认值1，范围1-256
  - **配置说明**：通过`backend.llama_cpp.n_seq_max`配置，建议值：
    - llama.cpp默认值：1（单序列处理）
    - 本项目建议值：8（支持8个并发请求）
    - 建议范围：1-256（根据硬件资源和并发需求调整，不能超过`LLAMA_MAX_SEQ`）
    - 建议值：等于或略小于`http.max_concurrent_requests`
  - **关系**：序列ID池大小应该等于此值

## 组件详细设计

### 1. 序列ID管理器（Sequence ID Manager）

**位置**：Inference层（LlamaCppBackend）

**职责**：
- 维护序列ID池（范围：0到n_seq_max-1，本项目n_seq_max=8时为0-7）
- 为每个请求分配唯一的序列ID（基于requestID绑定）
- 请求完成时回收序列ID
- 限制同时使用的序列ID数量（等于n_seq_max，本项目设置为8）

**实现要点**：
- **为什么需要映射**：llama.cpp的`llama_seq_id`必须是整数（范围：0到n_seq_max-1），而requestID是UUIDv4字符串或大整数，无法直接使用
  - 本项目n_seq_max=8时，seq_id范围为0-7
  - `llama_decode`和`llama_memory_seq_rm`等API要求`llama_seq_id`必须在n_seq_max范围内
  - llama.cpp内部使用seq_id来索引KV cache的slot，用于管理并行序列
- **映射数据结构**：使用`std::map<size_t requestId, int32_t seqId>`维护requestID到seqId的动态映射
- **可用池管理**：使用`std::vector<int32_t> availableSeqIds`或位图维护可用序列ID（范围：0到n_seq_max-1，本项目n_seq_max=8时为0-7）
- **分配逻辑**：
  - 从可用池中选取一个seq_id（如0），建立requestID → seqId映射
  - 如果池为空（n_seq_max个seq_id都被占用），请求保持在PENDING状态等待
- **释放逻辑**：
  - 删除requestID → seqId映射
  - 将seq_id返回可用池（使其可以被其他requestID重用）
  - 调用`llama_memory_seq_rm`清理llama.cpp内部的KV cache（基于seq_id）
  - KV缓存基于requestID独立管理，不依赖于seq_id的释放

### 2. KV缓存管理器（KV Cache Manager）

**位置**：Inference层（LlamaCppBackend）

**职责**：
- 基于requestID管理KV缓存（不是基于序列ID）
- 跟踪每个requestID的KV缓存使用情况
- 实现LRU淘汰策略（按requestID淘汰）
- 当KV缓存条目总数超过限制时，淘汰最久未使用的缓存
- 主动清理已完成的请求的KV缓存

**关键设计**：
- **基于requestID**：每个requestID对应一个完整的KV缓存
- **条目数量限制**：maxKVCachesItems (4*1024*1024) 是**缓存条目总数**，不是固定槽位数
  - 每个requestID的KV缓存可能包含多个条目（取决于序列长度）
  - 例如：一个requestID的序列长度为128，则该请求的KV缓存包含128个条目
- **内存限制**：kvCacheMaxMemoryMb (1024MB) 是所有KV缓存的总内存限制
- **淘汰策略**：按requestID淘汰，LRU策略，淘汰整个请求的KV缓存

**实现要点**：
- 使用`std::map<size_t requestId, KVCacheEntry>`维护requestId到KV缓存的映射
- 每个KVCacheEntry包含该requestId的所有token的KV缓存数据
- **淘汰保护**：只淘汰状态为PENDING或COMPLETED的请求对应的KV缓存
- 正在处理的请求（状态为PROCESSING）对应的KV缓存不会被淘汰
- 淘汰触发条件：
  - **条目数限制**：当前缓存条目总数 > `maxKVCachesItems × kvCacheEvictionThreshold`
  - **内存限制**：内存使用 > `kvCacheMaxMemoryMb × kvCacheEvictionThreshold`
- **LRU淘汰**：按照requestId的最后访问时间排序，淘汰最久未访问的requestId的整个KV缓存

### 3. 响应回调机制

**响应路径**：Inference层 → Scheduler层 → HTTP层

**回调机制**：
```
请求完成/超时/失败
    │
    ▼
Scheduler层（runningRequests_）
    ├─ 更新请求状态（COMPLETED/TIMEOUT/FAILED）
    ├─ 释放序列ID和KV缓存
    └─ 通知HTTP层（通过回调函数或事件通知）
        │
        ▼
HTTP层
    └─ 返回HTTP Response（200/408/500）
```

**实现要点**：
- Scheduler维护HTTP层回调函数（`std::function<void(RequestState&)>`）
- 请求完成/超时/失败时，调用回调函数通知HTTP层
- HTTP层根据请求状态返回相应的HTTP状态码

### 4. 错误处理机制

**错误类型**：
1. **Inference处理失败**：
   - 状态：PROCESSING → FAILED
   - 处理：释放序列ID → 清理KV缓存 → 通知HTTP层 → 返回HTTP 500

2. **HTTP连接断开**：
   - 处理：取消对应请求的处理 → 释放序列ID → 清理KV缓存 → 从runningRequests_释放

3. **资源不足**：
   - 序列ID用尽：请求保持在PENDING状态，等待序列ID可用
   - KV缓存满：触发LRU淘汰，为新请求腾出空间

4. **超时处理**：
   - 状态：PROCESSING → TIMEOUT
   - 处理：尝试取消Inference处理 → 释放序列ID → 清理KV缓存 → 通知HTTP层 → 返回HTTP 408

### 5. 请求完成后的自动流转

**触发机制**：请求从runningRequests_释放后，立即触发下一个请求处理

**实现流程**：
```
请求完成/超时/失败
    │
    ▼
从runningRequests_释放
    │
    ▼
检查RequestQueue是否为空
    ├─ 如果为空 → 结束
    └─ 如果不为空 → 检查runningRequests_.size() < maxConcurrentRequests
        ├─ 如果满了 → 等待，稍后重试
        └─ 如果未满 → 从RequestQueue取出下一个请求加入runningRequests_
```

**实现位置**：Scheduler层（在processBatch完成后自动触发）

### 6. KV缓存淘汰的时机和影响

**淘汰时机**：
- **定期清理**：每`kvCacheCleanupInterval`秒检查一次
- **主动淘汰**：当缓存使用率超过`kvCacheEvictionThreshold`时触发

**淘汰策略**：
- **基于requestID淘汰**：按requestID淘汰整个请求的KV缓存（不是按序列ID）
- **只淘汰未使用的缓存**：只淘汰状态为PENDING或COMPLETED的请求对应的KV缓存
- **保护正在使用的缓存**：状态为PROCESSING的请求对应的KV缓存不会被淘汰
- **LRU策略**：按照requestID的最后访问时间排序，淘汰最久未被访问的requestID的整个KV缓存

**淘汰流程**：
```
触发淘汰（缓存条目总数或内存 > threshold）
    │
    ▼
扫描KV缓存管理器中的所有requestID
    │
    ├─ 跳过状态为PROCESSING的requestID对应的缓存（保护）
    └─ 只考虑状态为PENDING或COMPLETED的requestID对应的缓存
        │
        ▼
按照LRU策略排序（按requestID的最后访问时间，最久未使用优先）
    │
    ▼
淘汰最久未使用的requestID的整个KV缓存
    │
    ├─ 计算该requestID的KV缓存条目数（序列长度）
    ├─ 从缓存条目总数中减去该条目数
    └─ 从内存使用中减去该请求的内存占用
        │
        ▼
继续淘汰（直到条目总数或内存 < threshold）
    │
    ▼
释放对应的序列ID（如果请求已完成且不再需要）
```

**关键点**：
- 正在处理的请求的KV缓存不会被淘汰，保证处理连续性
- 只淘汰已完成或等待中的请求对应的缓存，避免影响正在处理的请求

## 架构优势总结

### 采用方案B（移除ConcurrentQueue）的优势

1. **架构更清晰**：
   - HTTP层：只负责接收和排队（HttpRequestQueue）
   - Scheduler层：统一管理并发控制、批处理、资源管理（runningRequests_）
   - Inference层：序列ID管理、KV缓存管理

2. **性能更优**：
   - 减少50%的队列操作（4次 → 2次）
   - 减少75%的锁操作（4次 → 1次）
   - 降低内存占用

3. **复杂度更低**：
   - 单一数据源（runningRequests_），避免多套状态系统
   - 统一状态管理，减少状态不一致风险
   - 代码更简洁，易于维护

4. **易于扩展**：
   - 统一管理更容易添加新功能（如优先级、资源限制等）
   - 序列ID和KV缓存统一管理，便于优化

---

## 相关文档

- [序列ID优化策略分析](../analysis/sequence_id_optimization_strategies.md)
- [调度器卡住问题分析](../analysis/scheduler_deadlock_complete_analysis.md)
- [ConcurrentQueue必要性分析](./concurrent_queue_necessity_analysis.md)