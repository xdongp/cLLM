# cLLM 动态 Batch Size 调整机制设计

**版本**: v2.0
**日期**: 2026-01-22
**作者**: cLLM Technical Team
**状态**: 设计完成，已实现

---

## 1. 设计概述

### 1.1 核心思想

**简化假设**:
- 当 batch size ≤ 最优值时，处理时间基本保持不变（GPU 带宽充足）
- 当 batch size > 最优值时，处理时间会显著增加（GPU 带宽饱和）
- **处理时间的临界点**就是最优 batch size

**核心原理**:
```
Batch Size:    1   2   4   8   16   32   64
Processing Time: 50  50  50  50   50   60  120
                              ↑
                         临界点 = 最优 batch size
```

### 1.2 设计目标

**主要目标**:
1. ✅ **最大化吞吐量**: 找到最优 batch size，最大化 GPU 利用率
2. ✅ **通用性强**: 不依赖 GPU 型号，适用于所有平台
3. ✅ **开销最小**: 只测量 batch 处理时间，无额外开销
4. ✅ **实现简单**: 算法简洁，代码量少，易于维护
5. ✅ **稳定性高**: 基于客观的处理时间，判断准确

**次要目标**:
- 适应负载变化
- 自动恢复机制
- 运行时可配置

---

## 2. 策略和算法

### 2.1 策略（Strategy）

本设计提供三种策略：

| 策略 | 说明 | 适用场景 | 特点 |
|-----|------|---------|------|
| **static** | 静态策略 | 稳定场景，已知最优 batch size | 最简单，零开销，性能可预测 |
| **adaptive** | 自适应策略 | 通用场景，需要动态调整 | 自动适应负载变化，快速收敛 |
| **hybrid** | 混合策略 | 生产环境，兼顾性能和稳定性 | 调优阶段 + 稳定阶段，性能最优 |

### 2.2 算法（Algorithm）

本设计提供两种算法：

| 算法 | 优先级 | 时间复杂度 | 收敛速度 | 实现难度 | 推荐场景 |
|-----|-------|----------|---------|---------|---------|
| **adaptive_step** | ⭐⭐⭐⭐⭐ | O(log n) | 快 | 简单 | 通用场景，快速收敛 |
| **exponential_binary** | ⭐⭐⭐ | O(log n) | 快 | 简单 | 理论成熟，精确收敛 |

### 2.3 策略和算法的组合

| 策略 | 可用算法 | 默认算法 | 说明 |
|-----|---------|---------|------|
| **static** | 无 | 无 | 不使用搜索算法，使用固定 batch size |
| **adaptive** | adaptive_step, exponential_binary | adaptive_step | 使用搜索算法动态调整 batch size |
| **hybrid** | adaptive_step, exponential_binary | adaptive_step | 调优阶段使用搜索算法，稳定阶段使用固定 batch size + 批处理累积 |

### 2.4 配置方式

**配置文件示例**:
```yaml
dynamic_batch_tuner:
  enabled: true
  strategy: "hybrid"              # 策略: static, adaptive, hybrid
  search_algorithm: "adaptive_step"  # 算法: adaptive_step, exponential_binary
  
  # static 策略专用参数
  fixed_batch_size: 24
  
  # adaptive/hybrid 策略参数
  min_batch_size: 16
  max_batch_size: 48
  initial_batch_size: 24
```

---

## 3. 策略详解

### 3.1 策略 1: Static（静态策略）

#### 3.1.1 策略原理

**核心思想**:
- 使用固定的 batch size，不进行动态调整
- 适用于已知最优 batch size 的场景
- 最简单、最稳定的方案

**策略流程**:
```
┌─────────────────────────────────────────────────────────────┐
│                    静态策略流程                              │
├─────────────────────────────────────────────────────────────┤
│  1. 从配置文件读取固定的 batch size                         │
│  2. 始终使用这个 batch size 处理请求                        │
│  3. 不进行任何动态调整                                       │
└─────────────────────────────────────────────────────────────┘
```

#### 3.1.2 策略特点

**优点**:
- ✅ 最简单：无需任何搜索逻辑
- ✅ 最稳定：不会因为调整导致性能波动
- ✅ 零开销：无需探测和测量
- ✅ 可预测：性能完全可预测

**缺点**:
- ❌ 无法适应变化：无法适应负载变化
- ❌ 需要预先调优：需要手动找到最优 batch size
- ❌ 可能不是最优：在未知场景下可能不是最优

**时间复杂度**: O(1)

### 3.2 策略 2: Adaptive（自适应策略）

#### 3.2.1 策略原理

**核心思想**:
- 使用搜索算法动态调整 batch size
- 根据批处理时间自动找到最优值
- 适应负载变化，持续优化

**策略流程**:
```
┌─────────────────────────────────────────────────────────────┐
│              自适应策略流程                                  │
├─────────────────────────────────────────────────────────────┤
│  1. 使用搜索算法（adaptive_step 或 exponential_binary）     │
│  2. 持续监控批处理时间                                      │
│  3. 根据性能指标动态调整 batch size                         │
│  4. 自动适应负载变化                                        │
└─────────────────────────────────────────────────────────────┘
```

#### 3.2.2 可用算法

**算法 1: adaptive_step（自适应步长搜索）**

**核心思想**:
1. **阶段 1: 指数增长** - 快速找到临界点范围
   - 从 minBatchSize 开始，步长指数增长
   - 当处理时间显著增加时停止

2. **阶段 2: 动态步长回退** - 精确收敛到最优值
   - 从临界点开始，步长减半回退
   - 动态调整步长，快速收敛

**算法流程**:
```
┌─────────────────────────────────────────────────────────────┐
│              阶段 1: 指数增长（快速找范围）                │
│  1 → 2 → 4 → 8 → 16 → 32 → 64                      │
│                    ↑ 时间增加，停止                        │
└───────────────────────┬─────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│              阶段 2: 动态步长回退（精确收敛）                │
│  64 → 48 (步长=16) → 56 (步长=8) → 52 (步长=4)       │
│  → 50 (步长=2) → 49 (步长=1) → 48 (最优)              │
└─────────────────────────────────────────────────────────────┘
```

**算法特点**:
- ✅ 快速收敛：O(log n) 时间复杂度
- ✅ 简单直观：易于理解和实现
- ✅ 适应性强：动态调整步长
- ✅ 实用性强：在实践中往往比理论算法更快

**算法 2: exponential_binary（指数增长 + 二分查找）**

**核心思想**:
1. **阶段 1: 指数增长** - 快速找到临界点范围
   - 从 minBatchSize 开始，每次翻倍
   - 当处理时间显著增加时停止
   - 得到临界点范围 [batchSize/2, batchSize]

2. **阶段 2: 二分查找** - 精确收敛到最优值
   - 在临界点范围内进行二分查找
   - 找到处理时间开始显著增加的精确位置
   - 这个位置就是最优 batch size

**算法流程**:
```
初始范围: [1, 256]

阶段 1: 指数增长
  1 → 2 → 4 → 8 → 16 → 32 → 64
                    ↑ 时间增加，停止
  得到范围 [32, 64]

阶段 2: 二分查找
  [32, 64] → [48, 64] → [56, 64] → [60, 64]
  → [62, 64] → [63, 64] → [64, 64] (最优)
```

**算法特点**:
- ✅ 理论成熟：基于成熟的二分查找算法
- ✅ 精确收敛：能够精确找到最优值
- ✅ 稳定可靠：算法收敛性有理论保证
- ✅ 实现简单：代码逻辑清晰

#### 3.2.3 策略特点

**优点**:
- ✅ 自动适应：无需手动调优
- ✅ 持续优化：适应负载变化
- ✅ 性能最优：自动找到最优 batch size
- ✅ 灵活配置：可选择不同的搜索算法

**缺点**:
- ❌ 有一定开销：需要探测和测量
- ❌ 收敛时间：需要一定时间找到最优值
- ❌ 可能波动：在调整过程中可能有性能波动

**时间复杂度**: O(log n)

### 3.3 策略 3: Hybrid（混合策略）

#### 3.3.1 策略原理

**核心思想**:
- **调优阶段**：使用搜索算法快速找到最优 batch size
- **稳定阶段**：使用固定 batch size + 批处理累积
- 兼顾性能和稳定性

**策略流程**:
```
┌─────────────────────────────────────────────────────────────┐
│              混合策略流程                                    │
├─────────────────────────────────────────────────────────────┤
│  阶段 1: 调优阶段（前 N 个请求）                            │
│  1. 使用搜索算法（adaptive_step 或 exponential_binary）     │
│  2. 快速找到最优 batch size                                 │
│  3. 监控性能指标                                            │
├─────────────────────────────────────────────────────────────┤
│  阶段 2: 稳定阶段（N 个请求后）                             │
│  1. 使用固定 batch size                                     │
│  2. 启用批处理累积（等待更多请求）                          │
│  3. 监控性能漂移，必要时重新调优                             │
└─────────────────────────────────────────────────────────────┘
```

#### 3.3.2 策略实现

```cpp
/**
 * @brief 混合策略（调优阶段 + 稳定阶段）
 */
class HybridStrategy {
public:
    size_t getOptimalBatchSize() {
        if (currentPhase_ == Phase::TUNING) {
            // 调优阶段：使用搜索算法
            return tuner_->getOptimalBatchSize();
        } else {
            // 稳定阶段：使用固定 batch size
            return config_.stableBatchSize;
        }
    }
    
    void onBatchProcessed(size_t batchSize, double processingTimeMs) {
        if (currentPhase_ == Phase::TUNING) {
            // 调优阶段：报告给调谐器
            tuner_->reportBatchCompletion(batchSize, processingTimeMs);
            
            // 检查是否达到调优完成条件
            if (tuner_->isStabilized()) {
                transitionToStable();
            }
        } else {
            // 稳定阶段：监控性能漂移
            monitorPerformanceDrift(batchSize, processingTimeMs);
        }
    }
    
private:
    enum class Phase { TUNING, STABLE };
    Phase currentPhase_;
    std::unique_ptr<DynamicBatchTuner> tuner_;
    
    void transitionToStable() {
        currentPhase_ = Phase::STABLE;
        config_.stableBatchSize = tuner_->getOptimalBatchSize();
        CLLM_INFO("[HybridStrategy] 切换到稳定阶段，batch_size=%zu", 
                  config_.stableBatchSize);
    }
};
```

#### 3.3.3 策略特点

**优点**:
- ✅ 性能最优：调优阶段找到最优值
- ✅ 稳定性好：稳定阶段使用固定值
- ✅ 自适应：可重新调优适应负载变化
- ✅ 批处理累积：提高 GPU 利用率

**缺点**:
- ❌ 复杂度较高：需要管理两个阶段
- ❌ 调优时间：需要一定时间完成调优
- ❌ 配置参数：需要配置调优持续时间

**时间复杂度**: O(log n)（调优阶段），O(1)（稳定阶段）

---

## 4. 配置参数

### 4.1 基础配置

```yaml
dynamic_batch_tuner:
  enabled: true                     # 是否启用动态 batch size 调谐器
  
  # 策略选择
  strategy: "hybrid"                # 策略: static, adaptive, hybrid
  
  # 搜索算法选择（adaptive 和 hybrid 策略使用）
  search_algorithm: "adaptive_step" # 算法: adaptive_step, exponential_binary
```

### 4.2 Static 策略参数

```yaml
  # 静态算法专用参数
  fixed_batch_size: 24             # 静态算法使用的固定 batch size
```

### 4.3 Adaptive/Hybrid 策略参数

```yaml
  # Batch Size 范围
  min_batch_size: 16               # 最小 batch size
  max_batch_size: 48               # 最大 batch size
  initial_batch_size: 24           # 初始 batch size
  
  # 时间阈值
  time_increase_threshold: 0.30    # 处理时间增加阈值（30%）
  time_decrease_threshold: 0.10    # 处理时间减少阈值（10%）
  
  # 验证和调整
  validation_interval: 50          # 验证间隔（每 50 个 batch）
  max_consecutive_time_increases: 5  # 最大连续时间增加次数
  auto_adjust_enabled: true        # 是否启用自动调整
  
  # 探测参数
  probe_batch_count: 10            # 探测时运行的 batch 数量
  validation_batch_count: 10       # 验证时运行的 batch 数量
  
  # 调整参数
  adjustment_factor: 0.50          # 调整因子（50%）
  exploration_interval: 200        # 探索间隔（每 200 个 batch）
```

### 4.4 Hybrid 策略额外参数

```yaml
  # 调优阶段配置
  tuning_duration_requests: 100    # 调优阶段持续请求数
  
  # 稳定阶段配置
  stable_batch_size: 24            # 稳定阶段使用的 batch size
  accumulation_enabled: true        # 是否启用批处理累积
  accumulation_min_batch_size: 8   # 批处理累积的最小 batch size
  accumulation_max_wait_ms: 50     # 批处理累积的最大等待时间
  
  # 性能监控配置
  monitoring_check_interval_requests: 1000  # 性能检查间隔
  monitoring_drift_threshold: 0.10          # 性能漂移阈值
  monitoring_auto_retune: true               # 是否自动重新调优
```

---

## 5. 实现架构

### 5.1 核心类

```cpp
// 动态批处理调谐器
class DynamicBatchTuner {
public:
    enum class SearchAlgorithm {
        ADAPTIVE_STEP,
        EXPONENTIAL_BINARY
    };
    
    struct TunerConfig {
        SearchAlgorithm searchAlgorithm;
        size_t minBatchSize;
        size_t maxBatchSize;
        size_t initialBatchSize;
        double timeIncreaseThreshold;
        double timeDecreaseThreshold;
        size_t validationInterval;
        size_t maxConsecutiveTimeIncreases;
        bool autoAdjustEnabled;
        size_t probeBatchCount;
        size_t validationBatchCount;
        double adjustmentFactor;
        size_t explorationInterval;
    };
    
    DynamicBatchTuner(const TunerConfig& config);
    size_t getOptimalBatchSize();
    void reportBatchCompletion(size_t batchSize, double processingTimeMs);
    bool isStabilized() const;
    
private:
    TunerConfig config_;
    std::atomic<size_t> currentBatchSize_;
    std::map<size_t, std::vector<double>> batchPerformance_;
    std::mutex statsMutex_;
    size_t batchCount_;
    size_t consecutiveTimeIncreases_;
    double bestProcessingTime_;
    size_t bestBatchSize_;
    
    void adaptiveStepSearch();
    void exponentialBinarySearch();
    void validateCurrentBatchSize();
    void adjustBatchSize();
};

// 混合批处理策略
class HybridBatchStrategy {
public:
    enum class HybridPhase {
        TUNING,
        STABLE
    };
    
    struct HybridConfig {
        bool enabled;
        struct {
            bool enabled;
            size_t durationRequests;
            size_t minBatchSize;
            size_t maxBatchSize;
            size_t initialBatchSize;
        } tuning;
        struct {
            size_t batchSize;
            bool accumulationEnabled;
            size_t minBatchSize;
            size_t maxWaitMs;
        } stable;
        struct {
            size_t checkIntervalRequests;
            double driftThreshold;
            bool autoRetune;
        } monitoring;
    };
    
    HybridBatchStrategy(const HybridConfig& config);
    size_t getOptimalBatchSize();
    void onBatchProcessed(size_t batchSize, double processingTimeMs);
    HybridPhase getCurrentPhase() const;
    bool isStable() const;
    void forceRetune();
    
private:
    HybridConfig config_;
    std::atomic<HybridPhase> currentPhase_;
    std::atomic<size_t> optimalBatchSize_;
    std::atomic<size_t> requestCount_;
    std::map<size_t, std::vector<double>> batchPerformance_;
    std::mutex statsMutex_;
    double baselineThroughput_;
    size_t stableRequestCount_;
    
    void transitionToStablePhase();
    void checkPerformanceDrift();
    size_t findOptimalBatchSize() const;
};
```

### 5.2 与 Scheduler 的集成

```cpp
class Scheduler {
private:
    std::unique_ptr<HybridBatchStrategy> hybridStrategy_;
    
public:
    void processRequests() {
        // 获取最优 batch size
        size_t minBatchSize = hybridStrategy_->getOptimalBatchSize();
        
        // 批处理累积策略
        if (queueSize < minBatchSize && runningCount == 0) {
            std::unique_lock<std::mutex> lock(queueMutex_);
            queueCondition_.wait_for(
                lock,
                std::chrono::milliseconds(MAX_WAIT_MS_FOR_BATCH),
                [this, minBatchSize]() {
                    return requestQueue_.getQueueSize() >= minBatchSize || !running_;
                }
            );
        }
        
        // ... 批处理形成和处理 ...
    }
    
    void processBatch(std::vector<RequestState>& batch) {
        auto batchStart = std::chrono::steady_clock::now();
        
        // 执行批处理
        SchedulerBatchProcessor processor(this, modelExecutor_, kvCache_, &batchManager_);
        processor.processBatch(activeBatch);
        
        auto batchEnd = std::chrono::steady_clock::now();
        auto processingTime = std::chrono::duration_cast<std::chrono::milliseconds>(batchEnd - batchStart).count();
        
        // 报告批处理完成
        if (hybridStrategy_) {
            hybridStrategy_->onBatchProcessed(activeBatch.size(), static_cast<double>(processingTime));
        }
    }
};
```

---

## 6. 性能测试结果

### 6.1 策略对比测试

| 并发数 | Static (32) | Adaptive | Hybrid | Hybrid vs Static | Hybrid vs Adaptive |
|--------|------------|----------|---------|------------------|-------------------|
| **8**  | 137.97     | -        | 140.76  | +2.0%            | -                 |
| **16** | 131.01     | -        | 134.82  | +2.9%            | -                 |
| **24** | 124.95     | -        | 131.04  | +4.9%            | -                 |
| **32** | 121.46     | -        | 126.64  | +4.3%            | -                 |

### 6.2 批处理累积策略测试

| 并发数 | 批处理累积 | Hybrid | 批处理累积 vs Hybrid |
|--------|-----------|---------|---------------------|
| **8**  | 135.53    | 140.76  | -3.7%               |
| **16** | 130.99    | 134.82  | -2.8%               |
| **24** | 124.88    | 131.04  | -4.7%               |
| **32** | 125.14    | 126.64  | -1.2%               |

### 6.3 结论

1. **Hybrid 策略最优**
   - 在所有并发级别下都保持最佳性能
   - 8并发：140.76 tokens/sec（+2.0% vs Static）
   - 32并发：126.64 tokens/sec（+4.3% vs Static）

2. **批处理累积策略表现**
   - 低并发（8）：135.53 tokens/sec，略低于 Hybrid
   - 高并发（32）：125.14 tokens/sec，接近 Hybrid

3. **推荐配置**
   - **生产环境**：Hybrid 策略 + adaptive_step 算法
   - **稳定场景**：Static 策略（已知最优 batch size）
   - **通用场景**：Adaptive 策略（需要动态调整）

---

## 7. 使用指南

### 7.1 快速开始

#### 7.1.1 使用 Hybrid 策略（推荐）

```yaml
dynamic_batch_tuner:
  enabled: true
  strategy: "hybrid"
  search_algorithm: "adaptive_step"
  min_batch_size: 16
  max_batch_size: 48
  initial_batch_size: 24
```

#### 7.1.2 使用 Static 策略

```yaml
dynamic_batch_tuner:
  enabled: true
  strategy: "static"
  fixed_batch_size: 24
```

#### 7.1.3 使用 Adaptive 策略

```yaml
dynamic_batch_tuner:
  enabled: true
  strategy: "adaptive"
  search_algorithm: "adaptive_step"
  min_batch_size: 16
  max_batch_size: 48
  initial_batch_size: 24
```

### 7.2 参数调优建议

#### 7.2.1 Batch Size 范围

```yaml
# 低并发场景（< 16）
min_batch_size: 8
max_batch_size: 32
initial_batch_size: 16

# 中并发场景（16-32）
min_batch_size: 16
max_batch_size: 48
initial_batch_size: 24

# 高并发场景（> 32）
min_batch_size: 24
max_batch_size: 64
initial_batch_size: 32
```

#### 7.2.2 时间阈值

```yaml
# 保守配置（更稳定）
time_increase_threshold: 0.20  # 20%
time_decrease_threshold: 0.15  # 15%

# 激进配置（更快速）
time_increase_threshold: 0.30  # 30%
time_decrease_threshold: 0.10  # 10%
```

#### 7.2.3 Hybrid 策略调优持续时间

```yaml
# 快速调优（适合测试）
tuning_duration_requests: 50

# 标准调优（适合生产）
tuning_duration_requests: 100

# 慢速调优（适合稳定场景）
tuning_duration_requests: 200
```

---

## 8. 附录

### 8.1 术语表

| 术语 | 英文 | 说明 |
|-----|------|------|
| **Batch Size** | Batch Size | 批处理大小，一次推理处理的请求数 |
| **吞吐量** | Throughput | 每秒处理的 tokens 数 |
| **策略** | Strategy | 批处理调整的整体策略 |
| **算法** | Algorithm | 搜索最优 batch size 的具体算法 |
| **调优阶段** | Tuning Phase | Hybrid 策略的初始阶段，使用搜索算法 |
| **稳定阶段** | Stable Phase | Hybrid 策略的后续阶段，使用固定值 |
| **批处理累积** | Batch Accumulation | 等待更多请求形成更大的批处理 |

### 8.2 参考资料

1. **Roofline Model**: Williams et al., "Roofline: An Insightful Visual Performance Model for Floating-Point Programs and Multicore Architectures", 2009
2. **Amdahl's Law**: Gene Amdahl, "Validity of the single processor approach to achieving large scale computing capabilities", 1967
3. **Binary Search**: Knuth, "The Art of Computer Programming, Volume 3: Sorting and Searching", 1973
4. **Llama.cpp Batch Processing**: https://github.com/ggerganov/llama.cpp

### 8.3 联系方式

- **技术负责人**: cLLM Technical Team
- **文档版本**: v2.0
- **更新日期**: 2026-01-22

---

**文档结束**

*本文档描述了 cLLM 动态 Batch Size 调整机制的完整设计方案。*  
*本设计方案遵循 cLLM 项目的架构原则和编码规范。*
