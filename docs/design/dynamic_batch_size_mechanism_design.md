# 动态 Batch Size 调整机制设计文档

## 文档信息
- **文档版本**: v1.0
- **创建日期**: 2026-01-22
- **设计人**: cLLM Technical Team
- **状态**: 设计完成，待实现

---

## 1. 设计原理

### 1.1 核心问题分析

当前系统使用静态的 `MIN_BATCH_SIZE_FOR_ACCUMULATION = 8`，存在以下问题：

1. **GPU 资源利用不足**
   - 不同 GPU 型号（NVIDIA A100、Apple M3、AMD MI250等）的内存和计算能力差异巨大
   - 固定的 batch size 无法充分发挥高端 GPU 的性能
   - 小 batch size 导致 GPU 计算单元空闲，吞吐量下降

2. **批处理累积策略的局限性**
   ```cpp
   // 当前实现（scheduler.cpp:428-430）
   constexpr size_t MIN_BATCH_SIZE_FOR_ACCUMULATION = 8;
   constexpr size_t MAX_WAIT_MS_FOR_BATCH = 50;
   ```
   - 静态阈值无法适应动态负载变化
   - 高峰期可能等待过久，低峰期可能等待不足
   - 未考虑 GPU 内存和计算能力的约束

3. **性能评估指标缺失**
   - 没有量化指标评估当前 batch size 的合理性
   - 无法判断是内存受限还是计算受限
   - 缺乏自适应调整的反馈机制

### 1.2 核心算法设计

#### 1.2.1 性能评估指标体系

**一级指标（核心）**:
- **吞吐量 (Throughput)**: tokens/second（首要优化目标）
- **GPU 利用率 (GPU Utilization)**: 计算单元占用率
- **内存带宽利用率 (Memory Bandwidth Utilization)**: HBM/VRAM 带宽使用情况

**二级指标（辅助）**:
- **批处理时间 (Batch Processing Time)**: 单次推理延迟
- **内存占用 (Memory Footprint)**: 峰值内存使用量
- **计算密度 (Compute Density)**: FLOPs/byte（计算密集型 vs 内存密集型）

**评估公式**:
```
Score = α * Throughput + β * GPU_Utilization + γ * (1 - Memory_Waste)

其中:
- α + β + γ = 1（权重系数）
- Memory_Waste = (Peak_Memory - Used_Memory) / Peak_Memory
- 目标: Score → 最大值
```

#### 1.2.2 简化为三种机制

**static（静态，当前机制）**
- 采用现有调度与批处理策略，不启用调谐器。
- batch size 来自配置或现有启发式，不做动态调整。

**dynamic（动态）**
- 目标：在运行时持续找到更优 batch size。
- 算法：指数增加、二分下降，循环动态调整。
- 过程：
  1. 从 `min_batch_size` 开始指数增长（1 → 2 → 4 → 8 ...），直到性能下降或触达上限。
  2. 在“最后一次性能提升”与“首次下降”之间做二分搜索，确定最优区间内的 batch size。
  3. 进入在线调整阶段：按固定周期评估吞吐/时间指标，若性能下降则回退并缩小，若提升则尝试放大，持续迭代。

**hybird（混合）**
- 目标：先找到最优 batch size，再保持稳定。
- 算法：同 dynamic 的“指数增加 + 二分下降”，但只在启动或手动触发时执行。
- 找到最优值后锁定 batch size，不再进行在线动态调整。

#### 1.2.3 设计简化说明
- 统一“探测逻辑”为指数增加 + 二分下降。
- dynamic 与 hybird 的区别只在于“是否持续在线调整”。
- static 完全沿用当前机制，便于回退与对比。

#### 1.2.4 机制映射说明
- 文档内的“初始探测”对应 dynamic/hybird 共享流程。
- “动态调整”仅在 dynamic 策略启用；hybird 在探测完成后锁定。
- 若配置 `strategy=static`，调谐器完全不接入调度流程。

---

## 2. 实现方案

### 2.1 代码修改位置

#### 2.1.1 新增文件

**文件 1: `include/cllm/scheduler/dynamic_batch_tuner.h`**
```cpp
/**
 * @file dynamic_batch_tuner.h
 * @brief 动态 Batch Size 调谐器（简化版）
 * 
 * 负责:
 * - 指数增加 + 二分下降探测
 * - 动态/混合策略调整
 * - 批处理耗时反馈
 */

namespace cllm {

class DynamicBatchTuner {
private:
    // 状态管理
    enum class TuningPhase {
        INITIAL_PROBING,  // 初始探测阶段
        DYNAMIC_ADJUSTMENT, // 动态调整阶段
        STABLE_RUNNING     // 稳定运行阶段
    };
    
    TuningPhase currentPhase_;
    std::atomic<size_t> currentBatchSize_;
    
    // 性能指标
    struct PerformanceMetrics {
        double throughput;           // tokens/sec
        double gpuUtilization;       // 0-1
        double memoryUtilization;    // 0-1
        double processingTimeMs;     // 批处理时间
        size_t peakMemoryMb;         // 峰值内存
        std::chrono::steady_clock::time_point timestamp;
    };
    
    std::deque<PerformanceMetrics> metricsHistory_;
    std::mutex metricsMutex_;
    
    // 配置参数
    struct TunerConfig {
        size_t minBatchSize;
        size_t maxBatchSize;
        size_t initialBatchSize;
        
        // 探测阶段参数
        size_t maxProbingAttempts;
        double probingGrowthFactor;
        
        // 调整阶段参数
        double performanceThreshold;  // ε
        double growthCoefficient;     // α
        double decayCoefficient;      // β
        double oscillationCoefficient; // δ
        
        // 稳定阶段参数
        size_t stabilizationCycles;   // N
        double batchSizeChangeThreshold; // θ
        double performanceFluctuationThreshold; // φ
        
        // 安全参数
        size_t memorySafetyMarginMb;
        double memoryUsageLimit;
    };
    
    TunerConfig config_;
    
    // GPU 信息
    struct GPUInfo {
        std::string model;
        size_t totalMemoryMb;
        size_t availableMemoryMb;
        size_t computeUnits;
        double memoryBandwidthGbPerSec;
        double theoreticalPeakTflops;
    };
    
    GPUInfo gpuInfo_;
    
    // 历史最佳
    size_t bestBatchSize_;
    double bestScore_;
    
    // 安全机制
    bool isInSafeZone_;
    size_t consecutiveFailures_;
    
public:
    DynamicBatchTuner();
    ~DynamicBatchTuner();
    
    // 初始化
    void initialize(const TunerConfig& config);
    void detectGPU();
    
    // 核心接口
    size_t getOptimalBatchSize();
    void reportBatchCompletion(const PerformanceMetrics& metrics);
    
    // 状态查询
    TuningPhase getCurrentPhase() const { return currentPhase_; }
    size_t getCurrentBatchSize() const { return currentBatchSize_.load(); }
    bool isStabilized() const { return currentPhase_ == TuningPhase::STABLE_RUNNING; }
    
    // 配置接口
    void updateConfig(const TunerConfig& config);
    TunerConfig getConfig() const { return config_; }
    
    // 调试接口
    std::string getStatusReport() const;
    
private:
    // 阶段实现
    void runInitialProbing();
    void runDynamicAdjustment(const PerformanceMetrics& metrics);
    void runStableRunning(const PerformanceMetrics& metrics);
    
    // 辅助方法
    double calculateScore(const PerformanceMetrics& metrics) const;
    bool shouldIncreaseBatchSize(const PerformanceMetrics& metrics) const;
    bool shouldDecreaseBatchSize(const PerformanceMetrics& metrics) const;
    bool checkStabilizationCondition() const;
    
    // 安全检查
    bool isMemorySafe(size_t batchSize) const;
    void handleMemoryOverflow();
    void resetToSafeState();
    
    // 工具方法
    size_t estimateBatchSizeFromMemory(size_t availableMemoryMb) const;
    size_t estimateBatchSizeFromCompute() const;
};

} // namespace cllm
```

**文件 2: `src/scheduler/dynamic_batch_tuner.cpp`**
- 实现调谐器核心逻辑与状态机
- 指数增加 + 二分下降探测
- dynamic 的在线调整与 hybird 的稳定锁定
- 基础边界与安全检查

#### 2.1.2 修改现有文件

**修改 1: `src/scheduler/scheduler.cpp`**

```cpp
// 在 Scheduler 类中添加成员（scheduler.h）
private:
    std::unique_ptr<DynamicBatchTuner> batchTuner_;
    std::atomic<size_t> tunedMaxBatchSize_;

// 在 processRequests 方法中修改（scheduler.cpp:428）
void Scheduler::processRequests() {
    // ... 前置检查 ...
    
    // 🔥 关键修改: 使用动态 batch size
    size_t minBatchSize = batchTuner_->getCurrentBatchSize();
    constexpr size_t MAX_WAIT_MS_FOR_BATCH = 50;
    
    if (queueSize < minBatchSize && runningCount == 0) {
        CLLM_DEBUG("[Scheduler::processRequests] Queue size (%zu) < %zu (dynamic), waiting for more requests (max %dms)",
                  queueSize, minBatchSize, MAX_WAIT_MS_FOR_BATCH);
        
        // 等待逻辑保持不变
        std::unique_lock<std::mutex> lock(queueMutex_);
        auto waitStart = std::chrono::steady_clock::now();
        
        queueCondition_.wait_for(
            lock,
            std::chrono::milliseconds(MAX_WAIT_MS_FOR_BATCH),
            [this, minBatchSize]() {
                return requestQueue_.getQueueSize() >= minBatchSize || !running_;
            }
        );
        
        // ... 后续处理 ...
    }
    
    // ... 批处理形成 ...
    
    // 🔥 新增: 记录批处理性能指标
    auto batchStart = std::chrono::steady_clock::now();
    
    // 执行批处理
    SchedulerBatchProcessor processor(this, modelExecutor_, kvCache_, &batchManager_);
    processor.processBatch(activeBatch);
    
    auto batchEnd = std::chrono::steady_clock::now();
    auto processingTime = std::chrono::duration_cast<std::chrono::milliseconds>(batchEnd - batchStart).count();
    
    // 收集性能指标
    DynamicBatchTuner::PerformanceMetrics metrics;
    metrics.throughput = calculateThroughput(activeBatch, processingTime);
    metrics.gpuUtilization = queryGPUUtilization();
    metrics.memoryUtilization = queryMemoryUtilization();
    metrics.processingTimeMs = processingTime;
    metrics.peakMemoryMb = queryPeakMemoryUsage();
    metrics.timestamp = batchEnd;
    
    // 报告给调谐器
    batchTuner_->reportBatchCompletion(metrics);
}

// 在 Scheduler 构造函数中初始化
Scheduler::Scheduler() {
    // ... 现有初始化 ...
    
    // 初始化动态批处理调谐器
    DynamicBatchTuner::TunerConfig tunerConfig;
    tunerConfig.minBatchSize = 1;
    tunerConfig.maxBatchSize = 256;
    tunerConfig.initialBatchSize = 8;
    tunerConfig.maxProbingAttempts = 10;
    tunerConfig.probingGrowthFactor = 2.0;
    tunerConfig.performanceThreshold = 0.05;  // 5%
    tunerConfig.growthCoefficient = 0.2;
    tunerConfig.decayCoefficient = 0.3;
    tunerConfig.oscillationCoefficient = 0.05;
    tunerConfig.stabilizationCycles = 10;
    tunerConfig.batchSizeChangeThreshold = 0.10;  // 10%
    tunerConfig.performanceFluctuationThreshold = 0.03;  // 3%
    tunerConfig.memorySafetyMarginMb = 512;
    tunerConfig.memoryUsageLimit = 0.90;  // 90%
    
    batchTuner_ = std::make_unique<DynamicBatchTuner>();
    batchTuner_->initialize(tunerConfig);
    batchTuner_->detectGPU();
}
```

**修改 2: `include/cllm/scheduler/scheduler.h`**
- 添加 `DynamicBatchTuner` 前向声明
- 添加 `batchTuner_` 成员变量声明
- 添加相关的辅助方法声明

**修改 3: `config/config.yaml`**
```yaml
# 动态 Batch Size 调谐器配置
dynamic_batch_tuner:
  enabled: true                    # 总开关
  strategy: "dynamic"              # 可选: static | dynamic | hybird

  # static 专用（与现有机制一致）
  fixed_batch_size: 0              # 0 表示沿用现有 batch 计算逻辑

  # dynamic / hybird 基础配置
  min_batch_size: 1                # 最小 batch size
  max_batch_size: 256              # 最大 batch size
  initial_batch_size: 8            # 初始 batch size

  # 指数增加 + 二分下降配置
  probing_growth_factor: 2.0       # 指数增长因子
  max_probing_attempts: 10         # 最大探测次数

  # dynamic 在线调整参数（hybird 不使用）
  performance_threshold: 0.05      # 性能变化阈值
  adjustment_factor: 0.3           # 调整幅度（上调/下调）

  # 安全参数
  memory_usage_limit: 0.90         # 内存使用限制 (90%)
  max_consecutive_failures: 3      # 最大连续失败次数
```

### 2.2 关键函数设计

#### 2.2.1 GPU 性能探测函数

```cpp
void DynamicBatchTuner::detectGPU() {
    // 1. 识别 GPU 型号
    #ifdef __APPLE__
        // Apple Silicon: 使用 Metal API
        gpuInfo_.model = detectAppleGPUModel();
        gpuInfo_.totalMemoryMb = queryAppleGPUMemory();
        gpuInfo_.computeUnits = queryAppleGPUComputeUnits();
        gpuInfo_.memoryBandwidthGbPerSec = estimateAppleGPUMemoryBandwidth();
        gpuInfo_.theoreticalPeakTflops = calculateAppleGPUPerformance();
    #elif defined(__CUDA__)
        // NVIDIA GPU: 使用 CUDA API
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        if (deviceCount > 0) {
            cudaDeviceProp props;
            cudaGetDeviceProperties(&props, 0);
            gpuInfo_.model = props.name;
            gpuInfo_.totalMemoryMb = props.totalGlobalMem / (1024 * 1024);
            gpuInfo_.computeUnits = props.multiProcessorCount;
            gpuInfo_.memoryBandwidthGbPerSec = 
                (props.memoryBusWidth * props.memoryClockRate * 2) / 1e6;
            gpuInfo_.theoreticalPeakTflops = 
                calculateNVIDIAPeakPerformance(props);
        }
    #elif defined(__HIP_PLATFORM_HCC__)
        // AMD GPU: 使用 HIP API
        // 类似 CUDA 的实现
    #else
        // CPU fallback
        gpuInfo_.model = "CPU Only";
        gpuInfo_.totalMemoryMb = querySystemMemory();
        gpuInfo_.computeUnits = std::thread::hardware_concurrency();
    #endif
    
    CLLM_INFO("[DynamicBatchTuner] Detected GPU: %s", gpuInfo_.model.c_str());
    CLLM_INFO("[DynamicBatchTuner] Memory: %zu MB, Compute Units: %zu", 
              gpuInfo_.totalMemoryMb, gpuInfo_.computeUnits);
    CLLM_INFO("[DynamicBatchTuner] Memory Bandwidth: %.2f GB/s, Peak Performance: %.2f TFLOPS",
              gpuInfo_.memoryBandwidthGbPerSec, gpuInfo_.theoreticalPeakTflops);
}
```

#### 2.2.2 性能指标收集函数

```cpp
void Scheduler::collectPerformanceMetrics(
    const std::vector<RequestState>& batch,
    double processingTimeMs,
    DynamicBatchTuner::PerformanceMetrics& metrics) {
    
    // 1. 计算吞吐量
    size_t totalTokens = 0;
    for (const auto& request : batch) {
        totalTokens += request.generatedTokens.size();
    }
    metrics.throughput = totalTokens / (processingTimeMs / 1000.0);
    
    // 2. 查询 GPU 利用率
    #ifdef __APPLE__
        metrics.gpuUtilization = queryAppleGPUUtilization();
    #elif defined(__CUDA__)
        metrics.gpuUtilization = queryNVIDIAGPUUtilization();
    #else
        metrics.gpuUtilization = 0.5; // 默认值
    #endif
    
    // 3. 查询内存利用率
    size_t usedMemory = queryCurrentMemoryUsage();
    metrics.memoryUtilization = 
        static_cast<double>(usedMemory) / gpuInfo_.totalMemoryMb;
    
    // 4. 记录批处理时间
    metrics.processingTimeMs = processingTimeMs;
    
    // 5. 查询峰值内存
    metrics.peakMemoryMb = queryPeakMemoryUsage();
    
    // 6. 记录时间戳
    metrics.timestamp = std::chrono::steady_clock::now();
}
```

#### 2.2.3 Batch Size 计算函数

```cpp
size_t DynamicBatchTuner::getOptimalBatchSize() {
    std::lock_guard<std::mutex> lock(metricsMutex_);
    
    switch (currentPhase_) {
        case TuningPhase::INITIAL_PROBING:
            return getProbingBatchSize();
            
        case TuningPhase::DYNAMIC_ADJUSTMENT:
            return currentBatchSize_.load();
            
        case TuningPhase::STABLE_RUNNING:
            return currentBatchSize_.load();
            
        default:
            return config_.initialBatchSize;
    }
}

size_t DynamicBatchTuner::getProbingBatchSize() {
    // 指数增长探测
    static size_t attempt = 0;
    size_t probingSize = config_.initialBatchSize * 
        std::pow(config_.probingGrowthFactor, attempt);
    
    probingSize = std::min(probingSize, config_.maxBatchSize);
    probingSize = std::max(probingSize, config_.minBatchSize);
    
    return probingSize;
}
```

### 2.3 与现有调度系统的集成

#### 2.3.1 数据流图

```
用户请求
    ↓
RequestQueue (无锁队列)
    ↓
Scheduler::processRequests()
    ↓
DynamicBatchTuner::getOptimalBatchSize()
    ↓
[等待累积] 或 [立即处理]
    ↓
BatchManager::formBatch()
    ↓
BatchProcessor::processBatch()
    ↓
InferenceEngine::forwardBatch()
    ↓
LlamaCppBackend / LibTorchBackend / KylinBackend
    ↓
GPU 执行推理
    ↓
收集性能指标
    ↓
DynamicBatchTuner::reportBatchCompletion()
    ↓
更新历史记录 → 调整 batch size → 循环
```

#### 2.3.2 集成点说明

**集成点 1: Batch Size 查询**
- **位置**: `scheduler.cpp:428`
- **调用**: `batchTuner_->getOptimalBatchSize()`
- **频率**: 每次 `processRequests()` 调用时
- **开销**: < 1μs（原子操作）

**集成点 2: 性能指标报告**
- **位置**: `scheduler.cpp:700`（批处理完成后）
- **调用**: `batchTuner_->reportBatchCompletion(metrics)`
- **频率**: 每次批处理完成时
- **开销**: ~10μs（指标收集和分析）

**集成点 3: GPU 状态查询**
- **位置**: `llama_cpp_backend.cpp` / `libtorch_backend.cpp`
- **调用**: `queryGPUUtilization()`, `queryMemoryUtilization()`
- **频率**: 每 N 个批处理（可配置）
- **开销**: ~100μs（系统调用）

**集成点 4: 配置加载**
- **位置**: `config/config.cpp`
- **调用**: 解析 `dynamic_batch_tuner` 配置节
- **频率**: 系统启动时
- **开销**: 一次性

---

## 3. 自适应机制

### 3.1 GPU 性能探测流程

#### 3.1.1 初始探测阶段

**目标**: 在系统启动后的前 N 个批处理中，快速找到可行的 batch size 范围

**流程图**:

```
开始
  ↓
获取 GPU 信息（型号、内存、计算单元）
  ↓
估算初始 batch size
  ↓
for attempt = 0 to maxProbingAttempts:
  ↓
  使用指数增长的 batch size 执行推理
  1, 2, 4, 8, 16, 32, ...
  ↓
  收集性能指标
  - 吞吐量
  - GPU 利用率
  - 内存占用
  - 批处理时间
  ↓
  检查是否成功:
  - 内存是否溢出?
  - 性能是否下降?
  - 是否达到稳定?
  ↓
  if 成功:
      记录为可行值
      继续增大
  else:
      记录为上限
      使用二分查找精确定位
      break
  ↓
end for
  ↓
使用二分查找精确定位最优值
  ↓
记录最佳 batch size
  ↓
进入动态调整阶段
```

**关键代码**:

```cpp
void DynamicBatchTuner::runInitialProbing() {
    CLLM_INFO("[DynamicBatchTuner] Starting initial probing phase");
    
    size_t lowerBound = config_.minBatchSize;
    size_t upperBound = config_.maxBatchSize;
    size_t bestSize = config_.initialBatchSize;
    double bestScore = 0.0;
    
    // 阶段 1: 指数增长探测
    for (size_t attempt = 0; attempt < config_.maxProbingAttempts; ++attempt) {
        size_t probingSize = config_.initialBatchSize * 
            std::pow(config_.probingGrowthFactor, attempt);
        probingSize = std::min(probingSize, config_.maxBatchSize);
        
        if (!isMemorySafe(probingSize)) {
            upperBound = probingSize;
            break;
        }
        
        // 执行探测推理（实际会在 processBatch 中执行）
        currentBatchSize_ = probingSize;
        
        // 等待性能指标（异步）
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // 检查最新指标
        if (!metricsHistory_.empty()) {
            const auto& latest = metricsHistory_.back();
            double score = calculateScore(latest);
            
            if (score > bestScore) {
                bestScore = score;
                bestSize = probingSize;
            }
            
            // 检查性能是否下降
            if (attempt > 2 && score < bestScore * 0.8) {
                CLLM_DEBUG("[DynamicBatchTuner] Performance degraded at batch size %zu", probingSize);
                upperBound = probingSize;
                break;
            }
        }
        
        lowerBound = std::max(lowerBound, probingSize);
    }
    
    // 阶段 2: 二分查找精确定位
    CLLM_DEBUG("[DynamicBatchTuner] Binary search between %zu and %zu", lowerBound, upperBound);
    
    for (size_t i = 0; i < 5; ++i) { // 最多 5 次二分
        if (upperBound - lowerBound <= 1) {
            break;
        }
        
        size_t mid = (lowerBound + upperBound) / 2;
        
        if (!isMemorySafe(mid)) {
            upperBound = mid;
            continue;
        }
        
        currentBatchSize_ = mid;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        if (!metricsHistory_.empty()) {
            const auto& latest = metricsHistory_.back();
            double score = calculateScore(latest);
            
            if (score > bestScore * 0.95) {
                bestScore = score;
                bestSize = mid;
                lowerBound = mid;
            } else {
                upperBound = mid;
            }
        }
    }
    
    // 确定最终值
    currentBatchSize_ = bestSize;
    bestBatchSize_ = bestSize;
    bestScore_ = bestScore;
    
    CLLM_INFO("[DynamicBatchTuner] Initial probing complete. Best batch size: %zu (score: %.3f)", 
              bestSize, bestScore);
    
    // 进入动态调整阶段
    currentPhase_ = TuningPhase::DYNAMIC_ADJUSTMENT;
}
```

#### 3.1.2 动态调整阶段

**目标**: 在运行时持续优化 batch size，适应负载变化

**调整算法**:

```cpp
void DynamicBatchTuner::runDynamicAdjustment(const PerformanceMetrics& metrics) {
    double currentScore = calculateScore(metrics);
    
    // 1. 检查是否应该增大 batch size
    if (shouldIncreaseBatchSize(metrics)) {
        size_t newSize = currentBatchSize_ * (1 + config_.growthCoefficient);
        newSize = std::min(newSize, config_.maxBatchSize);
        newSize = std::min(newSize, bestBatchSize_ * 2); // 不超过历史最佳的 2 倍
        
        if (isMemorySafe(newSize)) {
            CLLM_DEBUG("[DynamicBatchTuner] Increasing batch size: %zu → %zu", 
                      currentBatchSize_, newSize);
            currentBatchSize_ = newSize;
            
            if (currentScore > bestScore_) {
                bestScore_ = currentScore;
                bestBatchSize_ = newSize;
            }
        }
    }
    
    // 2. 检查是否应该减小 batch size
    else if (shouldDecreaseBatchSize(metrics)) {
        size_t newSize = currentBatchSize_ * (1 - config_.decayCoefficient);
        newSize = std::max(newSize, config_.minBatchSize);
        
        CLLM_DEBUG("[DynamicBatchTuner] Decreasing batch size: %zu → %zu", 
                  currentBatchSize_, newSize);
        currentBatchSize_ = newSize;
    }
    
    // 3. 小幅震荡探索
    else {
        double oscillation = config_.oscillationCoefficient * 
            (std::rand() % 2 == 0 ? 1 : -1);
        size_t newSize = currentBatchSize_ * (1 + oscillation);
        newSize = std::clamp(newSize, config_.minBatchSize, config_.maxBatchSize);
        
        if (isMemorySafe(newSize)) {
            currentBatchSize_ = newSize;
        }
    }
    
    // 4. 检查是否进入稳定阶段
    if (checkStabilizationCondition()) {
        CLLM_INFO("[DynamicBatchTuner] Entering stable running phase. Batch size: %zu", 
                  currentBatchSize_);
        currentPhase_ = TuningPhase::STABLE_RUNNING;
    }
}

bool DynamicBatchTuner::shouldIncreaseBatchSize(const PerformanceMetrics& metrics) const {
    // 条件 1: 性能提升
    double currentScore = calculateScore(metrics);
    if (currentScore > bestScore_ * (1 + config_.performanceThreshold)) {
        return true;
    }
    
    // 条件 2: GPU 利用率低
    if (metrics.gpuUtilization < 0.5) {
        return true;
    }
    
    // 条件 3: 内存有富余
    if (metrics.memoryUtilization < 0.7) {
        return true;
    }
    
    return false;
}

bool DynamicBatchTuner::shouldDecreaseBatchSize(const PerformanceMetrics& metrics) const {
    // 条件 1: 性能下降
    double currentScore = calculateScore(metrics);
    if (currentScore < bestScore_ * (1 - config_.performanceThreshold)) {
        return true;
    }
    
    // 条件 2: 内存接近上限
    if (metrics.memoryUtilization > config_.memoryUsageLimit) {
        return true;
    }
    
    // 条件 3: 批处理时间过长
    if (metrics.processingTimeMs > 1000) { // > 1s
        return true;
    }
    
    return false;
}
```

#### 3.1.3 稳定运行阶段

**目标**: 保持最优 batch size，避免频繁波动

**稳定策略**:

```cpp
void DynamicBatchTuner::runStableRunning(const PerformanceMetrics& metrics) {
    // 1. 监控性能波动
    double currentScore = calculateScore(metrics);
    
    // 2. 检查是否需要重新调整
    if (currentScore < bestScore_ * 0.8) {
        // 性能下降超过 20%，重新进入调整阶段
        CLLM_WARN("[DynamicBatchTuner] Performance dropped significantly. Re-entering adjustment phase");
        currentPhase_ = TuningPhase::DYNAMIC_ADJUSTMENT;
        return;
    }
    
    // 3. 定期探索（防止陷入局部最优）
    static size_t stableCycles = 0;
    stableCycles++;
    
    if (stableCycles >= config_.stabilizationCycles * 10) {
        // 每 10 个稳定周期，进行一次探索
        stableCycles = 0;
        
        size_t explorationSize = currentBatchSize_ * (1 + config_.oscillationCoefficient);
        explorationSize = std::min(explorationSize, config_.maxBatchSize);
        
        if (isMemorySafe(explorationSize)) {
            CLLM_DEBUG("[DynamicBatchTuner] Exploration: trying batch size %zu", explorationSize);
            currentBatchSize_ = explorationSize;
        }
    }
    
    // 4. 负载变化检测
    if (isLoadChangedSignificantly()) {
        CLLM_DEBUG("[DynamicBatchTuner] Load changed significantly. Re-entering adjustment phase");
        currentPhase_ = TuningPhase::DYNAMIC_ADJUSTMENT;
    }
}

bool DynamicBatchTuner::checkStabilizationCondition() const {
    if (metricsHistory_.size() < config_.stabilizationCycles) {
        return false;
    }
    
    // 1. 检查 batch size 变化
    size_t firstSize = metricsHistory_[0].batchSize;
    size_t lastSize = metricsHistory_.back().batchSize;
    double sizeChange = std::abs(static_cast<double>(lastSize - firstSize) / firstSize);
    
    if (sizeChange > config_.batchSizeChangeThreshold) {
        return false;
    }
    
    // 2. 检查性能波动
    double sumScore = 0.0;
    double maxScore = 0.0;
    double minScore = std::numeric_limits<double>::max();
    
    for (const auto& m : metricsHistory_) {
        double score = calculateScore(m);
        sumScore += score;
        maxScore = std::max(maxScore, score);
        minScore = std::min(minScore, score);
    }
    
    double avgScore = sumScore / metricsHistory_.size();
    double fluctuation = (maxScore - minScore) / avgScore;
    
    if (fluctuation > config_.performanceFluctuationThreshold) {
        return false;
    }
    
    // 3. 检查连续周期数
    if (metricsHistory_.size() < config_.stabilizationCycles) {
        return false;
    }
    
    return true;
}
```

### 3.2 状态转换图

```
                    ┌──────────────────────────────────────┐
                    │                                      │
                    ▼                                      │
    ┌──────────────────────┐                               │
    │  INITIAL_PROBING     │                               │
    │  (初始探测阶段)       │── 探测完成 ──►                 │
    └──────────────────────┘                               │
           │                                               │
           │ 探测失败                                       │
           ▼                                               │
    ┌──────────────────────┐                               │
    │  SAFE_FALLBACK       │                               │
    │  (安全回退)          │── 恢复正常 ──►                 │
    └──────────────────────┘                               │
                                                           │
                    ┌──────────────────────────────────────┘
                    │
                    ▼
    ┌──────────────────────┐                               │
    │ DYNAMIC_ADJUSTMENT   │                               │
    │ (动态调整阶段)        │── 满足稳定条件 ──►             │
    └──────────────────────┘                               │
           │                                               │
           │ 性能下降 / 负载变化                           │
           ▼                                               │
    ┌──────────────────────┐                               │
    │  RE-EVALUATION       │                               │
    │  (重新评估)          │── 评估完成 ──►                 │
    └──────────────────────┘                               │
                                                           │
                    ┌──────────────────────────────────────┘
                    │
                    ▼
    ┌──────────────────────┐                               │
    │ STABLE_RUNNING       │                               │
    │ (稳定运行阶段)        │                               │
    └──────────────────────┘                               │
           │                                               │
           │ 定期探索 / 性能下降                           │
           └─────────────────► DYNAMIC_ADJUSTMENT          │
```

---

## 4. 安全机制

### 4.1 边界检查

#### 4.1.1 Batch Size 边界

```cpp
size_t DynamicBatchTuner::clampBatchSize(size_t batchSize) const {
    return std::clamp(batchSize, 
                      config_.minBatchSize, 
                      config_.maxBatchSize);
}

bool DynamicBatchTuner::isBatchSizeValid(size_t batchSize) const {
    return batchSize >= config_.minBatchSize && 
           batchSize <= config_.maxBatchSize;
}
```

#### 4.1.2 内存边界

```cpp
bool DynamicBatchTuner::isMemorySafe(size_t batchSize) const {
    // 估算内存需求
    size_t estimatedMemoryMb = estimateMemoryUsage(batchSize);
    
    // 检查是否在安全范围内
    size_t availableMemory = gpuInfo_.totalMemoryMb - config_.memorySafetyMarginMb;
    
    if (estimatedMemoryMb > availableMemory * config_.memoryUsageLimit) {
        return false;
    }
    
    return true;
}

size_t DynamicBatchTuner::estimateMemoryUsage(size_t batchSize) const {
    // 简化估算：
    // 内存 ≈ batch_size × avg_seq_len × memory_per_token
    // 
    // 对于 qwen3-0.6b:
    // - avg_seq_len ≈ 100 tokens
    // - memory_per_token ≈ 2 MB
    // 
    // 因此: memory ≈ batch_size × 200 MB
    
    return batchSize * 200; // MB
}
```

#### 4.1.3 性能边界

```cpp
bool DynamicBatchTuner::isPerformanceAcceptable(const PerformanceMetrics& metrics) const {
    // 1. 吞吐量检查
    if (metrics.throughput < 10) { // < 10 tokens/sec
        CLLM_WARN("[DynamicBatchTuner] Throughput too low: %.2f tokens/sec", metrics.throughput);
        return false;
    }
    
    // 2. 批处理时间检查
    if (metrics.processingTimeMs > 5000) { // > 5s
        CLLM_WARN("[DynamicBatchTuner] Processing time too long: %.2f ms", metrics.processingTimeMs);
        return false;
    }
    
    // 3. GPU 利用率检查
    if (metrics.gpuUtilization < 0.1) { // < 10%
        CLLM_WARN("[DynamicBatchTuner] GPU utilization too low: %.2f%%", 
                  metrics.gpuUtilization * 100);
        return false;
    }
    
    return true;
}
```

### 4.2 异常处理

#### 4.2.1 内存溢出处理

```cpp
void DynamicBatchTuner::handleMemoryOverflow() {
    CLLM_ERROR("[DynamicBatchTuner] Memory overflow detected! Current batch size: %zu", 
               currentBatchSize_);
    
    // 1. 记录失败次数
    consecutiveFailures_++;
    
    // 2. 立即减小 batch size
    size_t newSize = currentBatchSize_ / 2;
    newSize = std::max(newSize, config_.minBatchSize);
    
    CLLM_WARN("[DynamicBatchTuner] Reducing batch size to %zu", newSize);
    currentBatchSize_ = newSize;
    
    // 3. 检查是否需要回退到安全状态
    if (consecutiveFailures_ >= config_.maxConsecutiveFailures) {
        resetToSafeState();
    }
    
    // 4. 重新进入探测阶段
    currentPhase_ = TuningPhase::INITIAL_PROBING;
}

void DynamicBatchTuner::resetToSafeState() {
    CLLM_ERROR("[DynamicBatchTuner] Too many consecutive failures. Resetting to safe state");
    
    // 重置到最小 batch size
    currentBatchSize_ = config_.minBatchSize;
    
    // 清空历史记录
    metricsHistory_.clear();
    
    // 重置失败计数
    consecutiveFailures_ = 0;
    
    // 重置最佳记录
    bestBatchSize_ = config_.initialBatchSize;
    bestScore_ = 0.0;
    
    // 进入初始探测阶段
    currentPhase_ = TuningPhase::INITIAL_PROBING;
    
    CLLM_INFO("[DynamicBatchTuner] Reset to safe state. Batch size: %zu", 
              currentBatchSize_);
}
```

#### 4.2.2 性能异常处理

```cpp
void DynamicBatchTuner::handlePerformanceAnomaly(const PerformanceMetrics& metrics) {
    double currentScore = calculateScore(metrics);
    
    // 检查是否异常
    if (currentScore < bestScore_ * 0.5) { // 性能下降超过 50%
        CLLM_WARN("[DynamicBatchTuner] Performance anomaly detected! Score dropped from %.3f to %.3f", 
                  bestScore_, currentScore);
        
        // 1. 检查是否是临时波动
        if (isTemporaryFluctuation()) {
            CLLM_DEBUG("[DynamicBatchTuner] Likely temporary fluctuation, ignoring");
            return;
        }
        
        // 2. 尝试恢复到历史最佳
        if (bestBatchSize_ != currentBatchSize_) {
            CLLM_DEBUG("[DynamicBatchTuner] Reverting to best batch size: %zu", bestBatchSize_);
            currentBatchSize_ = bestBatchSize_;
            return;
        }
        
        // 3. 重新评估
        currentPhase_ = TuningPhase::DYNAMIC_ADJUSTMENT;
    }
}

bool DynamicBatchTuner::isTemporaryFluctuation() const {
    if (metricsHistory_.size() < 3) {
        return false;
    }
    
    // 检查最近 3 个指标的趋势
    const auto& m1 = metricsHistory_[metricsHistory_.size() - 3];
    const auto& m2 = metricsHistory_[metricsHistory_.size() - 2];
    const auto& m3 = metricsHistory_.back();
    
    double s1 = calculateScore(m1);
    double s2 = calculateScore(m2);
    double s3 = calculateScore(m3);
    
    // 如果是 V 型波动（下降后立即上升），可能是临时的
    if (s2 < s1 * 0.8 && s3 > s2 * 1.2) {
        return true;
    }
    
    return false;
}
```

### 4.3 回退策略

#### 4.3.1 多级回退机制

```cpp
enum class FallbackLevel {
    NONE,           // 无回退
    MINOR_ADJUSTMENT, // 小幅调整
    MODERATE_FALLBACK, // 中度回退
    MAJOR_FALLBACK,    // 大幅回退
    SAFE_MODE          // 安全模式
};

FallbackLevel DynamicBatchTuner::determineFallbackLevel(const PerformanceMetrics& metrics) const {
    double currentScore = calculateScore(metrics);
    
    // 计算性能下降比例
    double scoreDrop = 1.0 - (currentScore / bestScore_);
    
    if (scoreDrop < 0.1) {
        return FallbackLevel::NONE; // < 10% 下降
    }
    else if (scoreDrop < 0.3) {
        return FallbackLevel::MINOR_ADJUSTMENT; // 10-30% 下降
    }
    else if (scoreDrop < 0.5) {
        return FallbackLevel::MODERATE_FALLBACK; // 30-50% 下降
    }
    else if (scoreDrop < 0.8) {
        return FallbackLevel::MAJOR_FALLBACK; // 50-80% 下降
    }
    else {
        return FallbackLevel::SAFE_MODE; // > 80% 下降
    }
}

void DynamicBatchTuner::executeFallback(FallbackLevel level) {
    switch (level) {
        case FallbackLevel::NONE:
            // 无操作
            break;
            
        case FallbackLevel::MINOR_ADJUSTMENT:
            // 小幅调整: 减小 10%
            currentBatchSize_ = currentBatchSize_ * 0.9;
            CLLM_DEBUG("[DynamicBatchTuner] Minor adjustment: batch size %zu", currentBatchSize_);
            break;
            
        case FallbackLevel::MODERATE_FALLBACK:
            // 中度回退: 减小 30%，恢复到历史最佳
            currentBatchSize_ = std::min(currentBatchSize_ * 0.7, bestBatchSize_);
            CLLM_WARN("[DynamicBatchTuner] Moderate fallback: batch size %zu", currentBatchSize_);
            break;
            
        case FallbackLevel::MAJOR_FALLBACK:
            // 大幅回退: 减小 50%，重新进入调整阶段
            currentBatchSize_ = currentBatchSize_ * 0.5;
            currentPhase_ = TuningPhase::DYNAMIC_ADJUSTMENT;
            CLLM_WARN("[DynamicBatchTuner] Major fallback: batch size %zu", currentBatchSize_);
            break;
            
        case FallbackLevel::SAFE_MODE:
            // 安全模式: 重置到最小 batch size
            resetToSafeState();
            CLLM_ERROR("[DynamicBatchTuner] Entering safe mode");
            break;
    }
}
```

#### 4.3.2 自动恢复机制

```cpp
void DynamicBatchTuner::attemptRecovery() {
    if (currentPhase_ != TuningPhase::STABLE_RUNNING) {
        return;
    }
    
    // 检查是否可以恢复
    if (isPerformanceRecovering()) {
        CLLM_INFO("[DynamicBatchTuner] Performance is recovering. Attempting to increase batch size");
        
        // 逐步增大 batch size
        size_t targetSize = currentBatchSize_ * 1.1;
        targetSize = std::min(targetSize, bestBatchSize_);
        
        if (isMemorySafe(targetSize)) {
            currentBatchSize_ = targetSize;
        }
    }
}

bool DynamicBatchTuner::isPerformanceRecovering() const {
    if (metricsHistory_.size() < 5) {
        return false;
    }
    
    // 检查最近 5 个指标的趋势
    std::vector<double> recentScores;
    for (size_t i = metricsHistory_.size() - 5; i < metricsHistory_.size(); ++i) {
        recentScores.push_back(calculateScore(metricsHistory_[i]));
    }
    
    // 计算斜率
    double slope = calculateTrendSlope(recentScores);
    
    // 如果斜率为正且大于阈值，说明在恢复
    return slope > 0.01; // 每步提升 > 1%
}
```

---

## 5. 性能验证

### 5.1 测试方案

#### 5.1.1 不同 GPU 型号验证

**测试矩阵**:

| GPU 型号 | 内存 | 理论性能 | 预期 batch size | 测试场景 |
|---------|------|---------|----------------|--------|
| **NVIDIA A100 40GB** | 40GB | 312 TFLOPS | 64-128 | 高并发推理 |
| **NVIDIA A10 24GB** | 24GB | 19.5 TFLOPS | 32-64 | 中等并发 |
| **NVIDIA T4 16GB** | 16GB | 8.1 TFLOPS | 16-32 | 低并发 |
| **Apple M3 Ultra** | 128GB | ~100 TOPS | 32-64 | 混合负载 |
| **Apple M3 Pro** | 36GB | ~40 TOPS | 16-32 | 标准负载 |
| **AMD MI250X** | 64GB | 819 TFLOPS | 128-256 | 超高并发 |
| **Intel Arc A770** | 16GB | 21 TFLOPS | 16-32 | 入门级 |

#### 5.1.2 测试用例设计

**测试用例 1: 初始探测阶段验证**

```python
def test_initial_probing_phase():
    """验证初始探测阶段能否正确找到可行的 batch size 范围"""
    
    # 步骤 1: 启动服务器，启用动态批处理
    server = start_cllm_server(enable_dynamic_batch=True)
    
    # 步骤 2: 发送一系列请求
    for i in range(20):
        send_request(prompt="Hello", max_tokens=50)
    
    # 步骤 3: 检查调谐器状态
    tuner_status = server.get_tuner_status()
    
    # 断言 1: 应该完成初始探测
    assert tuner_status.phase == "DYNAMIC_ADJUSTMENT" or tuner_status.phase == "STABLE_RUNNING"
    
    # 断言 2: 应该找到合理的 batch size
    assert tuner_status.current_batch_size >= 1
    assert tuner_status.current_batch_size <= 256
    
    # 断言 3: 应该有历史最佳记录
    assert tuner_status.best_batch_size > 0
    assert tuner_status.best_score > 0
    
    print(f"✓ Initial probing phase completed. Best batch size: {tuner_status.best_batch_size}")
```

**测试用例 2: 动态调整阶段验证**

```python
def test_dynamic_adjustment_under_load():
    """验证动态调整阶段能否适应负载变化"""
    
    # 步骤 1: 启动服务器
    server = start_cllm_server(enable_dynamic_batch=True)
    
    # 步骤 2: 低负载运行
    print("Phase 1: Low load (2 concurrent requests)")
    for i in range(10):
        send_concurrent_requests(count=2, max_tokens=50)
    
    low_load_batch_size = server.get_tuner_status().current_batch_size
    
    # 步骤 3: 高负载运行
    print("Phase 2: High load (32 concurrent requests)")
    for i in range(10):
        send_concurrent_requests(count=32, max_tokens=50)
    
    high_load_batch_size = server.get_tuner_status().current_batch_size
    
    # 断言: 高负载下应该增大 batch size
    assert high_load_batch_size >= low_load_batch_size * 0.8, \
        f"Batch size should not decrease significantly under high load"
    
    print(f"✓ Dynamic adjustment works: {low_load_batch_size} → {high_load_batch_size}")
```

**测试用例 3: 内存溢出处理验证**

```python
def test_memory_overflow_handling():
    """验证内存溢出时的安全机制"""
    
    # 步骤 1: 限制 GPU 内存
    server = start_cllm_server(
        enable_dynamic_batch=True,
        max_gpu_memory_mb=4096  # 限制为 4GB
    )
    
    # 步骤 2: 发送大请求触发内存溢出
    try:
        send_request(prompt="Hello " * 1000, max_tokens=1000)
    except MemoryOverflowError:
        pass
    
    # 步骤 3: 检查调谐器状态
    tuner_status = server.get_tuner_status()
    
    # 断言 1: 应该减小 batch size
    assert tuner_status.current_batch_size < tuner_status.best_batch_size
    
    # 断言 2: 应该记录失败
    assert tuner_status.consecutive_failures > 0
    
    # 断言 3: 系统应该继续运行
    assert server.is_running()
    
    print("✓ Memory overflow handling works correctly")
```

**测试用例 4: 稳定运行阶段验证**

```python
def test_stable_running_phase():
    """验证稳定运行阶段能否保持最优 batch size"""
    
    # 步骤 1: 启动服务器并运行一段时间
    server = start_cllm_server(enable_dynamic_batch=True)
    
    # 步骤 2: 持续发送请求直到进入稳定阶段
    batch_sizes = []
    for i in range(50):
        send_concurrent_requests(count=16, max_tokens=50)
        
        status = server.get_tuner_status()
        batch_sizes.append(status.current_batch_size)
        
        if status.phase == "STABLE_RUNNING":
            print(f"Entered stable running phase after {i+1} batches")
            break
    
    # 步骤 3: 继续运行，检查 batch size 是否稳定
    stable_batch_sizes = []
    for i in range(20):
        send_concurrent_requests(count=16, max_tokens=50)
        status = server.get_tuner_status()
        stable_batch_sizes.append(status.current_batch_size)
    
    # 断言: batch size 应该保持相对稳定
    max_variation = max(stable_batch_sizes) - min(stable_batch_sizes)
    assert max_variation <= 4, \
        f"Batch size variation too large: {max_variation}"
    
    print(f"✓ Stable running phase maintained. Batch size variation: {max_variation}")
```

### 5.2 性能提升评估指标

#### 5.2.1 核心指标

```python
class PerformanceMetrics:
    def __init__(self):
        # 吞吐量指标
        self.throughput_improvement = 0.0  # 相对于静态配置的提升比例
        self.max_throughput = 0.0          # 最大吞吐量
        self.avg_throughput = 0.0          # 平均吞吐量
        
        # 延迟指标
        self.p50_latency = 0.0             # P50 延迟
        self.p95_latency = 0.0             # P95 延迟
        self.p99_latency = 0.0             # P99 延迟
        
        # 资源利用率指标
        self.avg_gpu_utilization = 0.0     # 平均 GPU 利用率
        self.peak_gpu_utilization = 0.0    # 峰值 GPU 利用率
        self.avg_memory_utilization = 0.0  # 平均内存利用率
        
        # 稳定性指标
        self.success_rate = 0.0            # 请求成功率
        self.batch_size_stability = 0.0    # batch size 稳定性（标准差）
        self.consecutive_failures = 0      # 连续失败次数
        
        # 收敛指标
        self.convergence_time = 0.0        # 收敛时间（秒）
        self.convergence_iterations = 0    # 收敛迭代次数
        
    def calculate_overall_score(self):
        """计算综合评分（0-100）"""
        score = 0.0
        
        # 吞吐量权重: 40%
        score += min(self.throughput_improvement * 100, 40)
        
        # GPU 利用率权重: 30%
        score += self.avg_gpu_utilization * 30
        
        # 稳定性权重: 20%
        score += self.success_rate * 20
        
        # 收敛速度权重: 10%
        if self.convergence_time < 60:  # < 1min
            score += 10
        elif self.convergence_time < 120:  # < 2min
            score += 5
        
        return score
```

#### 5.2.2 与静态设置的对比

**对比测试脚本**:

```python
def compare_static_vs_dynamic():
    """对比静态 batch size 和动态调整的性能"""
    
    # 测试配置
    test_configs = [
        {"name": "Static Batch Size 8", "dynamic": False, "batch_size": 8},
        {"name": "Static Batch Size 16", "dynamic": False, "batch_size": 16},
        {"name": "Static Batch Size 32", "dynamic": False, "batch_size": 32},
        {"name": "Static Batch Size 64", "dynamic": False, "batch_size": 64},
        {"name": "Dynamic Batch Tuner", "dynamic": True, "batch_size": None},
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\n{'='*60}")
        print(f"Testing: {config['name']}")
        print(f"{'='*60}")
        
        # 启动服务器
        server = start_cllm_server(
            enable_dynamic_batch=config['dynamic'],
            static_batch_size=config['batch_size']
        )
        
        # 运行测试
        metrics = run_benchmark(
            server,
            concurrent_requests=[8, 16, 24, 32],
            total_requests=72,
            max_tokens=50
        )
        
        results.append({
            'config': config['name'],
            'throughput': metrics['avg_throughput'],
            'gpu_utilization': metrics['avg_gpu_utilization'],
            'success_rate': metrics['success_rate'],
            'p95_latency': metrics['p95_latency'],
        })
        
        server.stop()
    
    # 生成对比报告
    print("\n" + "="*80)
    print("STATIC vs DYNAMIC BATCH SIZE COMPARISON")
    print("="*80)
    
    print(f"{'Configuration':<30} {'Throughput':>12} {'GPU Util':>12} {'Success Rate':>12} {'P95 Latency':>12}")
    print("-"*80)
    
    for result in results:
        print(f"{result['config']:<30} "
              f"{result['throughput']:>12.2f} "
              f"{result['gpu_utilization']*100:>11.1f}% "
              f"{result['success_rate']*100:>11.1f}% "
              f"{result['p95_latency']:>12.2f}s")
    
    # 计算动态调整的优势
    static_results = [r for r in results if 'Static' in r['config']]
    dynamic_result = next(r for r in results if 'Dynamic' in r['config'])
    
    best_static_throughput = max(r['throughput'] for r in static_results)
    throughput_improvement = (dynamic_result['throughput'] - best_static_throughput) / best_static_throughput * 100
    
    print("\n" + "="*80)
    print(f"Dynamic Batch Tuner优势: +{throughput_improvement:.1f}% 吞吐量")
    print("="*80)
```

### 5.3 性能目标

#### 5.3.1 必须满足的目标

| 指标 | 目标值 | 说明 |
|-----|--------|------|
| **吞吐量提升** | ≥ 20% | 相对于最优静态配置 |
| **GPU 利用率** | ≥ 70% | 平均利用率 |
| **请求成功率** | ≥ 99.9% | 无内存溢出导致的失败 |
| **收敛时间** | < 2分钟 | 从启动到稳定 |
| **Batch Size 稳定性** | ≤ 10% | 标准差/均值 |
| **系统开销** | < 5% | 调谐器本身的开销 |

#### 5.3.2 期望达到的目标

| 指标 | 目标值 | 说明 |
|-----|--------|------|
| **吞吐量提升** | ≥ 40% | 相对于最优静态配置 |
| **GPU 利用率** | ≥ 85% | 平均利用率 |
| **收敛时间** | < 1分钟 | 从启动到稳定 |
| **跨 GPU 兼容性** | 支持 ≥ 5 种 GPU | NVIDIA, AMD, Apple, Intel |
| **自适应能力** | 响应时间 < 10 批 | 对负载变化的响应 |

#### 5.3.3 性能回归测试

```python
def run_performance_regression_test():
    """运行性能回归测试，确保优化不会导致性能下降"""
    
    # 基准性能（来自历史最佳）
    baseline_metrics = {
        'throughput': 137.73,  # tokens/sec
        'gpu_utilization': 0.75,
        'success_rate': 0.995,
        'p95_latency': 5.36,  # seconds
    }
    
    # 运行测试
    current_metrics = run_benchmark(
        server=None,  # 使用默认配置
        concurrent_requests=[8, 16, 24, 32],
        total_requests=72,
        max_tokens=50
    )
    
    # 检查是否满足性能要求
    issues = []
    
    if current_metrics['throughput'] < baseline_metrics['throughput'] * 0.9:
        issues.append(f"吞吐量下降超过 10%: {current_metrics['throughput']:.2f} vs {baseline_metrics['throughput']:.2f}")
    
    if current_metrics['gpu_utilization'] < baseline_metrics['gpu_utilization'] * 0.8:
        issues.append(f"GPU 利用率下降超过 20%: {current_metrics['gpu_utilization']*100:.1f}% vs {baseline_metrics['gpu_utilization']*100:.1f}%")
    
    if current_metrics['success_rate'] < 0.99:
        issues.append(f"成功率低于 99%: {current_metrics['success_rate']*100:.1f}%")
    
    if current_metrics['p95_latency'] > baseline_metrics['p95_latency'] * 1.2:
        issues.append(f"P95 延迟增加超过 20%: {current_metrics['p95_latency']:.2f}s vs {baseline_metrics['p95_latency']:.2f}s")
    
    # 生成报告
    if issues:
        print("❌ 性能回归测试失败:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ 性能回归测试通过")
        return True
```

---

## 6. 配置接口

### 6.1 配置参数设计

#### 6.1.1 核心配置参数

```yaml
# 动态 Batch Size 调谐器配置
dynamic_batch_tuner:
  # 基础开关
  enabled: true                    # 是否启用动态调整
  
  # Batch Size 范围
  min_batch_size: 1                # 最小 batch size
  max_batch_size: 256              # 最大 batch size
  initial_batch_size: 8            # 初始 batch size
  
  # 探测阶段参数
  max_probing_attempts: 10         # 最大探测次数
  probing_growth_factor: 2.0       # 探测增长因子 (1.5-3.0)
  probing_timeout_ms: 5000         # 探测超时时间
  
  # 调整阶段参数
  performance_threshold: 0.05      # 性能变化阈值 (ε: 0.01-0.20)
  growth_coefficient: 0.2          # 增长系数 (α: 0.1-0.5)
  decay_coefficient: 0.3           # 衰减系数 (β: 0.2-0.5)
  oscillation_coefficient: 0.05    # 震荡系数 (δ: 0.01-0.10)
  adjustment_interval_ms: 100      # 调整间隔时间
  
  # 稳定阶段参数
  stabilization_cycles: 10         # 稳定周期数 (N: 5-20)
  batch_size_change_threshold: 0.10  # batch size 变化阈值 (θ: 0.05-0.20)
  performance_fluctuation_threshold: 0.03  # 性能波动阈值 (φ: 0.01-0.10)
  exploration_interval: 100        # 定期探索间隔（批次数）
  
  # 安全参数
  memory_safety_margin_mb: 512     # 内存安全余量 (256-2048 MB)
  memory_usage_limit: 0.90         # 内存使用限制 (0.70-0.95)
  max_consecutive_failures: 3      # 最大连续失败次数 (2-5)
  fallback_level: "moderate"       # 回退级别 (none/minor/moderate/major/safe)
  
  # 性能权重（α + β + γ = 1.0）
  throughput_weight: 0.5           # 吞吐量权重 (α: 0.3-0.7)
  gpu_utilization_weight: 0.3      # GPU 利用率权重 (β: 0.2-0.5)
  memory_efficiency_weight: 0.2    # 内存效率权重 (γ: 0.1-0.3)
  
  # 性能指标
  min_acceptable_throughput: 10    # 最小可接受吞吐量 (tokens/sec)
  max_acceptable_latency_ms: 5000  # 最大可接受延迟 (ms)
  target_gpu_utilization: 0.80     # 目标 GPU 利用率 (0.60-0.95)
  
  # 高级参数
  enable_oscillation: true         # 是否启用震荡探索
  enable_auto_recovery: true       # 是否启用自动恢复
  recovery_sensitivity: 0.5        # 恢复灵敏度 (0.1-1.0)
  load_change_threshold: 0.30      # 负载变化阈值 (0.20-0.50)
  
  # 调试参数
  enable_debug_logging: false      # 是否启用调试日志
  metrics_history_size: 100        # 历史记录大小 (50-200)
  report_interval_sec: 10          # 状态报告间隔 (5-30 sec)
```

#### 6.1.2 参数说明与推荐值

**Batch Size 范围参数**:

| 参数 | 推荐值 | 范围 | 说明 |
|-----|--------|------|------|
| `min_batch_size` | 1 | 1-8 | 最小 batch size，确保即使在低负载下也能处理请求 |
| `max_batch_size` | 256 | 64-512 | 最大 batch size，防止内存溢出。根据 GPU 内存调整 |
| `initial_batch_size` | 8 | 4-16 | 初始 batch size，用于探测阶段的起点 |

**探测阶段参数**:

| 参数 | 推荐值 | 范围 | 说明 |
|-----|--------|------|------|
| `max_probing_attempts` | 10 | 5-20 | 最大探测次数。次数越多，探测越充分，但启动时间越长 |
| `probing_growth_factor` | 2.0 | 1.5-3.0 | 探测增长因子。2.0 表示每次翻倍 (1→2→4→8...) |
| `probing_timeout_ms` | 5000 | 3000-10000 | 探测超时时间，防止探测阶段卡住 |

**调整阶段参数**:

| 参数 | 推荐值 | 范围 | 说明 |
|-----|--------|------|------|
| `performance_threshold` | 0.05 | 0.01-0.20 | 性能变化阈值 (ε)。值越小，对性能变化越敏感 |
| `growth_coefficient` | 0.2 | 0.1-0.5 | 增长系数 (α)。0.2 表示每次增加 20% |
| `decay_coefficient` | 0.3 | 0.2-0.5 | 衰减系数 (β)。0.3 表示每次减少 30% |
| `oscillation_coefficient` | 0.05 | 0.01-0.10 | 震荡系数 (δ)。0.05 表示 ±5% 的随机波动 |
| `adjustment_interval_ms` | 100 | 50-500 | 调整间隔时间。太小会导致频繁调整 |

**稳定阶段参数**:

| 参数 | 推荐值 | 范围 | 说明 |
|-----|--------|------|------|
| `stabilization_cycles` | 10 | 5-20 | 稳定周期数 (N)。需要连续 N 个周期满足稳定条件 |
| `batch_size_change_threshold` | 0.10 | 0.05-0.20 | batch size 变化阈值 (θ)。变化 < 10% 认为稳定 |
| `performance_fluctuation_threshold` | 0.03 | 0.01-0.10 | 性能波动阈值 (φ)。波动 < 3% 认为稳定 |
| `exploration_interval` | 100 | 50-200 | 定期探索间隔。每处理 100 个批次后进行一次探索 |

**安全参数**:

| 参数 | 推荐值 | 范围 | 说明 |
|-----|--------|------|------|
| `memory_safety_margin_mb` | 512 | 256-2048 | 内存安全余量。保留足够的内存防止溢出 |
| `memory_usage_limit` | 0.90 | 0.70-0.95 | 内存使用限制。使用 90% 的内存时开始限制 |
| `max_consecutive_failures` | 3 | 2-5 | 最大连续失败次数。超过后触发安全回退 |
| `fallback_level` | "moderate" | none/minor/moderate/major/safe | 回退级别。"moderate" 表示中度回退 |

**性能权重参数**:

| 参数 | 推荐值 | 范围 | 说明 |
|-----|--------|------|------|
| `throughput_weight` | 0.5 | 0.3-0.7 | 吞吐量权重 (α)。首要优化目标 |
| `gpu_utilization_weight` | 0.3 | 0.2-0.5 | GPU 利用率权重 (β) |
| `memory_efficiency_weight` | 0.2 | 0.1-0.3 | 内存效率权重 (γ) |

### 6.2 配置加载与更新

#### 6.2.1 配置加载代码

> ✅ 当前实现使用 `Config::dynamicBatchTunerConfig()` 直接读取配置。  
> 下面旧式加载示例仅作为历史参考，不再作为实现依据。

```cpp
// scheduler.cpp
auto tunerConfig = Config::instance().dynamicBatchTunerConfig();
if (tunerConfig.enabled) {
    // strategy: static | dynamic | hybird
    // 参数由 Config 统一解析并提供默认值
}
```

```cpp
// config.cpp
#include "cllm/config/config.h"
#include "cllm/scheduler/dynamic_batch_tuner.h"

namespace cllm {

void Config::loadDynamicBatchTunerConfig(const YAML::Node& config) {
    if (!config["dynamic_batch_tuner"]) {
        CLLM_WARN("[Config] dynamic_batch_tuner config not found, using defaults");
        return;
    }
    
    const auto& tunerConfig = config["dynamic_batch_tuner"];
    
    // 基础开关
    if (tunerConfig["enabled"]) {
        dynamicBatchTunerEnabled_ = tunerConfig["enabled"].as<bool>();
    }
    
    // Batch Size 范围
    if (tunerConfig["min_batch_size"]) {
        minBatchSize_ = tunerConfig["min_batch_size"].as<size_t>();
    }
    
    if (tunerConfig["max_batch_size"]) {
        maxBatchSize_ = tunerConfig["max_batch_size"].as<size_t>();
    }
    
    if (tunerConfig["initial_batch_size"]) {
        initialBatchSize_ = tunerConfig["initial_batch_size"].as<size_t>();
    }
    
    // 探测阶段参数
    if (tunerConfig["max_probing_attempts"]) {
        maxProbingAttempts_ = tunerConfig["max_probing_attempts"].as<size_t>();
    }
    
    if (tunerConfig["probing_growth_factor"]) {
        probingGrowthFactor_ = tunerConfig["probing_growth_factor"].as<double>();
    }
    
    // 调整阶段参数
    if (tunerConfig["performance_threshold"]) {
        performanceThreshold_ = tunerConfig["performance_threshold"].as<double>();
    }
    
    if (tunerConfig["growth_coefficient"]) {
        growthCoefficient_ = tunerConfig["growth_coefficient"].as<double>();
    }
    
    if (tunerConfig["decay_coefficient"]) {
        decayCoefficient_ = tunerConfig["decay_coefficient"].as<double>();
    }
    
    // 稳定阶段参数
    if (tunerConfig["stabilization_cycles"]) {
        stabilizationCycles_ = tunerConfig["stabilization_cycles"].as<size_t>();
    }
    
    if (tunerConfig["batch_size_change_threshold"]) {
        batchSizeChangeThreshold_ = tunerConfig["batch_size_change_threshold"].as<double>();
    }
    
    // 安全参数
    if (tunerConfig["memory_safety_margin_mb"]) {
        memorySafetyMarginMb_ = tunerConfig["memory_safety_margin_mb"].as<size_t>();
    }
    
    if (tunerConfig["memory_usage_limit"]) {
        memoryUsageLimit_ = tunerConfig["memory_usage_limit"].as<double>();
    }
    
    // 性能权重
    if (tunerConfig["throughput_weight"]) {
        throughputWeight_ = tunerConfig["throughput_weight"].as<double>();
    }
    
    if (tunerConfig["gpu_utilization_weight"]) {
        gpuUtilizationWeight_ = tunerConfig["gpu_utilization_weight"].as<double>();
    }
    
    if (tunerConfig["memory_efficiency_weight"]) {
        memoryEfficiencyWeight_ = tunerConfig["memory_efficiency_weight"].as<double>();
    }
    
    CLLM_INFO("[Config] Dynamic batch tuner config loaded: enabled=%s, min_batch_size=%zu, max_batch_size=%zu",
              dynamicBatchTunerEnabled_ ? "true" : "false",
              minBatchSize_, maxBatchSize_);
}

void Config::updateDynamicBatchTunerConfig(const DynamicBatchTuner::TunerConfig& config) {
    // 运行时更新配置
    std::lock_guard<std::mutex> lock(configMutex_);
    
    // 更新配置参数
    minBatchSize_ = config.minBatchSize;
    maxBatchSize_ = config.maxBatchSize;
    initialBatchSize_ = config.initialBatchSize;
    
    // ... 更新其他参数 ...
    
    CLLM_INFO("[Config] Dynamic batch tuner config updated at runtime");
}

} // namespace cllm
```

#### 6.2.2 配置验证

```cpp
bool Config::validateDynamicBatchTunerConfig() const {
    // 验证参数范围
    if (minBatchSize_ > maxBatchSize_) {
        CLLM_ERROR("[Config] minBatchSize (%zu) > maxBatchSize (%zu)", 
                   minBatchSize_, maxBatchSize_);
        return false;
    }
    
    if (initialBatchSize_ < minBatchSize_ || initialBatchSize_ > maxBatchSize_) {
        CLLM_ERROR("[Config] initialBatchSize (%zu) out of range [%zu, %zu]", 
                   initialBatchSize_, minBatchSize_, maxBatchSize_);
        return false;
    }
    
    if (probingGrowthFactor_ < 1.5 || probingGrowthFactor_ > 3.0) {
        CLLM_WARN("[Config] probingGrowthFactor (%.2f) outside recommended range [1.5, 3.0]", 
                  probingGrowthFactor_);
    }
    
    if (performanceThreshold_ < 0.01 || performanceThreshold_ > 0.20) {
        CLLM_WARN("[Config] performanceThreshold (%.2f) outside recommended range [0.01, 0.20]", 
                  performanceThreshold_);
    }
    
    if (growthCoefficient_ < 0.1 || growthCoefficient_ > 0.5) {
        CLLM_WARN("[Config] growthCoefficient (%.2f) outside recommended range [0.1, 0.5]", 
                  growthCoefficient_);
    }
    
    if (decayCoefficient_ < 0.2 || decayCoefficient_ > 0.5) {
        CLLM_WARN("[Config] decayCoefficient (%.2f) outside recommended range [0.2, 0.5]", 
                  decayCoefficient_);
    }
    
    if (memoryUsageLimit_ < 0.70 || memoryUsageLimit_ > 0.95) {
        CLLM_WARN("[Config] memoryUsageLimit (%.2f) outside recommended range [0.70, 0.95]", 
                  memoryUsageLimit_);
    }
    
    // 验证权重和为 1.0
    double weightSum = throughputWeight_ + gpuUtilizationWeight_ + memoryEfficiencyWeight_;
    if (std::abs(weightSum - 1.0) > 0.01) {
        CLLM_WARN("[Config] Performance weights sum to %.2f (should be 1.0)", weightSum);
    }
    
    CLLM_INFO("[Config] Dynamic batch tuner config validation passed");
    return true;
}
```

### 6.3 运行时配置更新接口

#### 6.3.1 HTTP API 接口

```cpp
// scheduler_http_api.cpp
#include "cllm/scheduler/scheduler.h"
#include "cllm/config/config.h"

namespace cllm {

class SchedulerHttpApi {
private:
    Scheduler* scheduler_;
    Config* config_;
    
public:
    // 获取调谐器状态
    crow::json::wvalue getTunerStatus() {
        crow::json::wvalue response;
        
        if (!scheduler_->batchTuner_) {
            response["error"] = "Dynamic batch tuner not enabled";
            return response;
        }
        
        auto* tuner = scheduler_->batchTuner_.get();
        
        response["enabled"] = true;
        response["phase"] = getPhaseName(tuner->getCurrentPhase());
        response["current_batch_size"] = static_cast<size_t>(tuner->getCurrentBatchSize());
        response["best_batch_size"] = tuner->getBestBatchSize();
        response["best_score"] = tuner->getBestScore();
        response["is_stabilized"] = tuner->isStabilized();
        response["consecutive_failures"] = tuner->getConsecutiveFailures();
        
        // 性能指标
        auto metrics = tuner->getLatestMetrics();
        response["throughput"] = metrics.throughput;
        response["gpu_utilization"] = metrics.gpuUtilization;
        response["memory_utilization"] = metrics.memoryUtilization;
        response["processing_time_ms"] = metrics.processingTimeMs;
        
        // GPU 信息
        auto gpuInfo = tuner->getGPUInfo();
        response["gpu_model"] = gpuInfo.model;
        response["gpu_memory_mb"] = gpuInfo.totalMemoryMb;
        
        return response;
    }
    
    // 更新调谐器配置
    crow::json::wvalue updateTunerConfig(const crow::json::rvalue& request) {
        crow::json::wvalue response;
        
        if (!scheduler_->batchTuner_) {
            response["error"] = "Dynamic batch tuner not enabled";
            return response;
        }
        
        try {
            DynamicBatchTuner::TunerConfig config = tuner->getConfig();
            
            // 更新请求中的参数
            if (request.has("min_batch_size")) {
                config.minBatchSize = request["min_batch_size"].i();
            }
            
            if (request.has("max_batch_size")) {
                config.maxBatchSize = request["max_batch_size"].i();
            }
            
            if (request.has("performance_threshold")) {
                config.performanceThreshold = request["performance_threshold"].d();
            }
            
            if (request.has("growth_coefficient")) {
                config.growthCoefficient = request["growth_coefficient"].d();
            }
            
            if (request.has("decay_coefficient")) {
                config.decayCoefficient = request["decay_coefficient"].d();
            }
            
            // 验证配置
            if (!validateConfigUpdate(config)) {
                response["error"] = "Invalid configuration";
                return response;
            }
            
            // 更新配置
            tuner->updateConfig(config);
            
            response["success"] = true;
            response["message"] = "Configuration updated successfully";
            response["new_config"] = serializeConfig(config);
            
            CLLM_INFO("[SchedulerHttpApi] Tuner config updated at runtime");
            
        } catch (const std::exception& e) {
            response["error"] = std::string("Failed to update config: ") + e.what();
            CLLM_ERROR("[SchedulerHttpApi] Failed to update tuner config: %s", e.what());
        }
        
        return response;
    }
    
    // 重置调谐器
    crow::json::wvalue resetTuner() {
        crow::json::wvalue response;
        
        if (!scheduler_->batchTuner_) {
            response["error"] = "Dynamic batch tuner not enabled";
            return response;
        }
        
        try {
            scheduler_->batchTuner_->resetToSafeState();
            
            response["success"] = true;
            response["message"] = "Tuner reset to safe state";
            
            CLLM_INFO("[SchedulerHttpApi] Tuner reset via HTTP API");
            
        } catch (const std::exception& e) {
            response["error"] = std::string("Failed to reset tuner: ") + e.what();
        }
        
        return response;
    }
    
    // 手动设置 batch size
    crow::json::wvalue setBatchSize(const crow::json::rvalue& request) {
        crow::json::wvalue response;
        
        if (!request.has("batch_size")) {
            response["error"] = "batch_size parameter required";
            return response;
        }
        
        size_t batchSize = request["batch_size"].i();
        
        if (!scheduler_->batchTuner_) {
            response["error"] = "Dynamic batch tuner not enabled";
            return response;
        }
        
        try {
            scheduler_->batchTuner_->setBatchSize(batchSize);
            
            response["success"] = true;
            response["message"] = "Batch size set successfully";
            response["batch_size"] = batchSize;
            
            CLLM_INFO("[SchedulerHttpApi] Batch size set to %zu via HTTP API", batchSize);
            
        } catch (const std::exception& e) {
            response["error"] = std::string("Failed to set batch size: ") + e.what();
        }
        
        return response;
    }
    
private:
    std::string getPhaseName(DynamicBatchTuner::TuningPhase phase) {
        switch (phase) {
            case DynamicBatchTuner::TuningPhase::INITIAL_PROBING:
                return "INITIAL_PROBING";
            case DynamicBatchTuner::TuningPhase::DYNAMIC_ADJUSTMENT:
                return "DYNAMIC_ADJUSTMENT";
            case DynamicBatchTuner::TuningPhase::STABLE_RUNNING:
                return "STABLE_RUNNING";
            default:
                return "UNKNOWN";
        }
    }
    
    bool validateConfigUpdate(const DynamicBatchTuner::TunerConfig& config) {
        if (config.minBatchSize > config.maxBatchSize) {
            return false;
        }
        
        if (config.performanceThreshold < 0.01 || config.performanceThreshold > 0.20) {
            return false;
        }
        
        return true;
    }
    
    crow::json::wvalue serializeConfig(const DynamicBatchTuner::TunerConfig& config) {
        crow::json::wvalue json;
        json["min_batch_size"] = config.minBatchSize;
        json["max_batch_size"] = config.maxBatchSize;
        json["initial_batch_size"] = config.initialBatchSize;
        json["performance_threshold"] = config.performanceThreshold;
        json["growth_coefficient"] = config.growthCoefficient;
        json["decay_coefficient"] = config.decayCoefficient;
        return json;
    }
};

} // namespace cllm
```

#### 6.3.2 API 使用示例

```bash
# 1. 获取调谐器状态
curl -X GET http://localhost:8080/api/scheduler/tuner/status

响应示例:
{
  "enabled": true,
  "phase": "STABLE_RUNNING",
  "current_batch_size": 16,
  "best_batch_size": 16,
  "best_score": 0.85,
  "is_stabilized": true,
  "consecutive_failures": 0,
  "throughput": 132.73,
  "gpu_utilization": 0.75,
  "memory_utilization": 0.68,
  "processing_time_ms": 125.5,
  "gpu_model": "Apple M3 Pro",
  "gpu_memory_mb": 36864
}

# 2. 更新调谐器配置
curl -X POST http://localhost:8080/api/scheduler/tuner/config \
  -H "Content-Type: application/json" \
  -d '{
    "min_batch_size": 4,
    "max_batch_size": 128,
    "performance_threshold": 0.03,
    "growth_coefficient": 0.15,
    "decay_coefficient": 0.25
  }'

响应示例:
{
  "success": true,
  "message": "Configuration updated successfully",
  "new_config": {
    "min_batch_size": 4,
    "max_batch_size": 128,
    "initial_batch_size": 8,
    "performance_threshold": 0.03,
    "growth_coefficient": 0.15,
    "decay_coefficient": 0.25
  }
}

# 3. 重置调谐器
curl -X POST http://localhost:8080/api/scheduler/tuner/reset

响应示例:
{
  "success": true,
  "message": "Tuner reset to safe state"
}

# 4. 手动设置 batch size
curl -X POST http://localhost:8080/api/scheduler/tuner/batch-size \
  -H "Content-Type: application/json" \
  -d '{"batch_size": 32}'

响应示例:
{
  "success": true,
  "message": "Batch size set successfully",
  "batch_size": 32
}
```

### 6.4 配置文件示例

#### 6.4.1 针对不同 GPU 的推荐配置

**配置 1: NVIDIA A100 40GB（高性能场景）**

```yaml
dynamic_batch_tuner:
  enabled: true
  
  min_batch_size: 4
  max_batch_size: 256
  initial_batch_size: 16
  
  max_probing_attempts: 15
  probing_growth_factor: 2.5
  
  performance_threshold: 0.03
  growth_coefficient: 0.25
  decay_coefficient: 0.35
  
  stabilization_cycles: 15
  batch_size_change_threshold: 0.08
  performance_fluctuation_threshold: 0.02
  
  memory_safety_margin_mb: 2048
  memory_usage_limit: 0.95
  
  throughput_weight: 0.6
  gpu_utilization_weight: 0.3
  memory_efficiency_weight: 0.1
```

**配置 2: Apple M3 Pro 36GB（标准场景）**

```yaml
dynamic_batch_tuner:
  enabled: true
  
  min_batch_size: 2
  max_batch_size: 64
  initial_batch_size: 8
  
  max_probing_attempts: 10
  probing_growth_factor: 2.0
  
  performance_threshold: 0.05
  growth_coefficient: 0.20
  decay_coefficient: 0.30
  
  stabilization_cycles: 10
  batch_size_change_threshold: 0.10
  performance_fluctuation_threshold: 0.03
  
  memory_safety_margin_mb: 512
  memory_usage_limit: 0.90
  
  throughput_weight: 0.5
  gpu_utilization_weight: 0.3
  memory_efficiency_weight: 0.2
```

**配置 3: NVIDIA T4 16GB（入门场景）**

```yaml
dynamic_batch_tuner:
  enabled: true
  
  min_batch_size: 1
  max_batch_size: 32
  initial_batch_size: 4
  
  max_probing_attempts: 8
  probing_growth_factor: 1.8
  
  performance_threshold: 0.08
  growth_coefficient: 0.15
  decay_coefficient: 0.25
  
  stabilization_cycles: 8
  batch_size_change_threshold: 0.15
  performance_fluctuation_threshold: 0.05
  
  memory_safety_margin_mb: 1024
  memory_usage_limit: 0.85
  
  throughput_weight: 0.4
  gpu_utilization_weight: 0.4
  memory_efficiency_weight: 0.2
```

**配置 4: 禁用动态调整（兼容模式）**

```yaml
dynamic_batch_tuner:
  enabled: false
  
  # 使用静态配置
  min_batch_size: 8
  max_batch_size: 8
  initial_batch_size: 8
```

---

## 7. 实现路线图

### 7.1 分阶段实现计划

#### Phase 1: 基础框架（1-2 周）

**目标**: 实现核心功能，能够运行

**任务列表**:
1. ✅ 设计完成（本文档）
2. 创建 `dynamic_batch_tuner.h` 和 `dynamic_batch_tuner.cpp`
3. 实现 GPU 信息探测功能
4. 实现初始探测阶段（简化版）
5. 集成到 `scheduler.cpp`
6. 实现基础配置加载
7. 编写单元测试

**验收标准**:
- 能够正确识别 GPU 型号和内存
- 能够完成初始探测，找到可行的 batch size
- 系统能够正常运行，无崩溃

#### Phase 2: 动态调整（2-3 周）

**目标**: 实现完整的自适应调整算法

**任务列表**:
1. 实现动态调整阶段（爬山算法）
2. 实现稳定运行阶段
3. 实现性能指标收集和分析
4. 实现状态转换逻辑
5. 优化调整算法
6. 编写集成测试

**验收标准**:
- 能够适应负载变化，自动调整 batch size
- 能够进入稳定阶段，保持最优 batch size
- 吞吐量提升 ≥ 10%（相对于静态配置）

#### Phase 3: 安全机制（1-2 周）

**目标**: 完善安全机制，确保系统稳定

**任务列表**:
1. 实现边界检查
2. 实现异常处理（内存溢出、性能异常）
3. 实现多级回退策略
4. 实现自动恢复机制
5. 编写压力测试和故障注入测试

**验收标准**:
- 内存溢出时能够自动回退，不崩溃
- 连续失败时能够进入安全模式
- 故障恢复后能够自动恢复性能

#### Phase 4: 配置与监控（1 周）

**目标**: 提供完善的配置接口和监控能力

**任务列表**:
1. 实现完整的配置参数体系
2. 实现 HTTP API 接口
3. 实现实时监控和状态报告
4. 实现运行时配置更新
5. 编写 API 文档

**验收标准**:
- 能够通过配置文件灵活调整参数
- 能够通过 HTTP API 查询状态和更新配置
- 能够实时监控调谐器运行状态

#### Phase 5: 优化与验证（2-3 周）

**目标**: 优化性能，完成多 GPU 验证

**任务列表**:
1. 优化算法性能（降低开销 < 5%）
2. 在多种 GPU 上测试（NVIDIA、AMD、Apple、Intel）
3. 性能对比测试（静态 vs 动态）
4. 编写性能报告
5. 文档完善

**验收标准**:
- 系统开销 < 5%
- 吞吐量提升 ≥ 20%（相对于最优静态配置）
- 支持 ≥ 5 种 GPU 型号
- 完整的测试报告和文档

### 7.2 关键里程碑

| 里程碑 | 时间 | 交付物 | 验收标准 |
|-------|------|--------|--------|
| **M1: 设计完成** | Day 3 | 设计文档 | 评审通过 |
| **M2: 基础框架完成** | Day 10 | 可运行代码 | 能够完成初始探测 |
| **M3: 动态调整完成** | Day 24 | 完整算法实现 | 吞吐量提升 ≥ 10% |
| **M4: 安全机制完成** | Day 38 | 稳定的系统 | 无崩溃，自动恢复 |
| **M5: 配置监控完成** | Day 45 | API 和文档 | 可配置、可监控 |
| **M6: 优化验证完成** | Day 60 | 生产就绪版本 | 吞吐量提升 ≥ 20%，支持多 GPU |

### 7.3 技术风险与应对

| 风险 | 影响 | 应对措施 |
|-----|------|--------|
| **GPU 信息探测失败** | 无法正确估算 batch size | 提供默认配置，降级到静态模式 |
| **调整算法振荡** | batch size 频繁波动 | 增加阻尼系数，延长稳定周期 |
| **内存估算不准确** | 可能导致溢出 | 保守估算，预留充足余量 |
| **性能指标收集开销大** | 影响系统性能 | 采样收集，降低频率 |
| **跨 GPU 兼容性问题** | 某些 GPU 无法工作 | 抽象 GPU 接口，提供平台特定实现 |
| **与现有调度系统冲突** | 调度逻辑混乱 | 仔细设计集成点，充分测试 |

---

## 8. 总结

### 8.1 设计亮点

1. **三阶段自适应算法**
   - 初始探测阶段：快速找到可行范围
   - 动态调整阶段：持续优化适应变化
   - 稳定运行阶段：保持最优避免波动

2. **多维度性能评估**
   - 吞吐量、GPU 利用率、内存效率三维评估
   - 可配置的权重系数，适应不同场景
   - 量化的性能评分体系

3. **完善的安全机制**
   - 多级回退策略（minor → moderate → major → safe）
   - 自动恢复机制
   - 边界检查和异常处理

4. **灵活的配置接口**
   - 丰富的配置参数（30+）
   - 运行时更新能力
   - HTTP API 接口

5. **跨 GPU 平台兼容性**
   - 支持 NVIDIA、AMD、Apple、Intel GPU
   - 自动探测 GPU 信息
   - 平台特定优化

### 8.2 预期收益

| 指标 | 静态配置 | 动态调整 | 提升 |
|-----|---------|---------|------|
| **吞吐量** | 80-120 t/s | 100-160 t/s | **+20-40%** |
| **GPU 利用率** | 50-70% | 70-90% | **+20-30%** |
| **内存效率** | 60-80% | 80-95% | **+20-25%** |
| **收敛时间** | N/A | < 2 min | - |
| **系统开销** | 0% | < 5% | - |

### 8.3 后续工作

1. **立即开始**
   - Phase 1: 基础框架实现
   - 预计 1-2 周

2. **并行进行**
   - 测试环境准备（多种 GPU）
   - 测试用例设计
   - 性能基准测试

3. **长期优化**
   - 支持多 GPU 协同
   - 支持模型动态切换
   - 支持负载预测

---

## 附录

### A. 参考文献

1. **Roofline Model**: Williams et al., "Roofline: An Insightful Visual Performance Model for Floating-Point Programs and Multicore Architectures", 2009
2. **Amdahl's Law**: Gene Amdahl, "Validity of the single processor approach to achieving large scale computing capabilities", 1967
3. **Hill-Climbing Algorithm**: Stuart Russell and Peter Norvig, "Artificial Intelligence: A Modern Approach", 2010
4. **Llama.cpp Batch Processing**: https://github.com/ggerganov/llama.cpp
5. **CUDA Best Practices Guide**: NVIDIA Corporation

### B. 术语表

| 术语 | 英文 | 说明 |
|-----|------|------|
| **Batch Size** | Batch Size | 批处理大小，一次推理处理的请求数 |
| **吞吐量** | Throughput | 每秒处理的 tokens 数 |
| **GPU 利用率** | GPU Utilization | GPU 计算单元的占用率 |
| **内存带宽** | Memory Bandwidth | GPU 内存的数据传输速率 |
| **调谐器** | Tuner | 动态调整 batch size 的组件 |
| **探测阶段** | Probing Phase | 系统启动时的 batch size 探索阶段 |
| **调整阶段** | Adjustment Phase | 运行时的 batch size 优化阶段 |
| **稳定阶段** | Stable Phase | 保持最优 batch size 的阶段 |

### C. 联系方式

- **技术负责人**: cLLM Technical Team
- **文档版本**: v1.0
- **更新日期**: 2026-01-22
- **反馈邮箱**: tech@cllm.ai

---

**文档结束**

*本文档描述了 cLLM 动态 Batch Size 调整机制的完整设计方案。*  
*所有代码示例均为伪代码，实际实现可能会有所不同。*  
*本设计方案遵循 cLLM 项目的架构原则和编码规范。*