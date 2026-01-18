# Phase 0: 现状评估报告

## 文档概述
本文档记录 Phase 0（现状评估与准备）阶段的审查结果，包括当前实现状态、已实现功能、待实现功能、已知问题和限制。

**审查日期**：2026-01-18  
**审查范围**：Scheduler、RequestQueue、llama.cpp后端集成、配置系统

---

## 一、代码审查结果

### 1.1 RequestState 结构审查

**文件位置**：`include/cllm/common/request_state.h`

**已实现字段**：
- ✅ `requestId` - 请求ID
- ✅ `tokenizedPrompt` - token化后的输入提示
- ✅ `generatedTokens` - 已生成的token序列
- ✅ `startTime` - 开始处理时间（**已存在，可用于超时检测**）
- ✅ `arrivalTime` - 请求到达时间
- ✅ `completionTime` - 完成时间
- ✅ `isCompleted` - 是否已完成（bool）
- ✅ `isRunning` - 是否正在运行（bool）
- ✅ `isFailed` - 是否失败（bool）
- ❌ `isTimeout` - 是否超时（**待添加，或使用startTime+requestTimeout判断**）

**状态判断逻辑**（使用现有字段组合）：
- **PENDING**：`!isCompleted && !isFailed && !isRunning`
- **PROCESSING**：`isRunning && !isCompleted && !isFailed`
- **COMPLETED**：`isCompleted`
- **FAILED**：`isFailed`
- **TIMEOUT**：需要新增 `isTimeout` 字段，或通过 `startTime` + `requestTimeout` 判断

**建议**：
- 保持现有 bool 字段组合，不添加 `RequestStatus` 枚举（向后兼容）
- Phase 1 中考虑添加 `isTimeout` 字段，或使用时间戳判断

---

### 1.2 Scheduler 实现审查

**文件位置**：
- `src/scheduler/scheduler.cpp`
- `include/cllm/scheduler/scheduler.h`

**已实现功能**：
- ✅ `runningRequests_` - 运行中的请求映射（`std::map<size_t, RequestState>`）
- ✅ `completedRequests_` - 已完成的请求映射（`std::map<size_t, RequestState>`）
- ✅ `processBatch` - 批处理逻辑（已有状态合并和过滤）
- ✅ `getRunningRequests()` - 获取运行中的请求（**需验证是否过滤已完成请求**）
- ✅ `processRequests()` - 请求处理入口（调用 `formBatch` 和 `processBatch`）
- ✅ `schedulerLoop` - 调度器主循环（事件驱动，使用 `condition_variable`）
- ✅ `queueCondition_` - 队列条件变量（用于事件驱动）

**当前实现逻辑**：
1. `processRequests()` 调用 `requestQueue_.getPendingRequests()` 获取待处理请求
2. `batchManager_.formBatch(pending, running)` 形成批处理（可能包含来自 `RequestQueue` 的新请求）
3. `processBatch` 中，新请求会被添加到 `runningRequests_`（如果不存在）
4. 请求完成后，从 `runningRequests_` 移动到 `completedRequests_`

**待实现功能**：
- ❌ 状态机完整实现（PENDING → PROCESSING → COMPLETED）
- ❌ 状态查询和过滤（`getRunningRequests()` 需要过滤已完成请求）
- ❌ 超时检测机制
- ❌ 请求从 `RequestQueue` 到 `runningRequests_` 的自动流转（当前通过 `formBatch` 间接实现）

**已知问题**：
- `getRunningRequests()` 可能返回已完成的请求（需要验证并修复）
- 状态转换逻辑不够明确（PENDING → PROCESSING 的转换时机不清晰）

---

### 1.3 RequestQueue 实现审查

**文件位置**：
- `src/common/queue.cpp`
- `include/cllm/common/queue.h`

**已实现功能**：
- ✅ `addRequest()` - 添加请求到队列
- ✅ `getPendingRequests()` - 获取待处理请求
- ✅ `getQueueSize()` - 获取队列大小
- ✅ 优先级队列支持

**当前实现逻辑**：
- 使用优先级队列管理待处理请求
- `getPendingRequests()` 返回所有待处理请求（可能包括已在 `runningRequests_` 中的请求）

**待实现功能**：
- ❌ 请求从 `RequestQueue` 到 `runningRequests_` 的自动流转机制
- ❌ 并发限制检查（`maxConcurrentRequests`）

**已知问题**：
- `getPendingRequests()` 可能返回已在 `runningRequests_` 中的请求（需要 `formBatch` 去重）

---

### 1.4 llama.cpp 后端集成审查

**文件位置**：`src/inference/llama_cpp_backend.cpp`

**已实现功能**：
- ✅ `forwardBatch` - 批处理推理
- ✅ `nextSeqId_` - 下一个可用的seq_id（循环分配）
- ✅ `seqPositions_` - seq_id到位置的映射（`std::map<int32_t, size_t>`）
- ✅ `seqLengths_` - seq_id到长度的映射（`std::map<int32_t, size_t>`）
- ✅ `clearKVCacheForSequence` - 清理特定序列的KV缓存
- ✅ `createContextParams` - 创建context参数（**需确认是否读取配置的n_seq_max**）

**当前实现问题**：
- ❌ **关键问题**：使用批处理索引（0,1,2...）作为 `seq_id`，而不是基于 requestId 的映射
  - 不同请求可能使用相同的 `seq_id`，导致KV缓存混乱
  - 当前实现：`seq_id = 批处理索引`，而不是 `seq_id = requestId的映射`
- ❌ **缺少**：requestId → seqId 的映射关系
- ❌ **缺少**：序列ID池管理（可用ID池、分配/释放逻辑）

**待实现功能**（Phase 2）：
- ❌ 建立 requestId → seqId 映射
- ❌ 序列ID分配和释放机制
- ❌ 序列ID池管理（0到n_seq_max-1）

**已知问题**：
- 序列ID重用可能导致KV缓存混乱（11个请求后卡住的根本原因）

---

## 二、基础设施准备

### 2.1 配置系统审查

**文件位置**：
- `src/common/config.cpp`
- `include/cllm/common/config.h`
- `config/config.yaml.template`

**已实现配置**：
- ✅ `backendLlamaCppNSeqMax()` - 读取 `backend.llama_cpp.n_seq_max`（默认值：1，范围：1-256）
- ✅ `config.yaml.template` 中已有 `n_seq_max` 配置项（默认值：8）

**配置读取实现**：
```cpp
// src/common/config.cpp:316
int Config::backendLlamaCppNSeqMax() const {
    std::lock_guard<std::mutex> lock(mutex_);
    int nSeqMax = config_["backend"]["llama_cpp"]["n_seq_max"].as<int>(1);
    // 验证范围：1-256
    if (nSeqMax < 1) {
        CLLM_WARN("backend.llama_cpp.n_seq_max (%d) is less than 1, using default value 1", nSeqMax);
        nSeqMax = 1;
    }
    if (nSeqMax > 256) {
        CLLM_WARN("backend.llama_cpp.n_seq_max (%d) exceeds LLAMA_MAX_SEQ (256), using 256", nSeqMax);
        nSeqMax = 256;
    }
    return nSeqMax;
}
```

**配置模板**：
```yaml
# config/config.yaml.template:133
llama_cpp:
  n_seq_max: 8  # llama.cpp最大序列数（定义支持的最大并行序列数）
```

**待确认**：
- ⚠️ `LlamaCppBackend::createContextParams` 是否已使用 `Config::backendLlamaCppNSeqMax()`？
  - 需要验证是否仍使用硬编码值

---

### 2.2 测试框架准备

**当前状态**：
- ✅ 单元测试框架已存在（`tests/` 目录）
- ✅ 集成测试框架已存在（`tests/integration/` 目录）

**待准备**：
- ⚠️ 需要创建 Phase 1-7 的测试用例清单

---

## 三、已实现功能清单

### 3.1 核心功能

| 功能 | 状态 | 位置 | 备注 |
|------|------|------|------|
| RequestState 结构 | ✅ 已实现 | `include/cllm/common/request_state.h` | 包含所有必要字段（除 `isTimeout`） |
| Scheduler 基础实现 | ✅ 已实现 | `src/scheduler/scheduler.cpp` | 包含 `runningRequests_`, `completedRequests_` |
| RequestQueue 实现 | ✅ 已实现 | `src/common/queue.cpp` | 优先级队列支持 |
| llama.cpp 后端集成 | ✅ 已实现 | `src/inference/llama_cpp_backend.cpp` | 基础推理功能 |
| n_seq_max 配置读取 | ✅ 已实现 | `src/common/config.cpp` | `Config::backendLlamaCppNSeqMax()` |

### 3.2 状态管理

| 功能 | 状态 | 位置 | 备注 |
|------|------|------|------|
| isCompleted/isFailed/isRunning 字段 | ✅ 已实现 | `RequestState` | 可用于状态判断 |
| startTime 字段 | ✅ 已实现 | `RequestState` | 可用于超时检测 |
| runningRequests_ 管理 | ✅ 已实现 | `Scheduler` | 需要完善状态转换逻辑 |
| completedRequests_ 管理 | ✅ 已实现 | `Scheduler` | 基本功能已实现 |

### 3.3 批处理逻辑

| 功能 | 状态 | 位置 | 备注 |
|------|------|------|------|
| formBatch 逻辑 | ✅ 已实现 | `BatchManager` | 形成批处理 |
| processBatch 逻辑 | ✅ 已实现 | `Scheduler` | 批处理执行 |
| 状态合并和过滤 | ✅ 已实现 | `processBatch` | 避免重复处理 |

---

## 四、待实现功能清单

### 4.1 Phase 1: 状态机核心实现

| 功能 | 优先级 | 位置 | 备注 |
|------|--------|------|------|
| 状态判断逻辑（使用现有字段组合） | P0 | `Scheduler` | 明确 PENDING/PROCESSING/COMPLETED 判断 |
| PENDING → PROCESSING 转换 | P0 | `processBatch` | 在批处理开始时标记 |
| PROCESSING → COMPLETED 转换 | P0 | `processBatch` | 在批处理完成时标记 |
| PROCESSING → FAILED 转换 | P0 | `processBatch` | 在批处理失败时标记 |
| `getRunningRequests()` 过滤已完成请求 | P0 | `Scheduler` | 只返回 PENDING/PROCESSING 请求 |
| 请求自动流转（RequestQueue → runningRequests_） | P0 | `processRequests` | 确保通过 `formBatch` 自动流转 |

### 4.2 Phase 2: 序列ID管理重构

| 功能 | 优先级 | 位置 | 备注 |
|------|--------|------|------|
| requestId → seqId 映射 | P0 | `LlamaCppBackend` | 替换批处理索引逻辑 |
| 序列ID分配机制 | P0 | `LlamaCppBackend` | 从可用池分配 |
| 序列ID释放机制 | P0 | `LlamaCppBackend` | 请求完成时释放 |
| 序列ID池管理（0到n_seq_max-1） | P0 | `LlamaCppBackend` | 可用ID池 |

### 4.3 Phase 3-7: 其他功能

| 功能 | 优先级 | 阶段 | 备注 |
|------|--------|------|------|
| 超时检测机制 | P1 | Phase 3 | 定期扫描PROCESSING请求 |
| KV缓存统计管理 | P1 | Phase 4 | 统计信息和清理协调 |
| KV缓存LRU淘汰 | P2 | Phase 5 | LRU淘汰策略 |
| HTTP层并发检查 | P3 | Phase 6 | 可选优化 |
| 响应回调机制 | P2 | Phase 7 | 回调链路 |

---

## 五、已知问题和限制

### 5.1 关键问题

1. **序列ID重用冲突**（Phase 2 修复）
   - **问题**：使用批处理索引作为 `seq_id`，导致不同请求使用相同的 `seq_id`
   - **影响**：KV缓存混乱，可能导致11个请求后卡住
   - **解决方案**：建立 requestId → seqId 映射

2. **状态转换逻辑不明确**（Phase 1 修复）
   - **问题**：PENDING → PROCESSING 的转换时机不清晰
   - **影响**：状态管理不一致
   - **解决方案**：在 `processBatch` 开始时明确标记 PROCESSING

3. **`getRunningRequests()` 可能返回已完成请求**（Phase 1 修复）
   - **问题**：没有过滤 COMPLETED/TIMEOUT/FAILED 请求
   - **影响**：`formBatch` 可能高估 `runningLength`
   - **解决方案**：过滤已完成请求

### 5.2 限制

1. **状态管理**：使用 bool 字段组合，不如枚举清晰（但保持向后兼容）
2. **KV缓存管理**：由 `llama.cpp` 内部管理，我们只管理统计信息
3. **超时检测**：需要定期扫描，可能影响性能

---

## 六、测试用例清单

### 6.1 Phase 1 测试用例

1. ✅ 测试状态转换：PENDING → PROCESSING → COMPLETED
2. ✅ 测试状态转换：PENDING → PROCESSING → FAILED
3. ✅ 测试 `getRunningRequests()` 过滤已完成请求
4. ✅ 测试请求从 `RequestQueue` 自动流转到 `runningRequests_`（通过 `formBatch`）
5. ✅ 测试状态一致性：多线程同时更新请求状态

### 6.2 Phase 2 测试用例

1. ✅ 测试序列ID分配（单个请求）
2. ✅ 测试序列ID释放（单个请求）
3. ✅ 测试序列ID池耗尽（n_seq_max个请求同时处理）
4. ✅ 测试序列ID分配失败（所有seq_id都被占用时，新请求保持PENDING）
5. ✅ 测试序列ID重用（请求完成后，seq_id可以被新请求使用）
6. ✅ 测试并发安全性（多线程同时分配/释放）

### 6.3 Phase 3-7 测试用例

（待 Phase 3-7 阶段定义）

---

## 七、下一步行动

### 7.1 立即开始（Phase 1）

1. **明确状态判断逻辑**：使用现有字段组合定义状态
2. **实现状态转换**：PENDING → PROCESSING → COMPLETED
3. **修复 `getRunningRequests()`**：过滤已完成请求
4. **验证请求流转**：确保 RequestQueue → runningRequests_ 自动流转

### 7.2 待确认

1. ⚠️ `LlamaCppBackend::createContextParams` 是否已使用 `Config::backendLlamaCppNSeqMax()`？
2. ⚠️ 是否需要添加 `isTimeout` 字段，或使用时间戳判断？

---

## 八、总结

### 8.1 当前状态

- ✅ **基础功能完整**：Scheduler、RequestQueue、llama.cpp后端集成基本完成
- ✅ **配置系统就绪**：`n_seq_max` 配置读取已实现
- ⚠️ **状态管理待完善**：需要明确状态转换逻辑
- ❌ **序列ID管理需重构**：当前实现存在冲突风险

### 8.2 可立即开始

- ✅ Phase 0 已完成（本报告）
- ✅ Phase 1 可以开始（状态机核心实现）
- ⚠️ Phase 2 需要 Phase 1 部分完成（序列ID管理重构）

---

**报告日期**：2026-01-18  
**审查人**：实施计划审查  
**状态**：Phase 0 完成，准备开始 Phase 1
