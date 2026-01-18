# 并发请求管理方案 - 分阶段实施计划

## 文档概述

本文档基于 `concurrent_request_management_design.md` 制定分阶段实施计划，将完整的并发请求管理系统分解为可执行的阶段，每个阶段包含明确的目标、任务、验收标准和依赖关系。

## 审查说明

**审查日期**：2026-01-18  
**审查状态**：已完成技术可行性审查，已修复关键技术问题

### 主要修复内容

1. **Phase 1 - 状态管理方案**：
   - ✅ 明确使用现有 bool 字段组合，而不是添加枚举（保持向后兼容）
   - ✅ 确认 `startTime` 字段已存在，无需添加
   - ✅ 调整工作量估算：3-5天 → 4-6天

2. **Phase 2 - 序列ID管理**：
   - ✅ 重新定义为"重构序列ID管理"，而不是"创建新的管理器"
   - ✅ 明确当前问题：使用批处理索引作为 `seq_id` 导致冲突风险
   - ✅ 调整工作量估算：2-3天 → 3-4天
   - ✅ 明确依赖：部分依赖 Phase 1（基础功能可独立实现）

3. **Phase 4 - KV缓存管理**：
   - ✅ 重新定义为"KV缓存统计管理"，而不是"KV缓存存储管理"
   - ✅ 明确职责：管理统计信息，协调清理，不存储KV缓存数据
   - ✅ 明确清理方式：调用 `llama_memory_seq_rm` 清理 `llama.cpp` 缓存
   - ✅ 调整工作量估算：4-5天 → 3-4天
   - ✅ 添加依赖：Phase 2（需要序列ID管理器来获取 requestId 对应的 seqId）

4. **Phase 5 - LRU淘汰**：
   - ✅ 明确依赖：Phase 1（必需）+ Phase 4（必需）
   - ✅ 明确淘汰保护机制：维护 `requestStatus_` 映射或查询 `Scheduler`

5. **技术细节优化**：
   - ✅ 明确KV缓存条目数计算：序列总长度（`tokenizedPrompt.size() + generatedTokens.size()`）
   - ✅ 明确请求流转逻辑：通过 `formBatch` 间接实现，不是直接转移
   - ✅ 完善验收标准和测试用例

详细审查报告请参考：`implementation_plan_review.md`

## 实施原则

1. **自底向上**：先实现基础功能，再实现依赖基础功能的特性
2. **渐进增强**：每个阶段都能独立运行，逐步增加功能
3. **可验证性**：每个阶段都有明确的验收标准
4. **最小依赖**：尽量减少阶段间的依赖，允许并行开发

## 阶段总览

| 阶段 | 名称 | 核心目标 | 优先级 | 预估工作量 | 依赖 |
|------|------|----------|--------|------------|------|
| Phase 0 | 现状评估与准备 | 评估当前实现，准备基础设施 | P0 | 1-2天 | 无 |
| Phase 1 | 状态机核心实现 | 实现runningRequests_状态管理（PENDING→PROCESSING→COMPLETED） | P0 | 4-6天 | Phase 0 |
| Phase 2 | 序列ID管理重构 | 重构序列ID分配与释放机制（基于requestId映射） | P0 | 3-4天 | Phase 1（部分） |
| Phase 3 | 超时检测机制 | 实现请求超时检测与处理 | P1 | 3-4天 | Phase 1（必需）+ Phase 2（可选） |
| Phase 4 | KV缓存统计管理 | 实现基于requestId的KV缓存统计和清理协调 | P1 | 3-4天 | Phase 1（部分） |
| Phase 5 | KV缓存LRU淘汰 | 实现LRU淘汰策略 | P2 | 3-4天 | Phase 1（必需）+ Phase 4 |
| Phase 6 | HTTP层优化 | 实现HTTP层并发检查（可选） | P3 | 2-3天 | Phase 1 |
| Phase 7 | 响应回调机制 | 实现完整的响应回调链路 | P2 | 2-3天 | Phase 1, Phase 3 |

**总预估工作量**：20-29天（4-6周）

---

## Phase 0: 现状评估与准备

### 目标
评估当前实现状态，准备基础设施和测试环境。

### 任务清单

#### 0.1 代码审查
- [ ] 审查 `Scheduler` 当前实现
  - `src/scheduler/scheduler.cpp`
  - `include/cllm/scheduler/scheduler.h`
- [ ] 审查 `RequestQueue` 实现
  - `src/common/queue.cpp`
  - `include/cllm/common/queue.h`
- [ ] 审查 `llama.cpp` 后端集成
  - `src/inference/llama_cpp_backend.cpp`
- [ ] 识别已实现功能和待实现功能

#### 0.2 基础设施准备
- [ ] 确认 `n_seq_max` 配置读取已实现（`Config::backendLlamaCppNSeqMax()`）
- [ ] 确认配置参数已添加到 `config.yaml.template`
- [ ] 准备单元测试框架
- [ ] 准备集成测试环境

#### 0.3 文档准备
- [ ] 记录当前实现状态
- [ ] 识别已知问题和限制
- [ ] 创建测试用例清单

### 验收标准
- ✅ 完成代码审查，形成现状评估报告
- ✅ 识别出待实现功能清单
- ✅ 测试环境准备就绪

### 依赖
无

### 输出
- 现状评估报告
- 测试用例清单
- 配置参数文档

---

## Phase 1: 状态机核心实现

### 目标
实现 `runningRequests_` 的完整状态机，包括状态转换（PENDING→PROCESSING→COMPLETED）和基本状态管理。

### 任务清单

#### 1.1 状态管理方案（使用现有字段组合）
- [ ] 明确状态判断逻辑（使用现有 bool 字段组合）
  - **PENDING**：`!isCompleted && !isFailed && !isRunning`
  - **PROCESSING**：`isRunning && !isCompleted && !isFailed`
  - **COMPLETED**：`isCompleted`
  - **FAILED**：`isFailed`
  - **TIMEOUT**：需要新增 `isTimeout` 字段，或通过 `startTime` + `requestTimeout` 判断
- [ ] 更新 `RequestState` 结构体（如果需要）
  - **确认**：`startTime` 字段已存在（`include/cllm/common/request_state.h`）
  - **可选**：添加 `isTimeout` 字段（用于超时检测），或使用时间戳判断
- [ ] **注意**：不添加 `RequestStatus` 枚举（保持向后兼容，减少代码修改）

#### 1.2 状态转换逻辑
- [ ] 实现 PENDING → PROCESSING 转换
  - 在 `processBatch` 中，将批处理中的请求状态标记为 PROCESSING
  - 记录 `startTime`
- [ ] 实现 PROCESSING → COMPLETED 转换
  - 在 `processBatch` 完成后，将成功的请求标记为 COMPLETED
  - 移动到 `completedRequests_`
- [ ] 实现 PROCESSING → FAILED 转换
  - 在 `processBatch` 出错时，将失败的请求标记为 FAILED
  - 移动到 `completedRequests_`

#### 1.3 状态查询与过滤
- [ ] 更新 `getRunningRequests()` 方法
  - 只返回状态为 PENDING 或 PROCESSING 的请求
  - 过滤掉 COMPLETED/TIMEOUT/FAILED 的请求
- [ ] 确保状态一致性
  - 请求完成后立即从 `runningRequests_` 移除
  - 添加到 `completedRequests_`

#### 1.4 从RequestQueue到runningRequests_的流转（确保逻辑正确）
- [ ] **明确当前实现**：请求从 `RequestQueue` 到 `runningRequests_` 是通过 `formBatch` 间接实现的
  - `processRequests()` 调用 `requestQueue_.getPendingRequests()` 获取待处理请求
  - `batchManager_.formBatch(pending, running)` 形成批处理（可能包含来自 `RequestQueue` 的新请求）
  - `processBatch` 中，新请求会被添加到 `runningRequests_`（状态为PENDING或PROCESSING）
- [ ] **确保请求流转逻辑**：
  - 确保 `formBatch` 能够从 `RequestQueue` 中取出新请求（如果 `runningRequests_.size() < maxConcurrentRequests`）
  - 确保 `processBatch` 能够将新请求添加到 `runningRequests_`（如果不存在）
  - 确保请求完成后，能够从 `RequestQueue` 取出下一个请求（通过 `queueCondition_.notify_one()`）
- [ ] **验证自动流转**：
  - 请求完成后，检查 `RequestQueue` 是否为空
  - 如果不为空且 `runningRequests_.size() < maxConcurrentRequests`，自动触发下一个请求的处理

### 验收标准
- ✅ 状态判断逻辑明确（使用现有字段组合）
- ✅ PENDING → PROCESSING → COMPLETED 状态转换正常工作
- ✅ PROCESSING → FAILED 状态转换正常工作
- ✅ `getRunningRequests()` 只返回活跃请求（PENDING/PROCESSING）
- ✅ 请求完成后自动从 `runningRequests_` 移除
- ✅ 请求自动从 `RequestQueue` 流转到 `runningRequests_`（通过 `formBatch` 机制）
- ✅ 状态一致性：多线程环境下状态更新正确（无竞态条件）
- ✅ 单元测试通过（至少5个测试用例）

### 依赖
Phase 0

### 输出
- 状态机实现代码
- 单元测试代码
- 状态转换流程图（代码注释）

### 测试用例
1. 测试状态转换：PENDING → PROCESSING → COMPLETED
2. 测试状态转换：PENDING → PROCESSING → FAILED
3. 测试 `getRunningRequests()` 过滤已完成请求
4. 测试请求从 `RequestQueue` 自动流转到 `runningRequests_`（通过 `formBatch`）
5. 测试状态一致性：多线程同时更新请求状态

---

## Phase 2: 序列ID管理重构

### 目标
重构序列ID管理机制，建立基于 requestId 的序列ID分配与释放机制，替换当前基于批处理索引的实现。

### 当前问题
- **问题**：当前实现使用批处理索引（0,1,2...）作为 `seq_id`，导致不同请求可能使用相同的 `seq_id`，导致KV缓存混乱
- **解决方案**：建立 requestId → seqId 的映射关系，确保每个 requestId 对应唯一的 seqId

### 任务清单

#### 2.1 序列ID管理重构方案
- [ ] **方案A（推荐）**：在 `LlamaCppBackend` 中重构序列ID管理，添加 requestId → seqId 映射
  - 在 `LlamaCppBackend` 中添加 `std::map<size_t requestId, int32_t seqId>` 映射
  - 添加 `std::vector<int32_t> availableSeqIds` - 可用序列ID池（0到n_seq_max-1）
  - 添加 `std::mutex` - 保护并发访问
- [ ] **方案B（可选）**：创建独立的 `SequenceIdManager` 类
  - 位置：`src/inference/sequence_id_manager.cpp`
  - 头文件：`include/cllm/inference/sequence_id_manager.h`
  - 由 `LlamaCppBackend` 调用

#### 2.2 序列ID分配逻辑（重构现有代码）
- [ ] 重构 `allocateSequenceId(size_t requestId)` 方法（或在 `LlamaCppBackend` 中添加）
  - 从可用池中选取一个 `seq_id`（范围：0到n_seq_max-1）
  - 建立 requestId → seqId 映射（替换当前基于批处理索引的逻辑）
  - 如果池为空（所有seq_id都被占用），返回错误或保持PENDING状态等待
- [ ] 重构 `LlamaCppBackend::forwardBatch`
  - **替换**：当前基于批处理索引的 `seq_id` 分配逻辑
  - **改为**：基于 requestId 查询对应的 `seq_id`（从映射中查找）
  - 如果是新请求（首次分配），调用 `allocateSequenceId(requestId)`
  - 使用分配的 `seq_id` 调用 `llama.cpp` API

#### 2.3 序列ID释放逻辑
- [ ] 实现 `releaseSequenceId(size_t requestId)` 方法
  - 查找 requestId 对应的 seqId
  - 删除 requestId → seqId 映射
  - 将 seq_id 返回可用池
- [ ] 集成到请求完成流程
  - 在请求 COMPLETED/TIMEOUT/FAILED 时调用 `releaseSequenceId`
  - 在 `processBatch` 完成后释放序列ID

#### 2.4 序列ID池初始化
- [ ] 从配置读取 `n_seq_max`
  - 使用 `Config::instance().backendLlamaCppNSeqMax()`
  - 初始化可用池：0 到 n_seq_max-1
- [ ] 确保 `llama.cpp` 的 `n_seq_max` 配置一致
  - 在 `LlamaCppBackend::createContextParams` 中设置相同的值

### 验收标准
- ✅ 序列ID管理重构完成（requestId → seqId 映射）
- ✅ 序列ID分配逻辑正确（循环分配，支持最大n_seq_max个并发）
- ✅ 序列ID释放逻辑正确（请求完成后自动释放）
- ✅ 序列ID池大小从配置读取
- ✅ 重构 `LlamaCppBackend::forwardBatch`，使用基于 requestId 的 `seq_id`
- ✅ 替换原有基于批处理索引的 `seq_id` 逻辑
- ✅ 单元测试通过（至少6个测试用例）

### 依赖
Phase 1（需要状态机来确定请求完成时机）

### 输出
- `SequenceIdManager` 实现代码
- 集成到 `LlamaCppBackend` 的代码修改
- 单元测试代码

### 测试用例
1. 测试序列ID分配（单个请求）
2. 测试序列ID释放（单个请求）
3. 测试序列ID池耗尽（n_seq_max个请求同时处理）
4. 测试序列ID分配失败（所有seq_id都被占用时，新请求保持PENDING）
5. 测试序列ID重用（请求完成后，seq_id可以被新请求使用）
6. 测试并发安全性（多线程同时分配/释放）

---

## Phase 3: 超时检测机制

### 目标
实现请求超时检测机制，定期扫描 `runningRequests_` 中的 PROCESSING 状态请求，超时后自动处理。

### 任务清单

#### 3.1 超时检测框架
- [ ] 在 `Scheduler` 中添加超时检测线程或定时器
  - 方案A：在 `schedulerLoop` 中定期检查（推荐，减少线程开销）
  - 方案B：独立超时检测线程（可选）
- [ ] 实现超时检测周期配置
  - 从配置读取：`requestTimeout`（默认60s）
  - 检测周期：1秒（可配置）

#### 3.2 超时检测逻辑
- [ ] 实现 `checkRequestTimeout()` 方法
  - 扫描 `runningRequests_` 中状态为 PROCESSING 的请求
  - 计算处理时间：`当前时间 - startTime`
  - 如果处理时间 > `requestTimeout`，标记为 TIMEOUT
- [ ] 实现超时处理流程
  - 标记状态为 TIMEOUT
  - 释放序列ID（如果已分配）
  - 清理KV缓存（如果已分配）
  - 从 `runningRequests_` 移动到 `completedRequests_`
  - 通知HTTP层（通过回调或事件）

#### 3.3 集成到Scheduler循环
- [ ] 在 `schedulerLoop` 中调用超时检测
  - 每次循环或每N次循环检查一次
  - 避免过于频繁的检查（降低CPU开销）
- [ ] 确保超时检测不影响正常请求处理
  - 使用锁保护，避免与 `processBatch` 冲突

#### 3.4 HTTP层超时响应
- [ ] 实现HTTP 408响应
  - 在HTTP层检测到请求TIMEOUT状态时，返回HTTP 408
  - 确保HTTP层能够及时获取超时通知

### 验收标准
- ✅ 超时检测框架实现完成
- ✅ 超时检测逻辑正确（只检测PROCESSING状态，计算时间准确）
- ✅ 超时处理流程完整（状态标记、资源释放、通知HTTP层）
- ✅ 超时检测集成到 `schedulerLoop`
- ✅ HTTP层能够接收超时通知并返回HTTP 408
- ✅ 单元测试通过（至少4个测试用例）

### 依赖
Phase 1（必需：需要状态机和 `startTime` 字段）
Phase 2（可选：超时后释放序列ID，如果Phase 2未完成，超时检测仍然可以工作，但不会释放序列ID）

### 输出
- 超时检测实现代码
- HTTP层超时响应代码
- 单元测试代码

### 测试用例
1. 测试超时检测（请求处理时间超过 `requestTimeout`）
2. 测试超时处理流程（状态标记、资源释放）
3. 测试超时检测频率（不影响正常请求处理）
4. 测试HTTP层超时响应（返回HTTP 408）

---

## Phase 4: KV缓存统计管理

### 目标
实现基于 requestId 的KV缓存统计管理，包括统计信息跟踪和清理协调。**注意**：KV缓存数据由 `llama.cpp` 内部管理，我们只需要管理统计信息和协调清理。

### 任务清单

#### 4.1 KV缓存统计数据结构
- [ ] 定义 `KVCacheStats` 结构（统计信息，不是缓存数据）
  - 包含：requestId、条目数（序列长度）、内存占用估算、最后访问时间
- [ ] 创建 `KVCacheManager` 类（统计管理器，不是存储管理器）
  - 位置：`src/inference/kv_cache_manager.cpp`
  - 头文件：`include/cllm/inference/kv_cache_manager.h`
- [ ] 定义核心数据结构
  - `std::map<size_t requestId, KVCacheStats>` - requestId到统计信息的映射
  - `std::map<size_t requestId, RequestStatus>` - requestId到请求状态的映射（用于淘汰保护）
  - `size_t totalItems` - 总条目数
  - `size_t totalMemoryMb` - 总内存占用估算（MB）
  - `std::mutex` - 保护并发访问

#### 4.2 KV缓存统计更新
- [ ] 实现 `updateKVCacheStats(size_t requestId, size_t sequenceLength)` 方法
  - **注意**：不存储KV缓存数据，只更新统计信息
  - KV缓存条目数 = `tokenizedPrompt.size() + generatedTokens.size()` = 序列总长度
  - 内存占用估算 = `序列长度 × 每条目内存大小`（可配置或估算）
  - 更新最后访问时间
- [ ] 集成到 `LlamaCppBackend::forwardBatch`
  - 在推理完成后，调用 `updateKVCacheStats` 更新统计信息
  - 基于requestId关联

#### 4.3 KV缓存统计查询
- [ ] 实现 `getKVCacheStats(size_t requestId)` 方法
  - 根据requestId查询KV缓存统计信息
  - 更新最后访问时间（用于LRU）
- [ ] 实现 `hasKVCacheStats(size_t requestId)` 方法
  - 检查是否存在KV缓存统计信息

#### 4.4 KV缓存清理协调
- [ ] 实现 `removeKVCache(size_t requestId)` 方法
  - **协调清理**：调用 `llama_memory_seq_rm` 清理 `llama.cpp` 中的KV缓存（需要seqId）
  - **统计清理**：删除统计信息，更新总条目数和内存占用
  - **注意**：需要根据 requestId 查找对应的 seqId（通过序列ID管理器）
- [ ] 集成到请求完成流程
  - 在请求 COMPLETED/TIMEOUT/FAILED 时调用 `removeKVCache`
  - 通过序列ID管理器获取 requestId 对应的 seqId，然后清理

#### 4.5 统计信息
- [ ] 实现 `getTotalItems()` 方法
- [ ] 实现 `getTotalMemoryMb()` 方法
- [ ] 实现 `getCacheCount()` 方法（缓存数量）

### 验收标准
- ✅ `KVCacheManager` 类实现完成（统计管理器）
- ✅ KV缓存统计更新逻辑正确（基于requestId，计算序列长度）
- ✅ KV缓存统计查询逻辑正确
- ✅ KV缓存清理协调逻辑正确（调用 `llama_memory_seq_rm` 清理 `llama.cpp` 缓存）
- ✅ 统计信息准确（条目数=序列长度，内存占用估算正确）
- ✅ 集成到 `LlamaCppBackend`
- ✅ 单元测试通过（至少6个测试用例）

### 依赖
Phase 1（部分依赖：统计更新可独立实现，但完整清理需要在请求完成时）
Phase 2（需要序列ID管理器来获取 requestId 对应的 seqId，以便清理 `llama.cpp` 缓存）

### 输出
- `KVCacheManager` 实现代码（统计管理器）
- 集成到 `LlamaCppBackend` 的代码修改
- 单元测试代码

### 测试用例
1. 测试KV缓存统计更新（单个requestId，序列长度计算）
2. 测试KV缓存统计查询（单个requestId）
3. 测试KV缓存清理协调（调用 `llama_memory_seq_rm` 清理）
4. 测试统计信息准确性（条目数=序列长度，内存占用估算）
5. 测试并发安全性（多线程同时更新/查询/清理统计信息）
6. 测试KV缓存统计与requestId的关联（多个requestId的统计互不干扰）

---

## Phase 5: KV缓存LRU淘汰

### 目标
实现KV缓存的LRU淘汰策略，当缓存条目数或内存超过限制时，自动淘汰最久未使用的缓存。

### 任务清单

#### 5.1 淘汰触发条件
- [ ] 实现淘汰触发检查
  - 条目数检查：`totalItems > maxKVCachesItems × kvCacheEvictionThreshold`
  - 内存检查：`totalMemoryMb > kvCacheMaxMemoryMb × kvCacheEvictionThreshold`
- [ ] 从配置读取淘汰参数
  - `maxKVCachesItems`（默认：4*1024*1024）
  - `kvCacheMaxMemoryMb`（默认：1024MB）
  - `kvCacheEvictionThreshold`（默认：0.8）

#### 5.2 LRU淘汰逻辑
- [ ] 实现 `evictLRUCache()` 方法
  - 按照requestId的最后访问时间排序
  - 只淘汰状态为 PENDING 或 COMPLETED 的请求对应的KV缓存
  - **保护**：不淘汰状态为 PROCESSING 的请求对应的KV缓存
  - 淘汰最久未使用的requestId的整个KV缓存
- [ ] 实现批量淘汰
  - 持续淘汰直到条目数或内存低于阈值

#### 5.3 定期清理
- [ ] 实现定期清理机制
  - 每 `kvCacheCleanupInterval` 秒检查一次（默认1秒）
  - 在 `schedulerLoop` 或独立线程中执行
- [ ] 集成到Scheduler循环
  - 在 `schedulerLoop` 中定期调用淘汰检查

#### 5.4 淘汰保护机制
- [ ] 实现淘汰保护
  - **方案A（推荐）**：`KVCacheManager` 维护 `requestStatus_` 映射（`std::map<size_t requestId, RequestStatus>`）
    - `Scheduler` 通过回调更新 `KVCacheManager` 中的请求状态
    - `KVCacheManager` 查询本地状态映射，决定是否淘汰
  - **方案B**：`KVCacheManager` 直接查询 `Scheduler` 的请求状态（增加依赖）
  - 只淘汰 PENDING 或 COMPLETED 状态的请求对应的KV缓存
  - 跳过 PROCESSING 状态的请求

### 验收标准
- ✅ 淘汰触发条件正确（条目数和内存限制）
- ✅ LRU淘汰逻辑正确（按最后访问时间排序）
- ✅ 淘汰保护机制正确（不淘汰PROCESSING状态的请求）
- ✅ 定期清理机制正常工作
- ✅ 淘汰后统计信息更新正确
- ✅ 单元测试通过（至少5个测试用例）

### 依赖
Phase 4（需要KV缓存管理基础）

### 输出
- LRU淘汰实现代码
- 定期清理机制代码
- 单元测试代码

### 测试用例
1. 测试淘汰触发（条目数超过阈值）
2. 测试淘汰触发（内存超过阈值）
3. 测试LRU排序（按最后访问时间）
4. 测试淘汰保护（不淘汰PROCESSING状态的请求）
5. 测试批量淘汰（持续淘汰直到低于阈值）

---

## Phase 6: HTTP层优化（可选）

### 目标
在HTTP层实现并发检查，超过 `maxConcurrentRequests` 时直接返回429，避免加入队列。

### 任务清单

#### 6.1 Scheduler接口扩展
- [ ] 在 `Scheduler` 中添加 `getRunningCount()` 方法
  - 返回 `runningRequests_.size()`
  - 线程安全（使用锁保护）
- [ ] 添加 `getMaxConcurrentRequests()` 方法
  - 返回 `maxConcurrentRequests` 配置值

#### 6.2 HTTP层并发检查
- [ ] 在 `GenerateEndpoint::handle` 中添加并发检查
  - 调用 `Scheduler::getRunningCount()`
  - 如果 `runningCount >= maxConcurrentRequests`，直接返回HTTP 429
  - 否则，调用 `Scheduler::addRequest()`
- [ ] 实现HTTP 429响应
  - 返回适当的错误消息
  - 包含重试建议（Retry-After头）

#### 6.3 性能优化（可选）
- [ ] 使用原子变量优化 `getRunningCount()` 性能
  - 如果性能敏感，可以使用 `std::atomic<size_t>` 跟踪计数
  - 减少锁竞争

### 验收标准
- ✅ `getRunningCount()` 方法实现完成
- ✅ HTTP层并发检查逻辑正确
- ✅ HTTP 429响应正确
- ✅ 性能测试通过（不影响正常请求处理）
- ✅ 单元测试通过（至少3个测试用例）

### 依赖
Phase 1（需要 `runningRequests_` 状态管理）

### 输出
- Scheduler接口扩展代码
- HTTP层并发检查代码
- 单元测试代码

### 测试用例
1. 测试并发检查（超过 `maxConcurrentRequests` 时返回429）
2. 测试HTTP 429响应格式
3. 测试性能影响（并发检查不影响正常请求处理）

### 备注
这是可选优化功能，如果当前实现已经满足需求，可以跳过此阶段。

---

## Phase 7: 响应回调机制

### 目标
实现完整的响应回调链路，确保请求完成/超时/失败后能够及时通知HTTP层。

### 任务清单

#### 7.1 回调接口定义
- [ ] 定义回调函数类型
  - `std::function<void(size_t requestId, const RequestState&)>`
- [ ] 在 `Scheduler` 中添加回调注册机制
  - `setResponseCallback(std::function<...>)` 方法
  - 存储回调函数

#### 7.2 回调触发
- [ ] 在请求完成时触发回调
  - COMPLETED：调用回调，传递requestId和RequestState
- [ ] 在请求超时时触发回调
  - TIMEOUT：调用回调，传递requestId和RequestState
- [ ] 在请求失败时触发回调
  - FAILED：调用回调，传递requestId和RequestState

#### 7.3 HTTP层回调处理
- [ ] 在HTTP层注册回调
  - 在 `GenerateEndpoint` 初始化时注册回调
  - 处理回调，更新HTTP响应
- [ ] 实现异步响应机制（可选）
  - 如果HTTP层使用异步模型，实现异步响应
  - 确保线程安全

#### 7.4 错误处理
- [ ] 实现回调异常处理
  - 回调函数执行失败不应影响请求处理
  - 记录回调错误日志

### 验收标准
- ✅ 回调接口定义完成
- ✅ 回调触发逻辑正确（COMPLETED/TIMEOUT/FAILED）
- ✅ HTTP层能够接收回调并更新响应
- ✅ 错误处理正确（回调异常不影响请求处理）
- ✅ 单元测试通过（至少4个测试用例）

### 依赖
Phase 1（需要状态机）
Phase 3（需要超时检测）

### 输出
- 回调机制实现代码
- HTTP层回调处理代码
- 单元测试代码

### 测试用例
1. 测试完成回调（COMPLETED）
2. 测试超时回调（TIMEOUT）
3. 测试失败回调（FAILED）
4. 测试回调异常处理（回调函数抛出异常）

---

## 实施建议

### 优先级建议
1. **P0（必须实现）**：Phase 0, Phase 1, Phase 2
   - 这是核心功能，必须实现才能支撑并发请求管理
2. **P1（重要优化）**：Phase 3, Phase 4
   - 超时检测和KV缓存管理是重要的稳定性和性能优化
3. **P2（可选优化）**：Phase 5, Phase 7
   - LRU淘汰和响应回调机制是进一步优化
4. **P3（未来优化）**：Phase 6
   - HTTP层并发检查是可选优化

### 并行开发建议
- **Phase 2 和 Phase 4** 可以部分并行开发（都部分依赖Phase 1，但基础功能可独立实现）
  - **注意**：两者都需要完成Phase 1的核心状态管理才能完整集成
- **Phase 3 和 Phase 4** 可以并行开发（独立功能）
- **Phase 5 依赖 Phase 1 + Phase 4**，必须在Phase 1和Phase 4完成后开始

### 测试策略
- **单元测试**：每个阶段完成后，编写单元测试验证功能
- **集成测试**：Phase 1-2完成后，编写集成测试验证端到端流程
- **压力测试**：Phase 4-5完成后，进行压力测试验证性能和稳定性

### 代码审查
- 每个阶段完成后，进行代码审查
- 重点关注：线程安全、资源管理、错误处理

### 文档更新
- 每个阶段完成后，更新相关文档
- 记录实现细节、已知问题和限制

---

## 里程碑

| 里程碑 | 包含阶段 | 目标日期 | 验收标准 |
|--------|----------|----------|----------|
| M1: 核心状态机 | Phase 0-1 | Week 1-2 | 状态机正常工作，请求能够正常流转 |
| M2: 序列ID管理 | Phase 2 | Week 2-3 | 序列ID分配和释放正常工作 |
| M3: 超时与缓存基础 | Phase 3-4 | Week 3-5 | 超时检测和KV缓存管理基础完成 |
| M4: 完整实现 | Phase 5-7 | Week 5-6 | 所有核心功能完成，系统稳定运行 |

---

## 风险评估

### 技术风险
- **线程安全**：多线程环境下的状态管理可能存在竞态条件
  - **缓解措施**：仔细设计锁机制，充分测试，避免死锁
  - **注意**：`runningRequests_`, `completedRequests_`, `RequestQueue` 都需要锁保护
- **性能影响**：超时检测和KV缓存淘汰可能影响性能
  - **缓解措施**：优化检测频率，使用高效的数据结构（如 `std::unordered_map`）
  - **注意**：定期扫描 `runningRequests_` 可能成为性能瓶颈
- **序列ID冲突**：重构序列ID管理时，需要确保与 `llama.cpp` 的集成正确
  - **缓解措施**：充分测试，确保 requestId → seqId 映射正确
- **KV缓存清理**：KV缓存由 `llama.cpp` 管理，清理时机需要协调
  - **缓解措施**：确保在请求完成时及时清理，避免内存泄漏

### 依赖风险
- **llama.cpp版本**：依赖llama.cpp的API稳定性
  - **缓解措施**：锁定llama.cpp版本，充分测试

---

## 附录

### 配置参数清单
参考 `concurrent_request_management_design.md` 的"配置说明"部分。

### 相关文档
- `concurrent_request_management_design.md` - 完整设计方案
- `naming_style_analysis.md` - 命名风格分析

### 更新日志
- 2026-01-18：初版创建
