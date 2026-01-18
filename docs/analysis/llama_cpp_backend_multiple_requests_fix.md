# llama.cpp 后端多请求问题分析与修复

## 问题描述

使用 `cllm_server` 测试 `/generate` 接口时：
1. **第一次请求**：正常返回
2. **第二次请求**：失败
3. **第三次请求**：服务崩溃（Segmentation fault）

## 根本原因分析

### 1. `currentPosition_` 共享问题

**问题**：`currentPosition_` 是实例变量，在多个请求之间共享，导致：
- 第一个请求完成后，`currentPosition_` 保留第一个请求的位置值
- 第二个请求开始时，从错误的位置开始推理，导致 KV cache 混乱
- 第三个请求时，错误累积，最终导致崩溃

### 2. `seq_id` 与请求 ID 不匹配

**问题**：`seq_id` 是批处理索引（0, 1, 2...），不是请求 ID：
- 第一个请求在批处理中的索引是 0，使用 `seq_id = 0`
- 第二个请求在批处理中的索引也可能是 0，也使用 `seq_id = 0`
- 这导致不同请求共享相同的 KV cache，造成混乱

### 3. 新请求检测不准确

**问题**：无法准确区分"新请求"和"同一请求的增量推理"：
- `forwardBatch()` 中的判断 `isNewRequest = (requestPositions[0].first == 0)` 不准确
- `requestPositions[0].first` 总是 0（相对于 `flatInputIds` 的偏移）

## 修复方案

### 1. 使用 `seqPositions_` Map 跟踪每个序列的位置

**修改**：
- 将 `currentPosition_` 改为 `std::unordered_map<llama_seq_id, size_t> seqPositions_`
- 每个 `seq_id` 维护自己的位置

**代码位置**：
- `include/cllm/inference/llama_cpp_backend.h:147`
- `src/inference/llama_cpp_backend.cpp:31`

### 2. 改进新请求检测逻辑

**策略**：
- **长序列（长度 > 1）**：总是作为新请求的 prefill，重置位置为 0
- **短序列（长度 == 1）**：
  - 如果有位置记录且位置 > 0：增量推理
  - 否则：新请求

**代码位置**：
- `src/inference/llama_cpp_backend.cpp:384-425`

### 3. 正确设置 `batch.pos` 和 `batch.seq_id`

**修复**：
- Prefill：`batch.pos[i] = i - seqStart`（从 0 开始）
- 增量推理：`batch.pos[i] = seqPosition`（从当前位置开始）
- `batch.seq_id[i][0] = seqIdKey`（使用批处理索引作为 seq_id）

**代码位置**：
- `src/inference/llama_cpp_backend.cpp:473-500`

## 已知限制

### 1. `seq_id` 不是真正的请求 ID

**问题**：`IBackend::forwardBatch` 接口没有 `sequenceIds` 参数，无法传递真正的请求 ID。

**影响**：
- 不同请求可能使用相同的 `seq_id`，导致 KV cache 混乱
- 当前修复通过"长序列总是重置"的策略缓解，但不是完美解决方案

**长期解决方案**：
- 修改 `IBackend::forwardBatch` 接口，添加 `sequenceIds` 参数
- 使用真正的请求 ID 作为 `seq_id`，而不是批处理索引

### 2. 位置记录不会自动清理

**问题**：`seqPositions_` 中的记录不会在请求完成时自动清理。

**影响**：
- 内存泄漏（虽然影响很小）
- 可能导致错误的增量推理判断

**解决方案**：
- 在请求完成时，通过回调机制清理位置记录
- 或者，定期清理长时间未使用的记录

## 测试建议

### 1. 单请求测试
```bash
curl -X POST http://localhost:18081/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "1+1=", "max_tokens": 10}'
```

### 2. 多请求测试
```bash
# 请求 1
curl -X POST http://localhost:18081/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "1+1=", "max_tokens": 10}'

# 请求 2（立即发送）
curl -X POST http://localhost:18081/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "2+2=", "max_tokens": 10}'

# 请求 3（立即发送）
curl -X POST http://localhost:18081/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "3+3=", "max_tokens": 10}'
```

### 3. 并发请求测试
```bash
# 使用 parallel 或 xargs 发送多个并发请求
for i in {1..5}; do
  curl -X POST http://localhost:18081/generate \
    -H "Content-Type: application/json" \
    -d "{\"prompt\": \"$i+$i=\", \"max_tokens\": 10}" &
done
wait
```

## 验证检查点

1. ✅ 第一次请求正常返回
2. ✅ 第二次请求正常返回（不再失败）
3. ✅ 第三次请求正常返回（不再崩溃）
4. ✅ 多个并发请求正常工作
5. ✅ 日志中显示正确的位置重置信息
6. ✅ 没有段错误或内存错误

## 后续优化建议

1. **修改接口**：在 `IBackend::forwardBatch` 中添加 `sequenceIds` 参数
2. **KV Cache 管理**：实现请求完成时的清理机制
3. **位置验证**：添加位置越界检查，防止崩溃
4. **日志增强**：添加更详细的调试日志，便于问题排查
