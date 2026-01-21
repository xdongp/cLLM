# cLLM 回退后完整并发测试报告

## 执行摘要

本报告展示了回退到 commit `0c7229e16d85b212a657f384d7a513c36b3189c4`（"测试成功，32并发超300"）后的完整测试结果。

### 回退原因
- JSON库迁移（simdjson）后性能下降
- 回退到使用 nlohmann/json 的稳定版本

### 测试配置
- **请求数量**: 72个
- **每个请求最大tokens**: 50
- **测试类型**: Concurrent (8/16/24/32并发)
- **测试时间**: 2026-01-20 (回退后)

## 回退后 cLLM 并发性能

| 并发数 | 成功请求 | 失败请求 | 总吞吐量 (t/s) | 平均响应时间 (s) | 总测试时间 (s) |
|--------|---------|---------|---------------|----------------|---------------|
| **8** | 71/72 | 1 | **111.61** | 3.83 | 35.17 |
| **16** | 71/72 | 1 | **173.83** | 8.50 | 40.26 |
| **24** | 72/72 | 0 | **159.39** | 16.10 | 54.32 |
| **32** | 72/72 | 0 | **189.10** | 20.00 | 48.88 |

### 关键指标

#### 吞吐量趋势
- **并发8**: 111.61 t/s
- **并发16**: 173.83 t/s（+55.8%）
- **并发24**: 159.39 t/s（-8.3%）
- **并发32**: 189.10 t/s（+18.6%）

**最佳性能点**: 并发32达到最高吞吐量 **189.10 t/s**

#### 稳定性
- **并发8**: 98.6% 成功率（1个失败）
- **并发16**: 98.6% 成功率（1个失败）
- **并发24**: 100% 成功率 ✅
- **并发32**: 100% 成功率 ✅

## 回退操作详情

### Git操作
1. ✅ 暂存了simdjson迁移的所有更改到stash
2. ✅ 执行 `git reset --hard 0c7229e16d85b212a657f384d7a513c36b3189c4`
3. ✅ 清理了simdjson相关文件（json_wrapper.h/cpp）
4. ✅ 重新编译项目

### 代码状态
- ✅ 使用 `nlohmann/json` 库（回退前状态）
- ✅ 移除了 `simdjson` 相关代码
- ✅ 编译成功

### Stash信息
- Stash名称: "Stash before reverting to 0c7229e16d85b212a657f384d7a513c36b3189c4"
- 查看命令: `git stash list`
- 恢复命令: `git stash pop`（如果需要恢复simdjson迁移）

## 测试命令

```bash
# 并发8
python3 tools/unified_benchmark.py --server-type cllm --test-type api-concurrent --requests 72 --concurrency 8 --max-tokens 50

# 并发16
python3 tools/unified_benchmark.py --server-type cllm --test-type api-concurrent --requests 72 --concurrency 16 --max-tokens 50

# 并发24
python3 tools/unified_benchmark.py --server-type cllm --test-type api-concurrent --requests 72 --concurrency 24 --max-tokens 50

# 并发32
python3 tools/unified_benchmark.py --server-type cllm --test-type api-concurrent --requests 72 --concurrency 32 --max-tokens 50
```

## 测试结果文件

- `/tmp/cllm_rebenchmark_8.json`
- `/tmp/cllm_rebenchmark_16.json`
- `/tmp/cllm_rebenchmark_24.json`
- `/tmp/cllm_rebenchmark_32.json`

## 与参考文档对比

### 回退后 vs 参考文档（修复后）

| 并发数 | 参考文档吞吐量 (t/s) | 回退后吞吐量 (t/s) | 差异 | 参考文档成功率 | 回退后成功率 |
|--------|-------------------|------------------|------|--------------|------------|
| **8** | 137.73 | 111.61 | **-19.0%** | 100% | 98.6% |
| **16** | 289.00 | 173.83 | **-39.8%** | 100% | 98.6% |
| **24** | 257.20 | 159.39 | **-38.0%** | 98.6% | 100% |
| **32** | 347.99 | 189.10 | **-45.7%** | 100% | 100% |

### 分析

1. **性能差异**: 回退后的吞吐量明显低于参考文档中的修复后性能
   - 可能原因：测试环境差异、系统负载不同、或参考文档中的修复措施未完全回退

2. **稳定性**: 
   - 并发8/16: 回退后有1个失败（参考文档中为0）
   - 并发24/32: 回退后100%成功率（与参考文档一致）

3. **性能趋势**:
   - 回退后：并发8→16提升55.8%，16→24下降8.3%，24→32提升18.6%
   - 参考文档：并发8→16提升109.8%，16→24下降11.0%，24→32提升35.3%
   - 回退后的扩展性不如参考文档中的表现

---

**报告生成时间**: 2026-01-20
**Git Commit**: 0c7229e16d85b212a657f384d7a513c36b3189c4
**测试工具**: tools/unified_benchmark.py
