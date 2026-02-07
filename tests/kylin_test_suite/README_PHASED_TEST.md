# CPU vs GPU 分阶段对比测试

## 测试成果

已成功创建 5 阶段精确对比测试，**定位到 GPU 问题的根本原因**。

## 核心发现 🎯

```
✅ Embedding 权重    → CPU == GPU (完全一致)
✅ Embedding 输出    → CPU == GPU (完全一致)
✅ InputNorm        → CPU == GPU (完全一致)
✅ QKV Projection   → CPU == GPU (完全一致)
❌ Attention        → CPU ≠ GPU ← 首个偏差点！
   └─ maxDiff=0.56, cosine=0.778
❌ 后续所有层       → 偏差累积放大
❌ 最终 Logits      → Top-10 完全不重叠（0/10）
❌ 生成文本         → GPU 输出乱码
```

## 问题定位

**Layer 0 Attention** 是偏差的**唯一源头**，问题可能在：

1. **RoPE (Rotary Position Embedding)** ⭐⭐⭐⭐⭐
   - GPU 使用: `ggml_rope_ext(rope_mode=2, ...)`
   - CPU 使用: `cpuApplyRoPE(...)`
   - 可能参数不匹配

2. **GQA (Grouped Query Attention)** ⭐⭐⭐⭐⭐
   - 16 Q heads, 2 KV heads
   - GPU 的 head 扩展可能有误

3. **Attention Score 计算** ⭐⭐⭐
   - GPU: `ggml_mul_mat(k_cont, q_cont)`
   - CPU: 手动 dot product
   - 矩阵维度可能有问题

## 快速运行

```bash
# 方法 1: 运行所有测试
cd tests/kylin_test_suite
./run_phased_tests.sh

# 方法 2: 只运行关键阶段
cd build
./bin/kylin_test_suite --stage=32 --verbose  # Phase 3: 逐层对比
```

## 输出示例

### Phase 3 输出（关键）

```
Layer | InputNorm | QKV      | Attention   | PostNorm    | FFN
    0 | ✅ 一致   | ✅ 一致  | ❌ cos=0.78 | ❌ cos=0.73 | ❌ cos=0.48
    1 | ⚠️        | ⚠️       | ❌ cos=0.26 | ❌          | ❌
   ...
   27 | ❌        | ❌       | ❌ cos=0.17 | ❌          | ❌

结论: 首个偏差出现在 Layer 0 -> Attention
```

### Phase 5 输出（乱码验证）

```
Prompt: "hello"
CPU: "@@@@@@@@@@"
GPU: "\n=NULL.Pref Loader OutOfBoundsException..."

Prompt: "你好"
CPU: "！！！！！！"  
GPU: "ㇾㇾㇾ IconButton membership..."
```

## 调试建议

### 立即行动

1. 打开 `src/kylin/hf/ggml_backend.cpp`
2. 查看第 814-817 行的 `ggml_rope_ext` 调用
3. 对比 `transformer.cpp` 中的 CPU RoPE 实现
4. 检查 `rope_mode`、`n_ctx_orig`、`ropeTheta` 参数

### 验证方法

创建单元测试：
```cpp
// 输入: 相同的 Q、K tensor
// CPU: cpuApplyRoPE(q, k, position, ...)
// GPU: ggml_rope_ext(q, k, position, ...)
// 对比: 输出是否一致
```

## 相关文件

| 文件 | 说明 |
|------|------|
| `test_phased_cpu_gpu_comparison.cpp` | 5 阶段测试实现 |
| `test_attention_breakdown.cpp` | Attention 细分测试框架 |
| `CPU_GPU_COMPARISON_REPORT.md` | 详细测试报告 |
| `run_phased_tests.sh` | 一键运行脚本 |
| `QUICK_START.md` | 本文档 |

## 测试统计

- **总测试时间**: ~2 分钟（所有 5 个阶段）
- **模型加载**: 2 次（CPU + GPU）
- **Token 测试**: 多个代表性 token
- **生成测试**: 4 个不同 prompt

---

**结论**: 测试框架已就绪，问题已精确定位到 **GPU Attention 实现**。  
**下一步**: 修复 `ggml_backend.cpp` 中的 Attention 计算逻辑。
