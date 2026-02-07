# Kylin CPU vs GPU 对比测试 - 快速开始

## 运行所有测试（推荐）

```bash
cd tests/kylin_test_suite
./run_phased_tests.sh
```

这会按顺序运行 5 个阶段的测试，并保存日志到 `test_logs/` 目录。

---

## 单独运行各阶段

```bash
cd build

# Phase 1: 权重一致性验证 (约 30 秒)
./bin/kylin_test_suite --stage=30 --verbose

# Phase 2: Embedding 层对比 (约 20 秒)
./bin/kylin_test_suite --stage=31 --verbose

# Phase 3: 逐层 Transformer 对比 ⭐ 最重要 (约 15 秒)
./bin/kylin_test_suite --stage=32 --verbose

# Phase 4: Logits 对比 (约 20 秒)
./bin/kylin_test_suite --stage=33 --verbose

# Phase 5: 生成文本对比 (约 20 秒)
./bin/kylin_test_suite --stage=34 --verbose
```

---

## 测试结果

### ✅ Phase 1-2: 基础验证通过

- ✅ 权重上传到 GPU 正确
- ✅ Embedding 查找正确

### ⚠️ Phase 3: 发现问题源头

**结果**: Layer 0 Attention 首次出现偏差

```
Layer | InputNorm | QKV Proj | Attention   | PostNorm    | FFN
    0 |   一致    |   一致   | ❌ 有偏差  | ❌ 偏差放大 | ❌ 偏差放大
      |           |          | maxDiff=0.56, cos=0.778
```

### ❌ Phase 4-5: 偏差影响

- ❌ Logits Top-10 完全不重叠（0/10）
- ❌ GPU 生成乱码文本

**GPU 生成示例**:
```
Prompt: "hello"
CPU: "@@@@@@@@@@@@@@"
GPU: "\n=NULL.Pref Loader OutOfBoundsException..."

Prompt: "你好"  
CPU: "！！！！！！"
GPU: "ㇾㇾㇾ IconButton membership..."
```

---

## 问题定位总结

🎯 **根本原因**: GPU 的 **Attention 计算**实现有误

📍 **精确位置**: Layer 0 Attention（所有层都继承了这个错误实现）

🔍 **可疑点**:
1. **RoPE (旋转位置编码)** - 最可疑 ⭐⭐⭐⭐⭐
   - CPU: `cpuApplyRoPE()`
   - GPU: `ggml_rope_ext(rope_mode=2, ...)`
   - 可能 mode 参数不对或 position 传递有误

2. **GQA (Grouped Query Attention)** - 很可疑 ⭐⭐⭐⭐⭐
   - 16 Q heads, 2 KV heads, ratio=8
   - GPU 的 head 扩展逻辑可能有误

3. **Attention Score 矩阵乘法** - 可疑 ⭐⭐⭐
   - CPU: 手动循环
   - GPU: `ggml_mul_mat(k_cont, q_cont)`
   - 维度/转置可能有问题

4. **KV Cache 索引** - 可疑 ⭐⭐
   - CPU 和 GPU 的索引方式不同

---

## 下一步调试建议

### 方法 1: 直接检查代码

打开 `src/kylin/hf/ggml_backend.cpp`：

1. **第 814-817 行**: 检查 RoPE 参数
   ```cpp
   q = ggml_rope_ext(ctx, q, pos, nullptr, headDim, 
                     rope_mode,      // 检查：应该是 0 还是 2？
                     n_ctx_orig,     // 检查：值是否正确？
                     config_.ropeTheta, ...);
   ```

2. **第 880-920 行**: 检查 GQA 实现
   ```cpp
   // 是否正确扩展了 KV heads？
   ggml_repeat(...) 的使用是否正确？
   ```

3. **第 926-932 行**: 检查 Attention Score 计算
   ```cpp
   ggml_tensor* kq = ggml_mul_mat(ctx, k_cont, q_cont);
   // k_cont 和 q_cont 的维度是否匹配？
   ```

### 方法 2: 添加调试输出

在 GPU 计算图中添加张量值打印：
```cpp
// 在第 814 行后添加
CLLM_INFO("[DEBUG] After RoPE: q[0]=%.6f, k[0]=%.6f", ...);

// 在第 926 行后添加  
CLLM_INFO("[DEBUG] Attention Score: kq[0]=%.6f", ...);

// 在第 932 行后添加
CLLM_INFO("[DEBUG] After Softmax: kq_soft[0]=%.6f", ...);
```

### 方法 3: 单独测试 RoPE

创建最小测试：
```cpp
// 输入相同的 Q、K
// 分别用 cpuApplyRoPE 和 ggml_rope_ext 处理
// 对比输出
```

---

## 相关文件

- 测试报告: `CPU_GPU_COMPARISON_REPORT.md`
- 测试脚本: `run_phased_tests.sh`
- 测试源码: `test_phased_cpu_gpu_comparison.cpp`
- GPU 后端: `../../src/kylin/hf/ggml_backend.cpp`
- CPU 实现: `../../src/kylin/hf/transformer.cpp`
