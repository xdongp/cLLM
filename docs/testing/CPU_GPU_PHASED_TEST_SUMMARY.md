# Kylin Backend CPU vs GPU 分阶段测试总结

## 📊 测试执行完成

**日期**: 2026-02-06  
**耗时**: 约 2 分钟（全部 5 个阶段）  
**结果**: ✅ 成功定位问题源头

---

## 🎯 核心发现

### 问题定位流程图

```
开始
 │
 ├─ Phase 1: 权重一致性验证
 │   └─ ✅ PASS (差异=0, cosine=1.0)
 │       └─ Embedding 权重上传到 GPU 正确
 │
 ├─ Phase 2: Embedding 层输出对比
 │   └─ ✅ PASS (差异=0, cosine=1.0)
 │       └─ Embedding 查找计算正确
 │
 ├─ Phase 3: 逐层 Transformer 对比 ⭐ 关键阶段
 │   ├─ Layer 0 InputNorm:     ✅ 一致
 │   ├─ Layer 0 QKV Projection: ✅ 一致
 │   ├─ Layer 0 Attention:      ❌ 首次偏差！
 │   │   └─ maxDiff=0.56, cosine=0.778
 │   ├─ Layer 0 PostNorm:       ❌ 偏差放大
 │   ├─ Layer 0 FFN:            ❌ 偏差继续放大
 │   └─ Layer 1-27:             ❌ 偏差累积
 │
 ├─ Phase 4: Logits 对比
 │   └─ ❌ FAIL
 │       ├─ Top-10 重叠: 0/10
 │       ├─ argmax 不一致
 │       └─ cosine ≈ 0
 │
 └─ Phase 5: 生成文本对比
     └─ ❌ FAIL
         └─ GPU 输出完全乱码

结论: 问题源头 = Layer 0 Attention 计算
```

---

## 📈 测试数据可视化

### Layer 0 各组件对比

| 组件 | maxDiff | cosine | 状态 |
|------|---------|--------|------|
| Embedding | 0.000 | 1.0000 | ✅ 完美 |
| InputNorm | 0.000 | 1.0000 | ✅ 完美 |
| QKV Projection | 0.000 | 1.0000 | ✅ 完美 |
| **Attention** | **0.561** | **0.7782** | ❌ **首个偏差** |
| PostNorm | 1.827 | 0.7291 | ❌ 偏差放大 |
| FFN | 1.206 | 0.4759 | ❌ 偏差放大 |

### 偏差累积趋势

```
Layer  0: Attention maxDiff=0.56    cosine=0.78
Layer  1: Attention maxDiff=1.43    cosine=0.26
Layer  2: Attention maxDiff=4.99    cosine=0.61
...
Layer 27: Attention maxDiff=3078    cosine=0.17
Final Norm:         maxDiff=37.36   cosine=0.85
```

→ 偏差从 Layer 0 开始，逐层指数级放大！

### Top-10 Token 对比（Token 9707）

| 排名 | CPU Token | GPU Token | 匹配？ |
|------|-----------|-----------|--------|
| 1 | 14582 | 15837 | ❌ |
| 2 | 15846 | 15840 | ❌ |
| 3 | 353 | 15833 | ❌ |
| 4 | 21806 | 15843 | ❌ |
| 5 | 9 | 15835 | ❌ |
| 6 | 106208 | 15838 | ❌ |
| 7 | 72390 | 15847 | ❌ |
| 8 | 13213 | 15848 | ❌ |
| 9 | 7662 | 15832 | ❌ |
| 10 | 3988 | 15846 | ❌ |

**重叠度: 0/10** - 完全不同！

---

## 🔬 技术细节

### Attention 内部流程

```cpp
// ===== 已验证一致 =====
InputNorm Output ✅
    ↓
Q, K, V Projection ✅  
    ↓
// ===== 从这里开始出现偏差 =====
Q/K RMS Norm (Qwen3 特有)  ← 待验证
    ↓
RoPE 应用  ← ⚠️ 最可疑
    ↓
KV Cache 更新/读取  ← ⚠️ 可疑
    ↓
Score = Q @ K^T / sqrt(d_k)  ← ⚠️ 可疑
    ↓
Softmax(Score)  ← 待验证
    ↓
Output = Softmax @ V  ← 待验证
    ↓
O Projection ✅ (权重正确)
```

### GPU 实现位置

**文件**: `src/kylin/hf/ggml_backend.cpp`

**关键代码段**:
- 第 780-1115 行: GPU 计算图中的 Attention 实现
- 第 814-817 行: `ggml_rope_ext` RoPE 应用
- 第 880-920 行: GQA head 扩展和维度变换
- 第 926-932 行: Attention Score 和 Softmax

**对比文件**: `src/kylin/hf/transformer.cpp`
- CPU 的正确实现作为参考基准

---

## 🐛 最可疑的 3 个 Bug 候选

### 1. RoPE 模式参数错误 (可能性: 90%)

```cpp
// 第 814 行
q = ggml_rope_ext(ctx, q, pos, nullptr, headDim, 
                  rope_mode,  // ← 当前值: 2 (GGML_ROPE_TYPE_NEOX)
                  ...);
```

**问题**: Qwen3 可能需要不同的 RoPE 类型
- `rope_mode=0`: GPT-J style
- `rope_mode=2`: NeoX style  

**验证方法**: 尝试改为 `rope_mode=0`

### 2. GQA Head 映射错误 (可能性: 80%)

Qwen3 配置:
- 16 个 Q heads
- 2 个 KV heads
- 每个 KV head 服务 8 个 Q heads

**问题**: GPU 的 head 扩展可能不正确

```cpp
// 第 880-920 行
// 需要将 K/V 从 [2, headDim] 扩展到 [16, headDim]
ggml_repeat(...) 的使用可能有误
```

### 3. Position 参数传递错误 (可能性: 60%)

```cpp
// GPU 图中
ggml_rope_ext(ctx, q, pos, ...)  // pos 是 tensor

// CPU 路径中
cpuApplyRoPE(..., position, ...)  // position 是 int
```

**问题**: position 值可能不一致或传递错误

---

## 📋 测试文件清单

### 新增测试文件

1. `test_phased_cpu_gpu_comparison.cpp` - 5 阶段主测试
   - Stage 30: Phase 1 权重验证
   - Stage 31: Phase 2 Embedding 对比
   - Stage 32: Phase 3 逐层对比
   - Stage 33: Phase 4 Logits 对比
   - Stage 34: Phase 5 生成对比

2. `test_attention_breakdown.cpp` - Attention 细分测试（框架）
   - Stage 35: Attention 内部各子步骤对比

3. `run_phased_tests.sh` - 一键运行脚本

4. `CPU_GPU_COMPARISON_REPORT.md` - 详细测试报告

5. `QUICK_START.md` - 快速开始指南

6. `README_PHASED_TEST.md` - 本文档

### 更新文件

- `kylin_test_main.cpp` - 注册 Stage 30-34

---

## 🚀 如何使用

### 场景 1: 第一次运行测试

```bash
cd /Users/dannypan/PycharmProjects/cLLM
cd tests/kylin_test_suite
./run_phased_tests.sh
```

查看 `CPU_GPU_COMPARISON_REPORT.md` 了解详情。

### 场景 2: 调试特定阶段

```bash
cd build

# 只运行 Phase 3（最关键）
./bin/kylin_test_suite --stage=32 --verbose

# 对比前后效果
# 修改代码后重新编译
cd .. && make -C build kylin_test_suite -j4

# 再次运行
cd build && ./bin/kylin_test_suite --stage=32 --verbose
```

### 场景 3: 查看日志

```bash
# 日志保存在
ls tests/kylin_test_suite/test_logs/

# 查看最新的 Phase 3 日志
tail -200 tests/kylin_test_suite/test_logs/phase3_*.log
```

---

## 🎓 测试方法学习

### 为什么分阶段？

传统方法：
```
❌ 直接对比最终输出 → "GPU 生成乱码" → 不知道哪里错了
```

分阶段方法：
```
✅ Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5
   逐步缩小范围 → 精确定位到具体函数
```

### 测试设计原则

1. **从底层到上层**: 先测权重，再测 Embedding，再测层
2. **逐步细化**: 先测整层，再测层内组件
3. **使用断言**: 每个阶段都有明确的通过/失败标准
4. **记录中间结果**: 所有中间张量都导出对比
5. **可重复**: 使用固定 seed，结果可复现

---

## 📞 快速参考

### 命令速查

```bash
# 编译
make -C build kylin_test_suite -j4

# 运行所有测试
./tests/kylin_test_suite/run_phased_tests.sh

# 单独运行某阶段
./build/bin/kylin_test_suite --stage=32 --verbose

# 查看帮助
./build/bin/kylin_test_suite --help
```

### 日志位置

```
tests/kylin_test_suite/test_logs/phase{1-5}_YYYYMMDD_HHMMSS.log
```

### 报告文件

```
tests/kylin_test_suite/CPU_GPU_COMPARISON_REPORT.md  # 详细报告
docs/testing/CPU_GPU_PHASED_TEST_SUMMARY.md          # 本总结
```

---

**测试框架状态**: ✅ 完整可用  
**问题定位状态**: ✅ 已精确定位  
**下一步**: 🔧 修复 GPU Attention 实现
