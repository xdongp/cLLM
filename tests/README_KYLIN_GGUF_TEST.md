# Kylin引擎GGUF Q4_K_M推理测试

## 概述

本测试用于验证Kylin引擎能够正确加载Q4_K_M格式的GGUF模型（`qwen3-0.6b-q4_k_m.gguf`）并执行推理。

## 测试文件

- **测试代码**: `tests/test_kylin_gguf_q4k.cpp`
- **运行脚本**: `tests/run_kylin_gguf_test.sh`

## 前置条件

1. **模型文件**: 确保模型文件存在于 `model/Qwen/qwen3-0.6b-q4_k_m.gguf`
2. **编译环境**: 已安装CMake、C++17编译器、Google Test
3. **依赖库**: 已编译cllm_core库

## 快速开始

### 方法1: 使用运行脚本（推荐）

```bash
cd /Users/dannypan/PycharmProjects/xllm/cpp/cLLM
./tests/run_kylin_gguf_test.sh
```

### 方法2: 手动编译和运行

```bash
# 1. 进入构建目录
cd build

# 2. 配置CMake（启用测试）
cmake .. -DBUILD_TESTS=ON

# 3. 编译测试
make test_kylin_gguf_q4k -j8

# 4. 运行测试
./bin/test_kylin_gguf_q4k --gtest_color=yes
```

## 测试内容

### 1. 模型加载测试 (`ModelLoadingTest`)
- 验证GGUF模型文件能够正确加载
- 验证配置参数从GGUF文件正确读取
- 验证模型初始化成功

### 2. 单序列推理测试 (`SingleSequenceInferenceTest`)
- 测试单序列前向推理
- 验证输出logits形状正确 `[seq_len, vocab_size]`
- 验证输出数值合理性（无NaN/Inf，数值范围合理）

### 3. 不同序列长度测试 (`DifferentSequenceLengthsTest`)
- 测试1、2、4、8个token的不同长度序列
- 验证所有长度都能正常推理

### 4. 批处理推理测试 (`BatchInferenceTest`)
- 测试批处理推理功能
- 验证多个请求能够并行处理

### 5. 推理一致性测试 (`InferenceConsistencyTest`)
- 验证相同输入产生相同输出（确定性推理）
- 检查浮点误差在可接受范围内

### 6. 错误处理测试 (`ErrorHandlingTest`)
- 测试无效token ID的处理
- 测试空输入的处理

### 7. 性能基准测试 (`PerformanceBenchmarkTest`)
- 测量推理性能
- 输出平均推理时间和吞吐量

## 预期输出

测试成功时，应该看到类似以下输出：

```
[==========] Running 7 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 7 tests from KylinGGUFQ4KTest
[ RUN      ] KylinGGUFQ4KTest.ModelLoadingTest
[KylinBackend] Initializing Kylin (麒麟) inference backend
[KylinBackend] Will load real weights from: model/Qwen/qwen3-0.6b-q4_k_m.gguf
[ModelLoader] detected GGUF format
[ModelLoader] GGUF model loaded successfully
  - Vocab size: 151936
  - Hidden size: 896
  - Num layers: 16
[       OK ] KylinGGUFQ4KTest.ModelLoadingTest (1234 ms)
[ RUN      ] KylinGGUFQ4KTest.SingleSequenceInferenceTest
[       OK ] KylinGGUFQ4KTest.SingleSequenceInferenceTest (567 ms)
...
[----------] 7 tests from KylinGGUFQ4KTest (12345 ms total)

[==========] 7 tests from 1 test suite ran. (12345 ms total)
[  PASSED  ] 7 tests.
```

## 故障排查

### 问题1: 模型文件未找到

```
错误: Model file not found: model/Qwen/qwen3-0.6b-q4_k_m.gguf
```

**解决方案**:
- 检查模型文件路径是否正确
- 确保文件存在且有读取权限
- 可以修改测试代码中的`modelPath_`变量

### 问题2: 模型加载失败

```
[KylinBackend] Failed to load model weights via ModelLoader
```

**可能原因**:
- GGUF文件格式不正确或损坏
- 模型文件不完整
- 内存不足

**解决方案**:
- 验证GGUF文件完整性
- 检查系统内存
- 查看详细日志定位问题

### 问题3: 推理输出异常

```
Logits contain NaN values
或
Max logit value too large
```

**可能原因**:
- 权重加载错误
- 量化格式解析错误
- 数值计算溢出

**解决方案**:
- 检查量化格式实现
- 验证权重反量化正确性
- 检查输入token ID是否在有效范围内

### 问题4: 编译错误

**常见错误**:
- 缺少头文件：检查include路径
- 链接错误：确保cllm_core库已编译
- C++17特性：确保编译器支持C++17

## 测试参数说明

### Token ID选择

测试中使用动态计算的token ID，基于词汇表大小：
- `vocabSize / 4`: 使用词汇表的1/4位置
- `vocabSize / 8`: 使用词汇表的1/8位置
- 等等...

这确保了token ID始终在有效范围内，避免越界错误。

### 配置参数

测试中的初始配置参数会被GGUFLoader自动覆盖：
- `vocabSize`: 从GGUF元数据读取
- `hiddenSize`: 从GGUF元数据读取
- `numLayers`: 从GGUF元数据读取
- 等等...

## 性能参考

在典型硬件上（如M1 Mac），Qwen3-0.6B Q4_K_M模型的推理性能：
- **单次推理时间**: 50-200ms（取决于序列长度）
- **吞吐量**: 50-200 tokens/sec

实际性能取决于：
- CPU性能
- 内存带宽
- 序列长度
- 批处理大小

## 扩展测试

可以基于此测试添加：
1. **更多量化格式**: Q5_K_M, Q6_K, Q8_0等
2. **更长序列**: 测试最大序列长度
3. **压力测试**: 连续多次推理
4. **内存测试**: 验证内存使用情况
5. **精度对比**: 与FP32模型对比输出差异

## 相关文档

- [GGUF Q4_K_M格式分析](../docs/research/gguf_q4k_inference_analysis.md)
- [Kylin模块代码审查](../docs/review/kylin_module_review.md)
- [GGUF规范](../docs/design/GGUF规范.md)

---

**最后更新**: 2025-01-XX
