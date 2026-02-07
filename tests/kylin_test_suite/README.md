# Kylin Backend 全面测试方案

## 概述

本测试方案基于 `direct_benchmark.cpp` 和现有测试文件，提供从模型加载到最终输出的完整流程测试。

## 测试架构

```
kylin_test_suite/
├── README.md                    # 本文件
├── kylin_test_framework.h       # 测试框架核心
├── kylin_test_main.cpp          # 测试主入口
├── test_model_loading.cpp       # 模型加载测试
├── test_inference_pipeline.cpp  # 推理流程测试
├── test_output_validation.cpp   # 输出验证测试
├── test_tokenizer_integration.cpp # Tokenizer集成测试
├── test_sampling_strategies.cpp # 采样策略测试
└── test_performance_benchmark.cpp # 性能基准测试
```

## 测试阶段

### Stage 1: 模型加载测试
- 配置文件解析
- 权重加载验证
- 内存分配检查
- 量化格式验证

### Stage 2: Tokenizer集成测试
- Tokenizer加载
- 编码/解码验证
- 特殊Token处理

### Stage 3: 推理流程测试
- 单次前向传播
- 批处理推理
- KV Cache管理
- 增量生成

### Stage 4: 输出验证测试
- Logits数值检查
- Token分布验证
- 生成文本质量

### Stage 5: 采样策略测试
- 贪婪采样
- 温度采样
- Top-K/Top-P采样

### Stage 6: 性能基准测试
- 吞吐量测试
- 延迟测试
- 内存使用

## 使用方法

```bash
# 编译测试
cd build
make kylin_test_suite

# 运行所有测试
./bin/kylin_test_suite --all

# 运行特定阶段测试
./bin/kylin_test_suite --stage=1    # 模型加载
./bin/kylin_test_suite --stage=3    # 推理流程
./bin/kylin_test_suite --test=tokenizer_integration

# 详细输出
./bin/kylin_test_suite --verbose

# 生成报告
./bin/kylin_test_suite --report=html
```

## 成功标准

1. **模型加载**: 100% 成功率，无内存泄漏
2. **Tokenizer**: 编解码往返误差 < 1%
3. **推理流程**: 输出有效的 logits，无 NaN/Inf
4. **输出验证**: 生成多样化的 token，非重复单一 token
5. **性能**: 达到预期的 tokens/second

## 调试信息

每个测试步骤会输出：
- [INFO] 一般信息
- [PASS] 测试通过
- [FAIL] 测试失败
- [WARN] 警告信息
- [DEBUG] 调试信息（verbose模式）
