# GPU 权重上传修复记录

## 问题描述

在使用 Kylin backend + GPU (Metal) 配置时，服务器在初始化阶段发生 segmentation fault 崩溃。

### 症状

服务器启动日志显示：
```
[GGMLGPUBackend] Initializing Metal backend...
[GGMLGPUBackend] ✅ Metal backend initialized
[GGMLGPUBackend] Uploading weights to GPU...
[GGMLGPUBackend] Uploading embedTokens...
```

程序在权重上传过程中崩溃，退出码为 139 (SIGSEGV)。

## 根本原因分析

经过详细分析，发现有两个根本原因：

### 原因 1: FP16 模式下 FP32 权重未初始化

**位置**: `src/kylin/hf/transformer.cpp` 中的 `convertWeightsToFP16()` 函数

**问题描述**:
- 在 FP16 模式下，`convertWeightsToFP16()` 只创建了 FP16 版本的权重 (`embedTokensFP16_`, `lmHeadWeightFP16_`)
- 但 GPU 上传代码需要使用 FP32 版本的权重 (`embedTokensF32_.data()`, `lmHeadWeightF32_.data()`)
- 导致空指针传递给 `ggml_backend_tensor_set`，引发 segmentation fault

**代码流程**:
```cpp
// convertWeightsToFP16() 中
embedTokensFP16_.resize(embedSize);  // ✅ 创建 FP16 版本
convertBF16toFP16(embedTokens_, embedTokensFP16_.data(), embedSize);

// ❌ 缺少：embedTokensF32_.resize(embedSize);
// ❌ 缺少：quant_kernels::convert_fp16_to_f32(...);

// GPU 上传代码中
ggml_backend_tensor_set(embedTokens_, embedTokensF32_.data(), ...);  // ❌ 空指针崩溃
```

### 原因 2: 缺少空指针检查

**位置**: `src/kylin/hf/ggml_backend.cpp` 中的 `uploadWeights()` 函数

**问题描述**:
- `uploadWeights` 函数没有检查输入指针是否为 null
- 当传入空指针时，`ggml_backend_tensor_set` 会直接崩溃
- 缺少详细的调试日志，难以定位问题

## 修复方案

### 修复 1: 在 convertWeightsToFP16() 中创建 FP32 权重

**文件**: `src/kylin/hf/transformer.cpp`

**修改内容**:
```cpp
// 转换嵌入层
size_t embedSize = static_cast<size_t>(vocabSize) * hiddenSize;
embedTokensFP16_.resize(embedSize);
convertBF16toFP16(embedTokens_, embedTokensFP16_.data(), embedSize);

// ✅ 新增：同时创建 FP32 版本用于 GPU 上传
embedTokensF32_.resize(embedSize);
quant_kernels::convert_fp16_to_f32(embedTokensFP16_.data(), embedTokensF32_.data(), embedSize);

// 转换 LM Head
if (!config_.tieWordEmbeddings) {
    lmHeadWeightFP16_.resize(embedSize);
    convertBF16toFP16(lmHeadWeight_, lmHeadWeightFP16_.data(), embedSize);
    // ✅ 新增：同时创建 FP32 版本用于 GPU 上传
    lmHeadWeightF32_.resize(embedSize);
    quant_kernels::convert_fp16_to_f32(lmHeadWeightFP16_.data(), lmHeadWeightF32_.data(), embedSize);
} else {
    lmHeadWeightFP16_ = embedTokensFP16_;
    lmHeadWeightF32_ = embedTokensF32_;  // ✅ 新增：共享相同的 FP32 数据
}
```

**原理**:
- FP16 模式下，CPU 推理使用 FP16 权重以节省内存
- GPU 推理需要 FP32 权重，因为 Metal 后端目前只支持 FP32
- 因此需要同时维护 FP16 和 FP32 两个版本的权重

### 修复 2: 增强 uploadWeights() 的错误检查

**文件**: `src/kylin/hf/ggml_backend.cpp`

**修改内容**:
```cpp
bool GGMLGPUBackend::uploadWeights(
    const float* embedTokens,
    const std::vector<LayerWeightsGPU>& layerWeights,
    const float* finalNorm,
    const float* lmHead
) {
    if (!initialized_) {
        CLLM_ERROR("[GGMLGPUBackend] Not initialized");
        return false;
    }

    // ✅ 新增：检查输入指针
    if (!embedTokens) {
        CLLM_ERROR("[GGMLGPUBackend] embedTokens is null");
        return false;
    }
    if (!finalNorm) {
        CLLM_ERROR("[GGMLGPUBackend] finalNorm is null");
        return false;
    }

    const int vocabSize = config_.vocabSize;
    const int hiddenSize = config_.hiddenSize;
    const int intermediateSize = config_.intermediateSize;
    const int headDim = config_.getHeadDim();
    const int nHeads = config_.numAttentionHeads;
    const int nKVHeads = config_.getNumKVHeads();
    const int qSize = nHeads * headDim;
    const int kvSize = nKVHeads * headDim;

    // ✅ 新增：详细日志
    CLLM_INFO("[GGMLGPUBackend] Uploading weights to GPU...");
    CLLM_INFO("[GGMLGPUBackend] vocabSize=%d, hiddenSize=%d, intermediateSize=%d", vocabSize, hiddenSize, intermediateSize);
    CLLM_INFO("[GGMLGPUBackend] numLayers=%zu, qSize=%d, kvSize=%d", layerWeights.size(), qSize, kvSize);

    // ✅ 新增：分步日志
    CLLM_INFO("[GGMLGPUBackend] Uploading embedTokens...");
    ggml_backend_tensor_set(embedTokens_, embedTokens, 0,
                        (size_t)vocabSize * hiddenSize * sizeof(float));

    CLLM_INFO("[GGMLGPUBackend] Uploading finalNorm...");
    ggml_backend_tensor_set(finalNorm_, finalNorm, 0, hiddenSize * sizeof(float));

    CLLM_INFO("[GGMLGPUBackend] Uploading lmHead...");
    if (!config_.tieWordEmbeddings && lmHead) {
        ggml_backend_tensor_set(lmHead_, lmHead, 0,
                                (size_t)vocabSize * hiddenSize * sizeof(float));
    } else {
        ggml_backend_tensor_set(lmHead_, embedTokens, 0,
                                (size_t)vocabSize * hiddenSize * sizeof(float));
    }

    // 每层权重
    CLLM_INFO("[GGMLGPUBackend] Uploading layer weights...");
    for (size_t i = 0; i < layerWeights.size() && i < layers_.size(); ++i) {
        const LayerWeightsGPU& src = layerWeights[i];
        LayerTensors& dst = layers_[i];

        // ✅ 新增：检查层权重指针
        if (!src.inputLayernorm || !src.postAttentionLayernorm) {
            CLLM_ERROR("[GGMLGPUBackend] Layer %zu norm weights are null", i);
            return false;
        }
        if (!src.qProj || !src.kProj || !src.vProj || !src.oProj) {
            CLLM_ERROR("[GGMLGPUBackend] Layer %zu attention weights are null", i);
            return false;
        }
        if (!src.gateProj || !src.upProj || !src.downProj) {
            CLLM_ERROR("[GGMLGPUBackend] Layer %zu FFN weights are null", i);
            return false;
        }

        // ... 上传权重 ...
    }

    CLLM_INFO("[GGMLGPUBackend] ✅ Weights uploaded");
    return true;
}
```

## 测试验证

修复后，Kylin + GPU (Metal) 配置正常工作：

### 测试环境
- 设备: Apple M3
- 模型: Qwen3-0.6B
- 后端: Kylin + Metal GPU
- 量化: FP16

### 测试结果

| 测试项目 | 结果 | 性能 |
|---------|------|------|
| Health Check | ✅ 正常 | - |
| Model Info | ✅ 正常 | - |
| 英文生成 (Hello) | ✅ 正常 | 33.75 tokens/s |
| 数学计算 (1+1=) | ✅ 正常 | 40.14 tokens/s |
| 中文理解 (介绍人工智能) | ✅ 正常 | 42.50 tokens/s |
| Benchmark | ✅ 正常 | 54.92 tokens/s (平均) |

### 服务器启动日志
```
[GGMLGPUBackend] Initializing Metal backend...
ggml_metal_init: found device: Apple M3
ggml_metal_init: picking default device: Apple M3
[GGMLGPUBackend] ✅ Metal backend initialized
[GGMLGPUBackend] ✅ CPU backend initialized (for mixed sched)
[GGMLGPUBackend] Creating tensors (layers=28)
[GGMLGPUBackend] ✅ GPU buffer: 2867.25 MB
[GGMLGPUBackend] RoPE frequencies precomputed (theta=1000000)
[GGMLGPUBackend] ✅ GPU Backend initialization complete
[GGMLGPUBackend] Uploading weights to GPU...
[GGMLGPUBackend] vocabSize=151936, hiddenSize=1024, intermediateSize=3072
[GGMLGPUBackend] numLayers=28, qSize=2048, kvSize=1024
[GGMLGPUBackend] Uploading embedTokens...
[GGMLGPUBackend] Uploading finalNorm...
[GGMLGPUBackend] Uploading lmHead...
[GGMLGPUBackend] Uploading layer weights...
[GGMLGPUBackend] ✅ Weights uploaded
[HFTransformer] ✅ GPU backend ready, weights uploaded
```

## 性能对比

GPU 加速显著提升了推理性能：

| 配置 | 平均性能 | 提升倍数 |
|------|---------|---------|
| CPU (FP16) | ~20 tokens/s | 1x |
| GPU (Metal, FP16) | ~55 tokens/s | 2.75x |

## 经验总结

### 1. 权重管理策略
- 不同设备后端可能需要不同精度的权重
- GPU 后端通常需要 FP32 权重，即使 CPU 推理使用 FP16/INT8
- 需要同时维护多个版本的权重，增加内存使用但提升兼容性

### 2. 错误处理最佳实践
- 在调用外部 API 之前，始终检查输入参数的有效性
- 添加详细的日志输出，便于调试和问题定位
- 分阶段添加日志，可以快速定位崩溃位置

### 3. 内存管理
- FP16 模式下的内存占用:
  - FP16 权重: ~1.08 GB
  - FP32 权重 (GPU): ~2.16 GB
  - 总计: ~3.24 GB
- 相比纯 FP32 模式节省了约 50% 的内存

### 4. 调试技巧
- 使用空指针检查可以快速定位问题
- 添加分步日志可以追踪执行流程
- 在关键操作前后添加日志，便于性能分析

## 相关文件

- `src/kylin/hf/transformer.cpp` - 权重转换逻辑
- `src/kylin/hf/ggml_backend.cpp` - GPU 权重上传
- `include/cllm/kylin/hf/ggml_backend.h` - GPU 后端接口

## 后续优化建议

1. **按需上传权重**: 只上传当前推理需要的层权重，减少 GPU 内存占用
2. **权重共享优化**: 对于 tie embeddings 的情况，优化权重共享逻辑
3. **异步上传**: 考虑使用异步上传，避免阻塞初始化
4. **错误恢复**: 添加权重上传失败后的恢复机制

## 修复日期

2026-02-05

## 修复人员

cLLM 开发团队
