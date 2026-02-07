# Kylin 模块重构方案

> **状态**: ✅ 已完成 (2026-01-24)

## 1. 重构目标

- **HuggingFace 模型优先**: 作为主要支持路径，提供最佳体验
- **GGUF 兼容保留**: 作为次要路径，保留功能但降低优先级
- **清晰模块边界**: 按功能分层，便于维护和扩展
- **减少代码耦合**: 核心算子与模型格式解耦

## 2. 新目录结构

```
src/kylin/
├── core/                   # 核心基础设施
│   ├── kernels.cpp         # 计算内核（matmul, softmax, silu）
│   ├── quantization.cpp    # 量化/反量化工具
│   └── tensor_stats.cpp    # 调试统计
│
├── ops/                    # 算子层（与模型格式无关）
│   ├── attention.cpp       # 基础 attention 算子
│   ├── attention_graph.cpp # 计算图优化
│   ├── feed_forward.cpp    # FFN 层
│   ├── rope.cpp            # RoPE 位置编码
│   └── kv_cache_ops.cpp    # KV Cache 操作
│
├── hf/                     # HuggingFace 后端（主要）
│   ├── config.cpp          # config.json 解析
│   ├── safetensors_loader.cpp  # .safetensors 加载
│   └── transformer.cpp     # HF Transformer 实现
│
├── gguf/                   # GGUF 后端（保留）
│   ├── loader.cpp          # GGUF 文件加载
│   ├── context.cpp         # GGML 上下文
│   ├── operator.cpp        # GGML/Native 算子
│   └── transformer.cpp     # GGUF Transformer 实现
│
└── model/                  # 模型抽象层
    ├── model_loader.cpp    # 模型加载接口
    ├── transformer_block.cpp   # Transformer 块抽象
    └── transformer_model.cpp   # 模型抽象

include/cllm/kylin/
├── core/
│   ├── kernels.h
│   ├── quantization.h
│   ├── tensor.h            # Tensor 类
│   ├── operators.h         # 算子接口
│   └── tensor_stats.h
│
├── ops/
│   ├── attention.h
│   ├── attention_graph.h
│   ├── feed_forward.h
│   ├── rope.h
│   └── kv_cache_ops.h
│
├── hf/
│   ├── config.h
│   ├── safetensors_loader.h
│   └── transformer.h
│
├── gguf/
│   ├── loader.h
│   ├── context.h
│   ├── operator.h
│   └── transformer.h
│
└── model/
    ├── model_loader.h
    ├── transformer_block.h
    └── transformer_model.h
```

## 3. 文件映射表

### 3.1 源文件 (.cpp)

| 原路径 | 新路径 | 说明 |
|--------|--------|------|
| `kernels.cpp` | `core/kernels.cpp` | 不变 |
| `quantization.cpp` | `core/quantization.cpp` | 不变 |
| `tensor_stats.cpp` | `core/tensor_stats.cpp` | 不变 |
| `attention.cpp` | `ops/attention.cpp` | 不变 |
| `attention_graph.cpp` | `ops/attention_graph.cpp` | 不变 |
| `feed_forward.cpp` | `ops/feed_forward.cpp` | 不变 |
| `rope.cpp` | `ops/rope.cpp` | 不变 |
| `kv_cache_ops.cpp` | `ops/kv_cache_ops.cpp` | 不变 |
| `hf_config.cpp` | `hf/config.cpp` | 重命名 |
| `safetensors_loader.cpp` | `hf/safetensors_loader.cpp` | 不变 |
| `hf_transformer.cpp` | `hf/transformer.cpp` | 重命名 |
| `gguf_loader.cpp` | `gguf/loader.cpp` | 重命名 |
| `ggml_context.cpp` | `gguf/context.cpp` | 重命名 |
| `ggml_operator.cpp` | `gguf/ggml_operator.cpp` | 保留原名 |
| `native_operator.cpp` | `gguf/native_operator.cpp` | 保留原名 |
| `operator_interface.cpp` | `gguf/operator_interface.cpp` | 保留原名 |
| `ggml_transformer.cpp` | `gguf/transformer.cpp` | 重命名 |
| `model_loader.cpp` | `model/model_loader.cpp` | 不变 |
| `transformer_block.cpp` | `model/transformer_block.cpp` | 不变 |
| `transformer_model.cpp` | `model/transformer_model.cpp` | 不变 |

### 3.2 头文件 (.h)

类似映射，增加对应子目录。

## 4. 实施步骤

### Phase 1: 创建目录结构
```bash
mkdir -p src/kylin/{core,ops,hf,gguf,model}
mkdir -p include/cllm/kylin/{core,ops,hf,gguf,model}
```

### Phase 2: 移动文件
按映射表移动文件，同时更新文件内的 `#include` 路径。

### Phase 3: 更新 CMakeLists.txt
更新源文件路径列表。

### Phase 4: 更新外部引用
更新 `kylin_backend.cpp` 等外部文件的 include 路径。

### Phase 5: 编译验证
确保编译通过并测试功能正常。

## 5. 兼容性考虑

为减少对现有代码的影响，可在旧位置创建转发头文件：

```cpp
// include/cllm/kylin/hf_transformer.h (兼容旧路径)
#pragma once
#include "cllm/kylin/hf/hf_transformer_model.h"
```

## 6. 时间线

- Phase 1-2: 目录创建和文件移动
- Phase 3-4: CMake 和引用更新
- Phase 5: 验证测试
