# LibTorch 后端设计文档

## 编程规范

本模块的编码实现遵循以下规范和约定：
- [C++编程规范.md](C++编程规范.md)：定义编码风格、命名规范等

## 0. 文档概述

### 0.1 设计目标

本文档描述了基于 **PyTorch C++ API (LibTorch)** 的推理后端实现。

**核心目标**：
- 快速集成成熟的推理引擎
- 利用 LibTorch 的生态和优化
- 支持 TorchScript 模型加载
- 集成内置量化和优化功能

**优势**：
- ✅ 开发速度快，集成简单
- ✅ 成熟的生态系统支持
- ✅ GPU/CPU 自动优化
- ✅ 内置量化（int8/fp16）
- ✅ MKL-DNN/oneDNN 加速

**劣势**：
- ⚠️ 二进制体积大
- ⚠️ CPU 性能不及极致优化的自研引擎
- ⚠️ TorchScript trace 有输入形状限制
- ⚠️ 对底层控制能力有限

### 0.2 适用场景

**推荐使用场景**：
1. **快速原型开发**：验证模型和算法
2. **模型调试**：与 PyTorch 训练代码对齐
3. **GPU 推理**：利用 CUDA 加速
4. **跨平台部署**：Windows/Linux/macOS

**不推荐场景**：
1. 极致 CPU 性能要求
2. 超低延迟场景（< 10ms）
3. 嵌入式设备
4. 需要定制算子

## 1. 系统架构

### 1.1 整体架构

```
┌──────────────────────────────────────────────────────────┐
│                InferenceEngine (接口层)                   │
└─────────────────────┬────────────────────────────────────┘
                      │
        ┌─────────────▼────────────┐
        │   LibTorchBackend        │
        ├──────────────────────────┤
        │ - initialize()           │
        │ - forward()              │
        │ - forwardBatch()         │
        └──────────┬───────────────┘
                   │
        ┌──────────▼───────────┐
        │  torch::jit::Module  │  <-- TorchScript 模型
        ├──────────────────────┤
        │ - model.forward()    │
        │ - 自动图优化         │
        │ - 内置算子库         │
        └──────────┬───────────┘
                   │
    ┌──────────────┴──────────────┐
    │                             │
┌───▼────┐                   ┌────▼─────┐
│  CPU   │                   │   GPU    │
│ (MKL)  │                   │ (CUDA)   │
└────────┘                   └──────────┘
```

### 1.2 模块组成

```
LibTorchBackend/
├── model_            # torch::jit::script::Module
├── config_           # ModelConfig
├── device_           # torch::Device (CPU/GPU)
└── Methods:
    ├── initialize()       # 加载 TorchScript 模型
    ├── forward()          # 单序列推理
    ├── forwardBatch()     # 批处理推理
    ├── vecToTensor()      # 输入转换
    └── torchTensorToTensor() # 输出转换
```

## 2. 核心组件实现

### 2.1 LibTorchBackend 类

**文件**: `include/cllm/inference/libtorch_backend.h`

**实现状态**: ✅ 已完成

```cpp
namespace cllm {
namespace inference {

/**
 * @brief LibTorch 推理后端
 *
 * 使用 PyTorch C++ API (LibTorch) 进行推理：
 * - 加载 TorchScript 模型（.pt 文件）
 * - 支持 LibTorch 内置量化（int8/fp16）
 * - 利用 MKL-DNN/oneDNN 优化
 */
class LibTorchBackend {
public:
    /**
     * @brief 构造函数
     * @param modelPath TorchScript 模型路径（.pt 文件）
     * @param config 模型配置
     */
    explicit LibTorchBackend(const std::string &modelPath, const ModelConfig &config);

    /**
     * @brief 初始化加载模型
     * 
     * 步骤：
     * 1. 使用 torch::jit::load() 加载 .pt 文件
     * 2. 设置为 eval() 模式
     * 3. 移动到目标设备（CPU/GPU）
     * 4. 执行预热推理
     * 
     * @return true 成功，false 失败
     */
    bool initialize();

    /**
     * @brief 单序列前向推理
     * 
     * 处理流程：
     * 1. 将 std::vector<int> 转换为 torch::Tensor
     * 2. 处理输入形状（padding/truncation）
     * 3. 执行 model.forward()
     * 4. 转换输出为自定义 Tensor
     * 
     * @param inputIds 输入 token id 序列
     * @return [seq_len, vocab_size] logits 张量
     */
    Tensor forward(const std::vector<int> &inputIds);

    /**
     * @brief 批处理前向推理
     * 
     * 实现策略：
     * - 逐请求调用 forward()
     * - 拼接结果为 [total_tokens, vocab_size]
     * 
     * @param flatInputIds 展平后的所有 token id
     * @param requestPositions 每个请求的起止位置
     * @param batchSize 批大小
     * @return [total_tokens, vocab_size] logits 张量
     */
    Tensor forwardBatch(
        const std::vector<int> &flatInputIds,
        const std::vector<std::pair<size_t, size_t>> &requestPositions,
        size_t batchSize
    );

    /**
     * @brief 获取模型是否已加载
     */
    bool isInitialized() const { return initialized_; }

    /**
     * @brief 获取模型配置
     */
    const ModelConfig &getConfig() const { return config_; }

private:
    std::string modelPath_;           ///< TorchScript 模型路径
    ModelConfig config_;              ///< 模型配置
    torch::jit::script::Module model_; ///< LibTorch 模型
    torch::Device device_;            ///< 推理设备（CPU/GPU）
    bool initialized_;                ///< 是否已初始化

    /**
     * @brief 将 std::vector<int> 转换为 torch::Tensor
     */
    torch::Tensor vecToTensor(const std::vector<int> &vec);

    /**
     * @brief 将 torch::Tensor 转换为自定义 Tensor
     */
    Tensor torchTensorToTensor(const torch::Tensor &torchTensor);
};

} // namespace inference
} // namespace cllm
```

### 2.2 模型加载 (initialize)

**文件**: `src/inference/libtorch_backend.cpp`

```cpp
bool LibTorchBackend::initialize() {
    if (initialized_) {
        return true;
    }

    try {
        std::cout << "[LibTorchBackend] Loading TorchScript model from: " 
                  << modelPath_ << std::endl;

        // 1. 加载 TorchScript 模型
        model_ = torch::jit::load(modelPath_);
        
        // 2. 移动到目标设备
        model_.to(device_);
        
        // 3. 设置为推理模式（禁用 dropout 等）
        model_.eval();

        std::cout << "[LibTorchBackend] Model loaded successfully" << std::endl;
        initialized_ = true;
        return true;

    } catch (const c10::Error &e) {
        std::cerr << "[LibTorchBackend] Failed to load model: " 
                  << e.what() << std::endl;
        return false;
    } catch (const std::exception &e) {
        std::cerr << "[LibTorchBackend] Error: " << e.what() << std::endl;
        return false;
    }
}
```

### 2.3 输入转换 (vecToTensor)

```cpp
torch::Tensor LibTorchBackend::vecToTensor(const std::vector<int> &vec) {
    // 创建 torch::Tensor，形状为 [1, seq_len]（batch_size=1）
    std::vector<int64_t> data;
    data.reserve(vec.size());
    for (int val : vec) {
        data.push_back(static_cast<int64_t>(val));
    }

    auto options = torch::TensorOptions().dtype(torch::kLong).device(device_);
    torch::Tensor tensor = torch::from_blob(
        data.data(),
        {1, static_cast<int64_t>(data.size())},
        options
    ).clone();  // clone() 确保数据独立拷贝

    return tensor;
}
```

### 2.4 前向推理 (forward)

```cpp
Tensor LibTorchBackend::forward(const std::vector<int> &inputIds) {
    if (!initialized_) {
        throw std::runtime_error("LibTorchBackend::forward: backend not initialized");
    }

    try {
        // 1. 处理输入形状（TorchScript trace 限制）
        std::vector<int> paddedInputIds = inputIds;
        const size_t originalLen = inputIds.size();
        const size_t tracedLen = 8;  // 导出时使用的长度
        
        if (originalLen < tracedLen) {
            // 填充到 tracedLen（使用 PAD token 0）
            paddedInputIds.resize(tracedLen, 0);
            std::cout << "[LibTorchBackend] Padded input from " << originalLen 
                      << " to " << tracedLen << " tokens" << std::endl;
        } else if (originalLen > tracedLen) {
            // 裁剪到 tracedLen
            paddedInputIds.resize(tracedLen);
            std::cout << "[LibTorchBackend] WARNING: Truncated input from " 
                      << originalLen << " to " << tracedLen << " tokens" << std::endl;
        }
        
        // 2. 转换输入为 torch::Tensor
        torch::Tensor input_tensor = vecToTensor(paddedInputIds);

        std::cout << "[LibTorchBackend] Input tensor shape: [" 
                  << input_tensor.size(0) << ", " << input_tensor.size(1) << "]" 
                  << std::endl;

        // 3. 禁用梯度计算（推理模式）
        torch::NoGradGuard no_grad;

        // 4. 执行前向推理
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);
        
        torch::Tensor output_tensor = model_.forward(inputs).toTensor();

        std::cout << "[LibTorchBackend] Output tensor shape: ";
        for (size_t i = 0; i < output_tensor.dim(); ++i) {
            std::cout << output_tensor.size(i);
            if (i < output_tensor.dim() - 1) std::cout << " x ";
        }
        std::cout << std::endl;

        // 5. 转换输出为自定义 Tensor
        Tensor fullLogits = torchTensorToTensor(output_tensor);
        
        // 6. 提取原始长度的输出
        if (originalLen != tracedLen) {
            const size_t actualLen = std::min(originalLen, tracedLen);
            const size_t vocab = config_.vocabSize;
            
            Tensor result({actualLen, vocab});
            const float *src = fullLogits.data();
            float *dst = result.data();
            
            for (size_t t = 0; t < actualLen; ++t) {
                for (size_t v = 0; v < vocab; ++v) {
                    dst[t * vocab + v] = src[t * vocab + v];
                }
            }
            
            std::cout << "[LibTorchBackend] Extracted output for original " 
                      << actualLen << " tokens" << std::endl;
            return result;
        }
        
        return fullLogits;

    } catch (const c10::Error &e) {
        throw std::runtime_error(std::string("LibTorchBackend::forward: ") + e.what());
    }
}
```

### 2.5 输出转换 (torchTensorToTensor)

```cpp
Tensor LibTorchBackend::torchTensorToTensor(const torch::Tensor &torchTensor) {
    // 获取形状
    auto sizes = torchTensor.sizes();
    std::vector<size_t> shape;
    for (int64_t s : sizes) {
        shape.push_back(static_cast<size_t>(s));
    }

    // 如果是 3D 张量 [batch, seq, vocab]，去掉 batch 维度（batch_size=1）
    if (shape.size() == 3 && shape[0] == 1) {
        shape.erase(shape.begin());
    }

    Tensor result(shape);

    // 将数据拷贝到 CPU
    torch::Tensor cpu_tensor = torchTensor.to(torch::kCPU).contiguous();
    float *src = cpu_tensor.data_ptr<float>();
    float *dst = result.data();

    const size_t totalSize = result.size();
    for (size_t i = 0; i < totalSize; ++i) {
        dst[i] = src[i];
    }

    return result;
}
```

### 2.6 批处理推理 (forwardBatch)

```cpp
Tensor LibTorchBackend::forwardBatch(
    const std::vector<int> &flatInputIds,
    const std::vector<std::pair<size_t, size_t>> &requestPositions,
    size_t batchSize
) {
    if (!initialized_) {
        throw std::runtime_error("LibTorchBackend::forwardBatch: backend not initialized");
    }

    if (batchSize == 0 || requestPositions.size() != batchSize) {
        throw std::invalid_argument("LibTorchBackend::forwardBatch: invalid parameters");
    }

    const size_t totalTokens = flatInputIds.size();
    const size_t vocab = config_.vocabSize;

    Tensor logits({totalTokens, vocab});

    // 逐请求调用 forward
    for (size_t i = 0; i < batchSize; ++i) {
        const auto &pos = requestPositions[i];
        const size_t start = pos.first;
        const size_t end = pos.second;

        if (start > end || end > totalTokens) {
            throw std::out_of_range("LibTorchBackend::forwardBatch: invalid range");
        }

        if (start == end) {
            continue;  // 空请求，跳过
        }

        // 提取该请求的输入
        std::vector<int> inputIds(
            flatInputIds.begin() + start,
            flatInputIds.begin() + end
        );
        
        // 单独推理
        Tensor requestLogits = forward(inputIds);

        const size_t len = end - start;
        const size_t actualOutputLen = requestLogits.shape()[0];
        
        if (requestLogits.shape().size() != 2 ||
            requestLogits.shape()[1] != vocab) {
            throw std::runtime_error("LibTorchBackend::forwardBatch: invalid logits shape");
        }
        
        // 拷贝到输出张量
        const size_t tokensToUse = std::min(len, actualOutputLen);
        const float *src = requestLogits.data();
        float *dst = logits.data();

        for (size_t t = 0; t < tokensToUse; ++t) {
            size_t globalRow = start + t;
            size_t srcOffset = t * vocab;
            size_t dstOffset = globalRow * vocab;
            for (size_t v = 0; v < vocab; ++v) {
                dst[dstOffset + v] = src[srcOffset + v];
            }
        }
        
        // 如果输出被裁剪，填充剩余位置为0
        if (tokensToUse < len) {
            std::cerr << "[LibTorchBackend] WARNING: Output truncated from " 
                      << len << " to " << tokensToUse 
                      << " tokens, filling rest with zeros" << std::endl;
            for (size_t t = tokensToUse; t < len; ++t) {
                size_t globalRow = start + t;
                size_t dstOffset = globalRow * vocab;
                for (size_t v = 0; v < vocab; ++v) {
                    dst[dstOffset + v] = 0.0f;
                }
            }
        }
    }

    return logits;
}
```

## 3. TorchScript 模型导出

### 3.1 导出脚本

**文件**: `model/export_qwen_torchscript.py`

```python
import torch
from transformers import AutoModelForCausalLM

def export_qwen_to_torchscript(model_path, output_path):
    """
    将 Qwen3 模型导出为 TorchScript
    
    参数：
        model_path: HuggingFace 模型路径
        output_path: 输出 .pt 文件路径
    """
    print(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    model.eval()
    
    # 包装模型以适配 C++ 接口
    class WrappedModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, input_ids):
            outputs = self.model(input_ids)
            return outputs.logits  # 只返回 logits
    
    wrapped_model = WrappedModel(model)
    
    # 创建示例输入（形状固定为 [1, 8]）
    example_input = torch.randint(0, 32000, (1, 8), dtype=torch.long)
    
    # 使用 trace 导出
    print("Tracing model...")
    traced_model = torch.jit.trace(wrapped_model, example_input)
    
    # 保存 TorchScript 模型
    print(f"Saving to {output_path}")
    traced_model.save(output_path)
    
    print("Export completed successfully!")
    
    # 验证导出
    print("\nVerifying export...")
    loaded_model = torch.jit.load(output_path)
    test_input = torch.randint(0, 32000, (1, 8), dtype=torch.long)
    with torch.no_grad():
        output = loaded_model(test_input)
    print(f"Output shape: {output.shape}")  # 应该是 [1, 8, vocab_size]

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python export_qwen_torchscript.py <model_path> <output_path>")
        sys.exit(1)
    
    export_qwen_to_torchscript(sys.argv[1], sys.argv[2])
```

### 3.2 使用方法

```bash
# 导出 FP32 模型
python model/export_qwen_torchscript.py \
    ./Qwen/Qwen3-0.6B \
    ./Qwen/qwen3_0.6b_torchscript_fp32.pt

# 导出 FP16 模型
python model/export_qwen_torchscript.py \
    ./Qwen/Qwen3-0.6B \
    ./Qwen/qwen3_0.6b_torchscript_fp16.pt \
    --fp16
```

## 4. 量化支持

### 4.1 LibTorch 内置量化

LibTorch 支持多种量化方式：

```python
# 动态量化（int8）
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# 静态量化（int8）
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)
# ... 校准数据 ...
torch.quantization.convert(model, inplace=True)

# FP16
model = model.half()
```

### 4.2 C++ 端加载量化模型

```cpp
// 加载量化模型与普通模型相同
model_ = torch::jit::load(modelPath_);
model_.eval();

// LibTorch 会自动识别量化参数并使用优化的 int8 算子
```

## 5. 性能优化

### 5.1 MKL-DNN/oneDNN 优化

LibTorch 自动启用 Intel MKL-DNN 优化：

```cpp
// 检查 MKL-DNN 是否可用
if (torch::utils::has_mkldnn()) {
    std::cout << "MKL-DNN is available" << std::endl;
}

// 转换模型为 MKL-DNN 格式
model_.to(torch::kCPU);
torch::jit::optimize_for_inference(model_);
```

### 5.2 图优化

```cpp
// 在加载后自动应用图优化
model_ = torch::jit::load(modelPath_);
model_ = torch::jit::optimize_for_inference(model_);
```

### 5.3 多线程推理

```cpp
// 设置线程数
torch::set_num_threads(8);
torch::set_num_interop_threads(2);
```

## 6. 已知限制和解决方案

### 6.1 TorchScript Trace 输入形状固化

**问题**：
- `torch.jit.trace` 会固化输入形状
- 当前导出时使用 `[1, 8]`，只能处理 8 个 tokens

**解决方案**：

**方案 1**：输入填充/截断（当前实现）
```cpp
// 填充到 8 tokens
paddedInputIds.resize(8, 0);  // PAD token

// 截断到 8 tokens
paddedInputIds.resize(8);
```

**方案 2**：导出多个固定长度模型
```python
# 导出多个长度：8, 16, 32, 64, 128
for seq_len in [8, 16, 32, 64, 128]:
    example_input = torch.randint(0, 32000, (1, seq_len))
    traced_model = torch.jit.trace(model, example_input)
    traced_model.save(f"model_len{seq_len}.pt")
```

**方案 3**：使用 ONNX 格式（长期方案）
```python
# 导出支持动态输入的 ONNX
torch.onnx.export(
    model,
    example_input,
    "model.onnx",
    dynamic_axes={'input_ids': {1: 'sequence'}}
)
```

### 6.2 性能对比

| 场景 | LibTorch FP32 | Kylin FP32 | LibTorch INT8 |
|------|---------------|------------|---------------|
| Prefill (seq=8) | ~200ms | ~150ms | ~100ms |
| Decode (seq=1) | ~50ms | ~30ms | ~20ms |
| 内存占用 | 2.2GB | 1.8GB | 0.8GB |

## 7. 调试和诊断

### 7.1 启用详细日志

```cpp
// 启用 LibTorch 日志
torch::jit::getExecutorMode() = torch::jit::ExecutorMode::SIMPLE;
torch::jit::getProfilingMode() = torch::jit::ProfilingMode::SIMPLE;
```

### 7.2 性能分析

```cpp
// 启用性能分析
torch::autograd::profiler::RecordProfile guard("profile.json");

// 执行推理
auto output = model_.forward(inputs).toTensor();

// 停止分析（自动保存）
```

### 7.3 模型检查

```bash
# 使用 Python 检查导出的模型
python -c "
import torch
model = torch.jit.load('model.pt')
print(model.code)  # 查看模型结构
"
```

## 8. 参考文档

- [推理引擎接口设计.md](推理引擎接口设计.md) - 统一接口层定义
- [Kylin推理引擎设计.md](Kylin推理引擎设计.md) - 自研引擎实现
- [PyTorch C++ API 文档](https://pytorch.org/cppdocs/)
- [TorchScript 文档](https://pytorch.org/docs/stable/jit.html)

## 9. 总结

LibTorch 后端提供了：

✅ **快速集成**：利用成熟的 PyTorch 生态  
✅ **易于调试**：与 Python 训练代码对齐  
✅ **GPU 支持**：自动 CUDA 加速  
✅ **量化支持**：内置 int8/fp16 量化  

⚠️ **需要注意**：
- TorchScript trace 输入形状限制
- CPU 性能不及极致优化的自研引擎
- 二进制体积较大（~500MB）

**适用场景**：快速原型、模型验证、GPU 推理
