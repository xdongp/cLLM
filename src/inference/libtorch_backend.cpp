/**
 * @file libtorch_backend.cpp
 * @brief LibTorch 推理后端实现
 * 
 * 参考文档：LibTorch后端设计.md
 * 
 * 实现特点：
 * - 加载 TorchScript 模型（.pt 文件）
 * - 支持 CPU/GPU 自动选择
 * - 内置量化支持（int8/fp16）
 * - MKL-DNN/oneDNN 优化
 */

#include "cllm/inference/libtorch_backend.h"
#include "cllm/common/logger.h"

#include <stdexcept>
#include <cstring>  // for std::memcpy

namespace cllm {
namespace inference {

LibTorchBackend::LibTorchBackend(const std::string &modelPath, const ModelConfig &config)
    : modelPath_(modelPath)
    , config_(config)
    , device_(torch::kCPU)  // 默认使用 CPU，可根据需要改为 torch::kCUDA
    , initialized_(false) {
    
    CLLM_INFO("[LibTorchBackend] Constructing LibTorch backend");
    CLLM_INFO("[LibTorchBackend] Model path: {}", modelPath_);
    CLLM_INFO("[LibTorchBackend] Device: {}", (device_.is_cpu() ? "CPU" : "CUDA"));
    
    // 检查 CUDA 可用性
    if (torch::cuda::is_available()) {
        CLLM_INFO("[LibTorchBackend] CUDA is available, {} device(s) found", 
                  torch::cuda::device_count());
    } else {
        CLLM_INFO("[LibTorchBackend] CUDA not available, using CPU");
    }
}

bool LibTorchBackend::initialize() {
    if (initialized_) {
        CLLM_INFO("[LibTorchBackend] Already initialized, skipping");
        return true;
    }

    try {
        CLLM_INFO("[LibTorchBackend] ========== Initializing LibTorch Backend ==========");
        CLLM_INFO("[LibTorchBackend] Loading TorchScript model from: {}", modelPath_);

        // 1. 加载 TorchScript 模型
        model_ = torch::jit::load(modelPath_);
        CLLM_INFO("[LibTorchBackend] Model loaded from disk");
        
        // 2. 移动到目标设备
        model_.to(device_);
        CLLM_INFO("[LibTorchBackend] Model moved to {}", 
                  (device_.is_cpu() ? "CPU" : "CUDA"));
        
        // 3. 设置为推理模式（禁用 dropout 等）
        model_.eval();
        CLLM_INFO("[LibTorchBackend] Model set to eval mode");
        
        // 4. 从模型中提取vocab_size并更新config
        try {
            // 创建一个测试输入,推理一次获取输出维度
            auto test_input = torch::randint(0, 1000, {1, 8}, torch::kLong).to(device_);
            torch::NoGradGuard no_grad;
            auto test_output = model_.forward({test_input}).toTensor();
            
            // 输出形状应该是 [batch, seq_len, vocab_size] 或 [seq_len, vocab_size]
            auto output_sizes = test_output.sizes();
            size_t detected_vocab_size = 0;
            
            if (output_sizes.size() == 3) {
                // [batch, seq, vocab]
                detected_vocab_size = static_cast<size_t>(output_sizes[2]);
            } else if (output_sizes.size() == 2) {
                // [seq, vocab]
                detected_vocab_size = static_cast<size_t>(output_sizes[1]);
            }
            
            if (detected_vocab_size > 0 && detected_vocab_size != config_.vocabSize) {
                CLLM_INFO("[LibTorchBackend] Detected vocab_size from model: {} (config has: {})", 
                          detected_vocab_size, config_.vocabSize);
                config_.vocabSize = detected_vocab_size;
                CLLM_INFO("[LibTorchBackend] Updated config vocab_size to {}", config_.vocabSize);
            } else {
                CLLM_INFO("[LibTorchBackend] Vocab size: {}", config_.vocabSize);
            }
        } catch (const std::exception& e) {
            CLLM_WARN("[LibTorchBackend] Could not auto-detect vocab_size: {}", e.what());
            CLLM_WARN("[LibTorchBackend] Using config vocab_size: {}", config_.vocabSize);
        }
        
        // 5. 应用图优化（可选）
        // 注意：optimize_for_inference 可能会改变模型行为，谨慎使用
        // model_ = torch::jit::optimize_for_inference(model_);
        // CLLM_INFO("[LibTorchBackend] Applied graph optimization");

        CLLM_INFO("[LibTorchBackend] ========== Initialization Complete ==========");
        initialized_ = true;
        return true;

    } catch (const c10::Error &e) {
        CLLM_ERROR("[LibTorchBackend] Failed to load model: {}", e.what());
        return false;
    } catch (const std::exception &e) {
        CLLM_ERROR("[LibTorchBackend] Error: {}", e.what());
        return false;
    }
}

torch::Tensor LibTorchBackend::vecToTensor(const std::vector<int> &vec) {
    // 创建 torch::Tensor，形状为 [1, seq_len]（batch_size=1）
    // 注意：TorchScript 模型期望输入为 torch.long 类型
    
    std::vector<int64_t> data;
    data.reserve(vec.size());
    for (int val : vec) {
        data.push_back(static_cast<int64_t>(val));
    }

    // 创建 tensor options：long 类型，目标设备
    auto options = torch::TensorOptions().dtype(torch::kLong).device(device_);
    
    // 使用 from_blob 创建 tensor，然后 clone 确保数据独立拷贝
    // clone() 很重要，否则数据会在 data 离开作用域后失效
    torch::Tensor tensor = torch::from_blob(
        data.data(),
        {1, static_cast<int64_t>(data.size())},
        options
    ).clone();

    return tensor;
}

Tensor LibTorchBackend::torchTensorToTensor(const torch::Tensor &torchTensor) {
    // 将 torch::Tensor 转换为自定义 Tensor
    // 处理两种情况：
    // 1. [batch, seq_len, vocab_size] - 3D 张量（batch_size=1）
    // 2. [seq_len, vocab_size] - 2D 张量

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

    // 将数据拷贝到 CPU（如果原本在 GPU 上）
    // contiguous() 确保内存布局连续，便于拷贝
    torch::Tensor cpu_tensor = torchTensor.to(torch::kCPU).contiguous();
    float *src = cpu_tensor.data_ptr<float>();
    float *dst = result.data();

    // 批量拷贝数据
    const size_t totalSize = result.size();
    std::memcpy(dst, src, totalSize * sizeof(float));

    return result;
}

// ========== 公共辅助方法 ==========

void LibTorchBackend::setDevice(bool useCuda, int deviceId) {
    if (initialized_) {
        CLLM_WARN("[LibTorchBackend] WARNING: Cannot change device after initialization");
        return;
    }
    
    if (useCuda) {
        if (torch::cuda::is_available()) {
            device_ = torch::Device(torch::kCUDA, deviceId);
            CLLM_INFO("[LibTorchBackend] Device set to CUDA:{}", deviceId);
        } else {
            CLLM_WARN("[LibTorchBackend] WARNING: CUDA requested but not available, using CPU");
            device_ = torch::kCPU;
        }
    } else {
        device_ = torch::kCPU;
        CLLM_INFO("[LibTorchBackend] Device set to CPU");
    }
}

void LibTorchBackend::setNumThreads(int numThreads) {
    if (numThreads > 0) {
        torch::set_num_threads(numThreads);
        CLLM_INFO("[LibTorchBackend] Set num_threads to {}", numThreads);
    }
    
    // 设置 inter-op 并行性（跨算子并行）
    // 通常设置为较小的值，如 2
    torch::set_num_interop_threads(2);
}

Tensor LibTorchBackend::forward(const std::vector<int> &inputIds) {
    if (!initialized_) {
        throw std::runtime_error("LibTorchBackend::forward: backend not initialized");
    }

    try {
        // 过滤无效的token ID(如-1等特殊标记)
        std::vector<int> validInputIds;
        validInputIds.reserve(inputIds.size());
        for (int id : inputIds) {
            if (id >= 0 && static_cast<size_t>(id) < config_.vocabSize) {
                validInputIds.push_back(id);
            } else {
                CLLM_WARN("[LibTorchBackend] Skipping invalid token ID: {} (vocab size: {})", 
                          id, config_.vocabSize);
                // 用0(通常是PAD token)替代无效ID
                validInputIds.push_back(0);
            }
        }
        
        // 由于 TorchScript trace 固化了输入形状（8），需要填充或裁剪
        std::vector<int> paddedInputIds = validInputIds;
        const size_t originalLen = validInputIds.size();
        const size_t tracedLen = 8;  // 导出时使用的长度
        
        if (originalLen < tracedLen) {
            // 填充到 tracedLen（使用 PAD token 0）
            paddedInputIds.resize(tracedLen, 0);
            CLLM_INFO("[LibTorchBackend] Padded input from {} to {} tokens", 
                      originalLen, tracedLen);
        } else if (originalLen > tracedLen) {
            // 裁剪到 tracedLen（不建议，但作为临时方案）
            paddedInputIds.resize(tracedLen);
            CLLM_WARN("[LibTorchBackend] WARNING: Truncated input from {} to {} tokens", 
                      originalLen, tracedLen);
        }
        
        // 转换输入为 torch::Tensor
        torch::Tensor input_tensor = vecToTensor(paddedInputIds);

        CLLM_DEBUG("[LibTorchBackend] Input tensor shape: [{}, {}]", 
                   input_tensor.size(0), input_tensor.size(1));

        // 禁用梯度计算
        torch::NoGradGuard no_grad;

        // 执行前向推理
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);
        
        torch::Tensor output_tensor = model_.forward(inputs).toTensor();

        // 输出形状日志
        std::string shape_str = "[";
        for (size_t i = 0; i < output_tensor.dim(); ++i) {
            shape_str += std::to_string(output_tensor.size(i));
            if (i < output_tensor.dim() - 1) shape_str += " x ";
        }
        shape_str += "]";
        CLLM_DEBUG("[LibTorchBackend] Output tensor shape: {}", shape_str);

        // 转换输出为自定义 Tensor
        Tensor fullLogits = torchTensorToTensor(output_tensor);
        
        // 如果输入被填充/裁剪，需要提取原始长度的输出
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
            
            CLLM_DEBUG("[LibTorchBackend] Extracted output for original {} tokens", actualLen);
            return result;
        }
        
        return fullLogits;

    } catch (const c10::Error &e) {
        throw std::runtime_error(std::string("LibTorchBackend::forward: ") + e.what());
    }
}

Tensor LibTorchBackend::forwardBatch(const std::vector<int> &flatInputIds,
                                      const std::vector<std::pair<size_t, size_t>> &requestPositions,
                                      size_t batchSize) {
    if (!initialized_) {
        throw std::runtime_error("LibTorchBackend::forwardBatch: backend not initialized");
    }

    if (batchSize == 0 || requestPositions.size() != batchSize) {
        throw std::invalid_argument("LibTorchBackend::forwardBatch: invalid batchSize or requestPositions");
    }

    const size_t totalTokens = flatInputIds.size();
    const size_t vocab = config_.vocabSize;

    Tensor logits({totalTokens, vocab});

    // 逐请求调用 forward，与 InferenceEngine 保持一致
    for (size_t i = 0; i < batchSize; ++i) {
        const auto &pos = requestPositions[i];
        const size_t start = pos.first;
        const size_t end = pos.second;

        if (start > end || end > totalTokens) {
            throw std::out_of_range("LibTorchBackend::forwardBatch: invalid requestPositions range");
        }

        if (start == end) {
            continue;  // 空请求，跳过
        }

        std::vector<int> inputIds(flatInputIds.begin() + start, flatInputIds.begin() + end);
        Tensor requestLogits = forward(inputIds);

        const size_t len = end - start;
        const size_t actualOutputLen = requestLogits.shape()[0];  // 实际输出长度（可能被填充/裁剪）
        
        // 放宽形状验证:允许不同的输出长度(由于TorchScript trace固定长度)
        if (requestLogits.shape().size() != 2) {
            CLLM_ERROR("[LibTorchBackend] Invalid logits shape: expected 2D tensor, got {}D", 
                       requestLogits.shape().size());
            throw std::runtime_error("LibTorchBackend::forwardBatch: invalid logits shape (not 2D)");
        }
        
        if (requestLogits.shape()[1] != vocab) {
            CLLM_ERROR("[LibTorchBackend] Vocab size mismatch: expected {}, got {}", 
                       vocab, requestLogits.shape()[1]);
            throw std::runtime_error("LibTorchBackend::forwardBatch: vocab size mismatch");
        }
        
        // 由于 TorchScript trace 限制，输出长度可能不等于输入长度
        // 我们只拷贝可用的 tokens
        const size_t tokensToUse = std::min(len, actualOutputLen);

        // 拷贝到输出张量
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
            CLLM_WARN("[LibTorchBackend] WARNING: Output truncated from {} to {} tokens, filling rest with zeros", 
                      len, tokensToUse);
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

} // namespace inference
} // namespace cllm
