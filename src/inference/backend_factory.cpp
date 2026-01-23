/**
 * @file backend_factory.cpp
 * @brief 后端工厂实现
 */

#include "cllm/inference/backend_interface.h"
#include "cllm/inference/kylin_backend.h"
#include "cllm/inference/libtorch_backend.h"
#include "cllm/kylin/gguf/operator_interface.h"
#include "cllm/common/config.h"
#ifdef CLLM_USE_LLAMA_CPP
#include "cllm/inference/llama_cpp_backend.h"
#endif
#include "cllm/common/logger.h"
#include <stdexcept>

namespace cllm {
namespace inference {

namespace {

// 将配置字符串转换为 OperatorBackend 枚举
kylin::OperatorBackend parseOperatorBackend(const std::string& str) {
    if (str == "native" || str == "Native") {
        return kylin::OperatorBackend::Native;
    } else if (str == "ggml" || str == "GGML") {
        return kylin::OperatorBackend::GGML;
    } else {
        // 默认为 Auto
        return kylin::OperatorBackend::Auto;
    }
}

} // anonymous namespace

std::unique_ptr<IBackend> BackendFactory::createBackend(
    const std::string &backendType,
    const ModelConfig &config,
    const std::string &modelPath
) {
    CLLM_INFO("[BackendFactory] Creating backend: %s", backendType.c_str());
    
    if (backendType == "libtorch" || backendType == "LibTorch") {
        return std::make_unique<LibTorchBackend>(modelPath, config);
    } else if (backendType == "kylin" || backendType == "Kylin") {
        // 从配置读取算子后端类型
        std::string opBackendStr = Config::instance().backendKylinOperatorBackend();
        kylin::OperatorBackend opBackend = parseOperatorBackend(opBackendStr);
        
        CLLM_INFO("[BackendFactory] Kylin backend using operator: %s", opBackendStr.c_str());
        return std::make_unique<KylinBackend>(config, modelPath, opBackend);
    } else if (backendType == "llama_cpp" || backendType == "llama.cpp" || backendType == "LlamaCpp") {
#ifdef CLLM_USE_LLAMA_CPP
        return std::make_unique<LlamaCppBackend>(config, modelPath);
#else
        throw std::runtime_error(
            "BackendFactory::createBackend: llama.cpp backend not available in this build"
        );
#endif
    } else {
        throw std::runtime_error(
            "BackendFactory::createBackend: unsupported backend type: " + backendType
        );
    }
}

} // namespace inference
} // namespace cllm
