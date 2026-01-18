/**
 * @file backend_factory.cpp
 * @brief 后端工厂实现
 */

#include "cllm/inference/backend_interface.h"
#include "cllm/inference/kylin_backend.h"
#include "cllm/inference/libtorch_backend.h"
#ifdef CLLM_USE_LLAMA_CPP
#include "cllm/inference/llama_cpp_backend.h"
#endif
#include "cllm/common/logger.h"
#include <stdexcept>

namespace cllm {
namespace inference {

std::unique_ptr<IBackend> BackendFactory::createBackend(
    const std::string &backendType,
    const ModelConfig &config,
    const std::string &modelPath
) {
    CLLM_INFO("[BackendFactory] Creating backend: %s", backendType.c_str());
    
    if (backendType == "libtorch" || backendType == "LibTorch") {
        return std::make_unique<LibTorchBackend>(modelPath, config);
    } else if (backendType == "kylin" || backendType == "Kylin") {
        return std::make_unique<KylinBackend>(config, modelPath);
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
