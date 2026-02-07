/**
 * @file backend_factory.cpp
 * @brief 后端工厂实现
 */

#include "cllm/kylin/backend/backend_interface.h"
#include "cllm/kylin/backend/cpu_backend.h"
#include "cllm/kylin/backend/gpu_backend.h"
#include "cllm/kylin/core/ggml_kernels.h"

#include <cstdlib>
#include <string>

namespace cllm {
namespace kylin {

std::unique_ptr<IComputeBackend> BackendFactory::create(DeviceType device) {
    switch (device) {
        case DeviceType::Metal:
#ifdef GGML_USE_METAL
            return std::make_unique<GPUBackend>();
#else
            // Metal 不支持时回退到 CPU
            return std::make_unique<CPUBackend>();
#endif
        case DeviceType::CUDA:
#ifdef GGML_USE_CUDA
            return std::make_unique<GPUBackend>();
#else
            // CUDA 不支持时回退到 CPU
            return std::make_unique<CPUBackend>();
#endif
        case DeviceType::CPU:
        default:
            return std::make_unique<CPUBackend>();
    }
}

DeviceType BackendFactory::getDefaultDevice() {
    // 检查环境变量
    const char* envDevice = std::getenv("CLLM_DEVICE");
    if (envDevice != nullptr) {
        std::string deviceStr(envDevice);
        if (deviceStr == "metal" || deviceStr == "Metal") {
#ifdef GGML_USE_METAL
            return DeviceType::Metal;
#else
            return DeviceType::CPU;
#endif
        }
        if (deviceStr == "cuda" || deviceStr == "CUDA") {
#ifdef GGML_USE_CUDA
            return DeviceType::CUDA;
#else
            return DeviceType::CPU;
#endif
        }
        if (deviceStr == "cpu" || deviceStr == "CPU") {
            return DeviceType::CPU;
        }
    }
    
    // 默认使用 CPU
    return DeviceType::CPU;
}

} // namespace kylin
} // namespace cllm
