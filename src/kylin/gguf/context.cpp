/**
 * @file ggml_context.cpp
 * @brief GGML 上下文封装类实现
 */

#include "cllm/kylin/gguf/context.h"
#include "cllm/common/logger.h"

#include <stdexcept>
#include <cstring>

namespace cllm {
namespace kylin {

GGMLContext::GGMLContext(size_t memSize, BackendType backend)
    : ctx_(nullptr)
    , memSize_(memSize)
    , backendType_(backend)
    , backend_(nullptr)
    , backendBuffer_(nullptr) {
    
    CLLM_INFO("[GGMLContext] Creating context with %zu MB memory", memSize / (1024 * 1024));
    
    // 分配内存缓冲区
    buffer_.resize(memSize);
    
    // 初始化 GGML 上下文
    struct ggml_init_params params = {
        .mem_size   = memSize,
        .mem_buffer = buffer_.data(),
        .no_alloc   = false,  // 允许分配
    };
    
    ctx_ = ggml_init(params);
    if (!ctx_) {
        throw std::runtime_error("Failed to initialize GGML context");
    }
    
    // 初始化后端
    initBackend();
    
    CLLM_INFO("[GGMLContext] Context created successfully");
}

GGMLContext::~GGMLContext() {
    cleanup();
}

GGMLContext::GGMLContext(GGMLContext&& other) noexcept
    : ctx_(other.ctx_)
    , buffer_(std::move(other.buffer_))
    , memSize_(other.memSize_)
    , backendType_(other.backendType_)
    , backend_(other.backend_)
    , backendBuffer_(other.backendBuffer_) {
    
    other.ctx_ = nullptr;
    other.backend_ = nullptr;
    other.backendBuffer_ = nullptr;
}

GGMLContext& GGMLContext::operator=(GGMLContext&& other) noexcept {
    if (this != &other) {
        cleanup();
        
        ctx_ = other.ctx_;
        buffer_ = std::move(other.buffer_);
        memSize_ = other.memSize_;
        backendType_ = other.backendType_;
        backend_ = other.backend_;
        backendBuffer_ = other.backendBuffer_;
        
        other.ctx_ = nullptr;
        other.backend_ = nullptr;
        other.backendBuffer_ = nullptr;
    }
    return *this;
}

void GGMLContext::initBackend() {
    CLLM_DEBUG("[GGMLContext] initBackend called with type: %d", static_cast<int>(backendType_));
    
    switch (backendType_) {
        case BackendType::CPU:
        case BackendType::Auto:
            // CPU 后端
            CLLM_DEBUG("[GGMLContext] Initializing CPU backend...");
            backend_ = ggml_backend_cpu_init();
            if (!backend_) {
                CLLM_WARN("[GGMLContext] Failed to init CPU backend, using default compute");
            } else {
                CLLM_INFO("[GGMLContext] CPU backend initialized");
            }
            break;
            
#ifdef GGML_USE_CUDA
        case BackendType::CUDA:
            CLLM_DEBUG("[GGMLContext] Initializing CUDA backend...");
            backend_ = ggml_backend_cuda_init(0);  // GPU 0
            if (!backend_) {
                CLLM_WARN("[GGMLContext] CUDA backend not available, falling back to CPU");
                backend_ = ggml_backend_cpu_init();
                backendType_ = BackendType::CPU;
            } else {
                CLLM_INFO("[GGMLContext] CUDA backend initialized");
            }
            break;
#else
        case BackendType::CUDA:
            CLLM_WARN("[GGMLContext] CUDA requested but not compiled (GGML_USE_CUDA not defined), using CPU");
            backend_ = ggml_backend_cpu_init();
            backendType_ = BackendType::CPU;
            break;
#endif

#ifdef GGML_USE_METAL
        case BackendType::Metal:
            CLLM_INFO("[GGMLContext] Initializing Metal backend...");
            backend_ = ggml_backend_metal_init();
            if (!backend_) {
                CLLM_WARN("[GGMLContext] Metal backend not available, falling back to CPU");
                backend_ = ggml_backend_cpu_init();
                backendType_ = BackendType::CPU;
            } else {
                CLLM_INFO("[GGMLContext] ✅ Metal backend initialized successfully");
            }
            break;
#else
        case BackendType::Metal:
            CLLM_WARN("[GGMLContext] Metal requested but not compiled (GGML_USE_METAL not defined), using CPU");
            backend_ = ggml_backend_cpu_init();
            backendType_ = BackendType::CPU;
            break;
#endif

        default:
            CLLM_WARN("[GGMLContext] Unknown backend type %d, using CPU", static_cast<int>(backendType_));
            backend_ = ggml_backend_cpu_init();
            backendType_ = BackendType::CPU;
            break;
    }
    
    CLLM_DEBUG("[GGMLContext] Backend initialization complete, final type: %d", static_cast<int>(backendType_));
}

void GGMLContext::cleanup() {
    if (backendBuffer_) {
        ggml_backend_buffer_free(backendBuffer_);
        backendBuffer_ = nullptr;
    }
    
    if (backend_) {
        ggml_backend_free(backend_);
        backend_ = nullptr;
    }
    
    if (ctx_) {
        ggml_free(ctx_);
        ctx_ = nullptr;
    }
}

ggml_tensor* GGMLContext::newTensor1D(ggml_type type, int64_t ne0) {
    if (!ctx_) {
        throw std::runtime_error("GGMLContext not initialized");
    }
    return ggml_new_tensor_1d(ctx_, type, ne0);
}

ggml_tensor* GGMLContext::newTensor2D(ggml_type type, int64_t ne0, int64_t ne1) {
    if (!ctx_) {
        throw std::runtime_error("GGMLContext not initialized");
    }
    return ggml_new_tensor_2d(ctx_, type, ne0, ne1);
}

ggml_tensor* GGMLContext::newTensor3D(ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2) {
    if (!ctx_) {
        throw std::runtime_error("GGMLContext not initialized");
    }
    return ggml_new_tensor_3d(ctx_, type, ne0, ne1, ne2);
}

ggml_tensor* GGMLContext::newTensor4D(ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    if (!ctx_) {
        throw std::runtime_error("GGMLContext not initialized");
    }
    return ggml_new_tensor_4d(ctx_, type, ne0, ne1, ne2, ne3);
}

ggml_cgraph* GGMLContext::buildGraph(ggml_tensor* output) {
    if (!ctx_ || !output) {
        throw std::runtime_error("Invalid context or output tensor");
    }
    
    // 创建计算图
    // 注意：GGML 的新版本使用 ggml_new_graph 需要上下文
    ggml_cgraph* graph = ggml_new_graph(ctx_);
    ggml_build_forward_expand(graph, output);
    
    return graph;
}

void GGMLContext::compute(ggml_cgraph* graph) {
    if (!ctx_ || !graph) {
        throw std::runtime_error("Invalid context or graph");
    }
    
    // 优先使用后端计算（支持 Metal/CUDA 加速）
    if (backend_) {
        ggml_backend_graph_compute(backend_, graph);
    } else {
        // 回退到默认 CPU 计算
        ggml_graph_compute_with_ctx(ctx_, graph, /* n_threads */ 4);
    }
}

void GGMLContext::computeWithBackend(ggml_cgraph* graph, int nThreads) {
    if (!graph) {
        throw std::runtime_error("Invalid graph");
    }
    
    if (backend_) {
        // 使用后端计算
        ggml_backend_graph_compute(backend_, graph);
    } else {
        // 回退到默认计算
        int threads = (nThreads > 0) ? nThreads : 4;
        ggml_graph_compute_with_ctx(ctx_, graph, threads);
    }
}

void GGMLContext::setBackend(BackendType type) {
    if (type == backendType_) {
        return;
    }
    
    // 清理旧后端
    if (backend_) {
        ggml_backend_free(backend_);
        backend_ = nullptr;
    }
    
    backendType_ = type;
    initBackend();
}

bool GGMLContext::isGPUAvailable() {
#ifdef GGML_USE_CUDA
    return true;
#elif defined(GGML_USE_METAL)
    return true;
#else
    return false;
#endif
}

size_t GGMLContext::usedMemory() const {
    if (!ctx_) {
        return 0;
    }
    return ggml_used_mem(ctx_);
}

void GGMLContext::reset() {
    if (ctx_) {
        ggml_free(ctx_);
    }
    
    // 重新初始化
    struct ggml_init_params params = {
        .mem_size   = memSize_,
        .mem_buffer = buffer_.data(),
        .no_alloc   = false,
    };
    
    ctx_ = ggml_init(params);
    if (!ctx_) {
        throw std::runtime_error("Failed to reinitialize GGML context");
    }
}

std::string GGMLContext::typeToString(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:   return "F32";
        case GGML_TYPE_F16:   return "F16";
        case GGML_TYPE_Q4_0:  return "Q4_0";
        case GGML_TYPE_Q4_1:  return "Q4_1";
        case GGML_TYPE_Q5_0:  return "Q5_0";
        case GGML_TYPE_Q5_1:  return "Q5_1";
        case GGML_TYPE_Q8_0:  return "Q8_0";
        case GGML_TYPE_Q8_1:  return "Q8_1";
        case GGML_TYPE_Q4_K:  return "Q4_K";
        case GGML_TYPE_Q5_K:  return "Q5_K";
        case GGML_TYPE_Q6_K:  return "Q6_K";
        case GGML_TYPE_I8:    return "I8";
        case GGML_TYPE_I16:   return "I16";
        case GGML_TYPE_I32:   return "I32";
        case GGML_TYPE_I64:   return "I64";
        case GGML_TYPE_F64:   return "F64";
        default:              return "Unknown";
    }
}

} // namespace kylin
} // namespace cllm
