/**
 * @file cpu_backend.cpp
 * @brief CPU 后端实现入口
 *
 * 实现委托给 ggml_cpu_impl.cpp 中的完整 CPU 计算逻辑。
 */

#include "cllm/kylin/backend/cpu/cpu_backend.h"
#include "cllm/kylin/core/ggml_kernels.h"
#include "cllm/common/logger.h"

#include <cstring>
#include <cmath>
#include <algorithm>

namespace cllm {
namespace kylin {
namespace backend {

// 注意：完整的 CPU 实现位于 ggml_cpu_impl.cpp
// 这个文件作为 backend/cpu 模块的入口点

} // namespace backend
} // namespace kylin
} // namespace cllm
