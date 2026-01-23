/**
 * @file backend_interface.h
 * @brief 推理后端统一接口定义
 * 
 * 参考文档：推理引擎接口设计.md
 */
#pragma once

#include "cllm/kylin/core/tensor.h"
#include "cllm/model/config.h"

#include <string>
#include <vector>
#include <memory>

namespace cllm {
namespace inference {

// 使用 kylin::Tensor 作为统一的张量类型
using Tensor = kylin::Tensor;

/**
 * @brief 推理后端接口基类
 * 
 * 所有后端（LibTorch、Kylin 等）必须实现此接口
 * 
 * 设计原则：
 * - 统一的初始化和推理接口
 * - 支持单序列和批处理推理
 * - 后端独立演进，互不影响
 */
class IBackend {
public:
    virtual ~IBackend() = default;

    /**
     * @brief 初始化后端
     * 
     * 负责：
     * - 加载模型权重
     * - 初始化内部数据结构
     * - 执行预热推理（可选）
     * 
     * @return true 成功，false 失败
     */
    virtual bool initialize() = 0;

    /**
     * @brief 单序列前向推理
     * 
     * @param inputIds 输入 token id 序列
     * @return [seq_len, vocab_size] logits 张量
     * 
     * @throws std::runtime_error 如果推理失败
     */
    virtual Tensor forward(const std::vector<int> &inputIds) = 0;

    /**
     * @brief 批处理前向推理
     * 
     * 与 BatchInput 语义对齐，支持多请求并行处理
     * 
     * @param flatInputIds 展平后的所有 token id
     * @param requestPositions 每个请求在 flatInputIds 中的起止位置 [start, end)
     * @param batchSize 批大小（请求数）
     * @param sequenceIds 每个请求的序列ID（requestId），用于序列ID管理（可选，默认空向量）
     * @return [total_tokens, vocab_size] logits 张量
     * 
     * @throws std::runtime_error 如果推理失败
     * @throws std::invalid_argument 如果参数无效
     */
    virtual Tensor forwardBatch(
        const std::vector<int> &flatInputIds,
        const std::vector<std::pair<size_t, size_t>> &requestPositions,
        size_t batchSize,
        const std::vector<size_t> &sequenceIds = {}
    ) = 0;

    /**
     * @brief 获取后端名称
     * 
     * @return 后端名称字符串（如 "LibTorch", "Kylin"）
     */
    virtual std::string getName() const = 0;

    /**
     * @brief 检查是否已初始化
     * 
     * @return true 已初始化，false 未初始化
     */
    virtual bool isInitialized() const = 0;

    /**
     * @brief 获取后端配置
     * 
     * @return 模型配置引用
     */
    virtual const ModelConfig &getConfig() const = 0;
};

/**
 * @brief 后端工厂类
 * 
 * 负责创建不同类型的后端实例
 */
class BackendFactory {
public:
    /**
     * @brief 创建后端实例
     * 
     * @param backendType 后端类型（"libtorch", "kylin"）
     * @param config 模型配置
     * @param modelPath 模型路径
     * @return 后端实例的智能指针
     * 
     * @throws std::runtime_error 如果后端类型不支持
     */
    static std::unique_ptr<IBackend> createBackend(
        const std::string &backendType,
        const ModelConfig &config,
        const std::string &modelPath
    );
};

} // namespace inference
} // namespace cllm
