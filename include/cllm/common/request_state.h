/**
 * @file request_state.h
 * @brief 请求状态结构，记录生成请求的完整状态信息
 * @author cLLM Team
 * @date 2024-01-01
 */

#pragma once

#include <vector>
#include <string>
#include <cstdint>

namespace cllm {

/**
 * @brief 请求状态结构
 * 
 * 记录一个生成请求的完整状态信息，包括输入、输出、参数、时间等。
 */
struct RequestState {
    size_t requestId = 0;                       ///< 请求ID
    std::vector<int> tokenizedPrompt;           ///< token化后的输入提示
    std::vector<int> generatedTokens;           ///< 已生成的token序列
    int maxTokens = 100;                        ///< 最大生成token数
    float temperature = 0.7f;                   ///< 温度参数
    int topK = 50;                              ///< Top-K采样参数
    float topP = 0.9f;                          ///< Top-P采样参数

    // 停止条件（由上层根据 tokenizer/model 注入；-1 表示未知/不启用）
    int eosTokenId = -1;                        ///< EOS token id（生成遇到该 id 应停止）

    std::string samplingStrategy = "temperature"; ///< 采样策略
    size_t arrivalTime = 0;                     ///< 请求到达时间
    size_t startTime = 0;                       ///< 开始处理时间
    size_t completionTime = 0;                  ///< 完成时间
    float priority = 0.0f;                      ///< 优先级
    bool isCompleted = false;                   ///< 是否已完成
    bool isRunning = false;                     ///< 是否正在运行
    bool isFailed = false;                      ///< 是否失败
    bool isTimeout = false;                     ///< 是否超时
    std::string errorMessage;                   ///< 错误信息
    
    /**
     * @brief 默认构造函数
     */
    RequestState() = default;
    
    /**
     * @brief 带请求ID的构造函数
     * @param id 请求ID
     */
    explicit RequestState(size_t id) : requestId(id) {}
    
    /**
     * @brief 获取提示长度
     * @return 提示的token数量
     */
    size_t getPromptLength() const {
        return tokenizedPrompt.size();
    }
    
    /**
     * @brief 获取总长度（提示+生成）
     * @return 总token数量
     */
    size_t getTotalLength() const {
        return tokenizedPrompt.size() + generatedTokens.size();
    }
    
    /**
     * @brief 计算请求优先级
     * @param currentTime 当前时间
     * @return 计算后的优先级
     */
    float calculatePriority(size_t currentTime) const;
    
    /**
     * @brief 判断请求状态：PENDING
     * PENDING: 未完成、未失败、未运行（等待处理）
     * @return true 如果是PENDING状态
     */
    bool isPending() const {
        return !isCompleted && !isFailed && !isTimeout && !isRunning;
    }
    
    /**
     * @brief 判断请求状态：PROCESSING
     * PROCESSING: 正在运行、未完成、未失败（正在处理中）
     * @return true 如果是PROCESSING状态
     */
    bool isProcessing() const {
        return isRunning && !isCompleted && !isFailed && !isTimeout;
    }
    
    /**
     * @brief 判断请求状态：COMPLETED
     * COMPLETED: 已完成（成功完成）
     * @return true 如果是COMPLETED状态
     */
    bool isCompletedState() const {
        return isCompleted;
    }
    
    /**
     * @brief 判断请求状态：FAILED
     * FAILED: 失败（处理失败）
     * @return true 如果是FAILED状态
     */
    bool isFailedState() const {
        return isFailed;
    }
    
    /**
     * @brief 判断请求状态：TIMEOUT（基于时间戳判断）
     * TIMEOUT: 处理时间超过超时阈值
     * @param currentTime 当前时间
     * @param timeoutSeconds 超时时间（秒）
     * @return true 如果是TIMEOUT状态
     */
    bool checkTimeout(size_t currentTimeMs, float timeoutSeconds) const {
        if (startTime == 0) return false;  // 未开始处理
        if (isCompleted || isFailed || isTimeout) return false;  // 已完成/失败/超时
        if (currentTimeMs < startTime) return false;
        float elapsedSeconds = static_cast<float>(currentTimeMs - startTime) / 1000.0f;
        return elapsedSeconds > timeoutSeconds;
    }

    /**
     * @brief 判断请求状态：TIMEOUT（标记状态）
     * @return true 如果是TIMEOUT状态
     */
    bool isTimeoutState() const {
        return isTimeout;
    }
    
    /**
     * @brief 判断请求是否为活跃状态（PENDING或PROCESSING）
     * 活跃状态：可以继续处理或等待处理
     * @return true 如果是活跃状态
     */
    bool isActive() const {
        return !isCompleted && !isFailed && !isTimeout;
    }
};

}  // namespace cllm