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
    std::string samplingStrategy = "temperature"; ///< 采样策略
    size_t arrivalTime = 0;                     ///< 请求到达时间
    size_t startTime = 0;                       ///< 开始处理时间
    size_t completionTime = 0;                  ///< 完成时间
    size_t priority = 0;                        ///< 优先级
    bool isCompleted = false;                   ///< 是否已完成
    bool isRunning = false;                     ///< 是否正在运行
    bool isFailed = false;                      ///< 是否失败
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
};

}  // namespace cllm