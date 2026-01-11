#pragma once

#include "tokenizer.h"
#include <vector>
#include <string>
#include <memory>
#include <future>
#include <thread>

namespace cllm {

/**
 * @brief 批处理分词器工具类
 * 
 * 提供批量编码和解码功能，支持多线程并行处理以提升吞吐量。
 * 设计目标：
 * - 吞吐量提升：3-5x 相比单线程处理
 * - 线程安全：支持多线程并发调用
 * - 错误处理：单个失败不影响其他请求
 */
class BatchTokenizer {
public:
    /**
     * @brief 批处理编码结果
     */
    struct BatchEncodeResult {
        std::vector<std::vector<llama_token>> tokenized;  ///< 编码结果
        std::vector<bool> success;                         ///< 每个请求是否成功
        std::vector<std::string> errors;                   ///< 错误信息（如果失败）
    };

    /**
     * @brief 批处理解码结果
     */
    struct BatchDecodeResult {
        std::vector<std::string> decoded;    ///< 解码结果
        std::vector<bool> success;           ///< 每个请求是否成功
        std::vector<std::string> errors;     ///< 错误信息（如果失败）
    };

    /**
     * @brief 批量编码文本
     * 
     * @param tokenizer 分词器实例
     * @param texts 待编码的文本列表
     * @param addSpecialTokens 是否添加特殊token
     * @param maxParallel 最大并行线程数（0表示自动检测，默认为 CPU 核心数）
     * @return BatchEncodeResult 编码结果
     */
    static BatchEncodeResult batchEncode(
        CTokenizer* tokenizer,
        const std::vector<std::string>& texts,
        bool addSpecialTokens = true,
        int maxParallel = 0
    );

    /**
     * @brief 批量解码token序列
     * 
     * @param tokenizer 分词器实例
     * @param tokenSequences token序列列表
     * @param skipSpecialTokens 是否跳过特殊token
     * @param maxParallel 最大并行线程数（0表示自动检测）
     * @return BatchDecodeResult 解码结果
     */
    static BatchDecodeResult batchDecode(
        CTokenizer* tokenizer,
        const std::vector<std::vector<llama_token>>& tokenSequences,
        bool skipSpecialTokens = true,
        int maxParallel = 0
    );

    /**
     * @brief 批量编码（使用指定配置）
     * @param tokenizer 分词器实例
     * @param texts 待编码的文本列表
     * @param config 性能配置
     * @param addSpecialTokens 是否添加特殊 token
     * @return 编码结果（保持与输入顺序一致）
     */
    static BatchEncodeResult batchEncode(
        CTokenizer* tokenizer,
        const std::vector<std::string>& texts,
        const TokenizerPerformanceConfig& config,
        bool addSpecialTokens = true
    );
    
    /**
     * @brief 批量解码（使用指定配置）
     * @param tokenizer 分词器实例
     * @param tokensList 待解码的 token 列表
     * @param config 性能配置
     * @param skipSpecialTokens 是否跳过特殊 token
     * @return 解码结果（保持与输入顺序一致）
     */
    static BatchDecodeResult batchDecode(
        CTokenizer* tokenizer,
        const std::vector<std::vector<llama_token>>& tokensList,
        const TokenizerPerformanceConfig& config,
        bool skipSpecialTokens = true
    );

private:
    /**
     * @brief 获取最优线程数
     */
    static int getOptimalThreadCount(int maxParallel, size_t taskCount);
};

} // namespace cllm
