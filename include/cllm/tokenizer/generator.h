/**
 * @file generator.h
 * @brief 流式生成器
 * @author cLLM Team
 * @date 2024-01-09
 */

#ifndef CLLM_STREAM_GENERATOR_H
#define CLLM_STREAM_GENERATOR_H

#include <string>
#include <vector>
#include "cllm/tokenizer/response.h"
#include "i_tokenizer.h"

namespace cllm {

class ModelExecutor;
class Tokenizer;

/**
 * @brief 流式生成器类
 * 
 * 实现流式文本生成，逐token返回结果。
 */
class StreamGenerator {
public:
    /**
     * @brief 构造函数
     * @param requestId 请求ID
     * @param inputIds 输入token IDs
     * @param maxTokens 最大token数
     * @param temperature 温度参数
     * @param modelExecutor 模型执行器指针
     * @param tokenizer Tokenizer指针
     */
    explicit StreamGenerator(
        const std::string& requestId,
        const std::vector<int>& inputIds,
        int maxTokens,
        float temperature,
        ModelExecutor* modelExecutor,
        ITokenizer* tokenizer
    );
    
    /**
     * @brief 析构函数
     */
    ~StreamGenerator();
    
    /**
     * @brief 检查是否有下一个token
     * @return true 如果还有token，false 否则
     */
    bool hasNext();
    
    /**
     * @brief 获取下一个生成响应
     * @return 生成响应对象
     */
    GenerationResponse next();
    
    /**
     * @brief 检查生成是否完成
     * @return true 如果完成，false 否则
     */
    bool isFinished() const;
    
    /**
     * @brief 获取已生成的token数
     * @return 生成的token数
     */
    int getGeneratedTokenCount() const;
    
private:
    void generateNextToken();  ///< 生成下一个token
    std::string extractNewText();  ///< 提取新文本
    
    std::string requestId_;          ///< 请求ID
    std::vector<int> inputIds_;      ///< 输入token IDs
    std::vector<int> generatedTokens_;  ///< 已生成的tokens
    std::string previousText_;       ///< 之前的文本
    
    int maxTokens_;                  ///< 最大token数
    float temperature_;              ///< 温度参数
    
    ModelExecutor* modelExecutor_;   ///< 模型执行器指针
    ITokenizer* tokenizer_;           ///< Tokenizer指针
    
    bool finished_;                  ///< 是否完成
    int currentTokenIndex_;          ///< 当前token索引
};

}

#endif
