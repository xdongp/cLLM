/**
 * @file manager.h
 * @brief Tokenizer管理器，统一管理编解码和生成
 * @author cLLM Team
 * @date 2024-01-09
 */

#ifndef CLLM_TOKENIZER_MANAGER_H
#define CLLM_TOKENIZER_MANAGER_H

#include <string>
#include <vector>
#include <mutex>
#include "cllm/tokenizer/tokenizer.h"
#include "cllm/tokenizer/config.h"
#include "cllm/tokenizer/request.h"
#include "cllm/tokenizer/response.h"
#include "cllm/tokenizer/stats.h"
#include "i_tokenizer.h"

namespace cllm {

class ModelExecutor;
class KVCache;

/**
 * @brief Tokenizer管理器类
 * 
 * 统一管理文本编解码和文本生成功能，协调tokenizer、模型执行器和KV缓存。
 */
class TokenizerManager {
public:
    /**
     * @brief 构造函数
     * @param modelPath 模型路径
     * @param modelExecutor 模型执行器指针
     */
    enum class TokenizerImpl {
        AUTO,       // 自动选择
        HF,         // tokenizers-cpp
        NATIVE      // 自研CTokenizer
    };

    explicit TokenizerManager(
        const std::string& modelPath,
        ModelExecutor* modelExecutor = nullptr,
        TokenizerImpl impl = TokenizerImpl::AUTO
    );
    
    /**
     * @brief 析构函数
     */
    ~TokenizerManager();
    
    /**
     * @brief 编码文本
     * @param text 输入文本
     * @return Token IDs列表
     */
    std::vector<int> encode(const std::string& text);
    
    /**
     * @brief 解码tokens
     * @param tokenIds Token IDs列表
     * @return 解码后的文本
     */
    std::string decode(const std::vector<int>& tokenIds);
    
    /**
     * @brief 生成文本
     * @param requestId 请求ID
     * @param prompt 输入提示词
     * @param maxTokens 最大token数
     * @param temperature 温度参数
     * @param topP Top-P参数
     * @return 生成的文本
     */
    std::string generate(
        const std::string& requestId,
        const std::string& prompt,
        int maxTokens = 100,
        float temperature = 0.7f,
        float topP = 0.9f
    );
    
    /**
     * @brief 流式生成文本
     * @param requestId 请求ID
     * @param prompt 输入提示词
     * @param maxTokens 最大token数
     * @param temperature 温度参数
     * @param topP Top-P参数
     * @return 生成响应列表
     */
    std::vector<GenerationResponse> generateStream(
        const std::string& requestId,
        const std::string& prompt,
        int maxTokens = 100,
        float temperature = 0.7f,
        float topP = 0.9f
    );
    
    void setModelExecutor(ModelExecutor* modelExecutor);  ///< 设置模型执行器
    void setKVCache(KVCache* kvCache);  ///< 设置KV缓存
    
    ITokenizer* getTokenizer() const;  ///< 获取tokenizer
    ModelExecutor* getModelExecutor() const;  ///< 获取模型执行器
    KVCache* getKVCache() const;  ///< 获取KV缓存
    
    /**
     * @brief 获取统计信息
     * @return Tokenizer统计信息
     */
    TokenizerStats getStats() const;
    
    /**
     * @brief 重置统计信息
     */
    void resetStats();
    
private:
    std::vector<int> encodePrompt(const std::string& prompt);  ///< 编码提示词
    std::string decodeTokens(const std::vector<int>& tokens);  ///< 解码tokens
    
    bool isStopToken(int tokenId);  ///< 检查是否为停止token
    void updateStats(const std::string& requestId, int tokenCount, float time);  ///< 更新统计信息
    
    ITokenizer* tokenizer_;           ///< Tokenizer指针
    ModelExecutor* modelExecutor_;   ///< 模型执行器指针
    KVCache* kvCache_;               ///< KV缓存指针
    
    std::vector<int> stopTokens_;    ///< 停止tokens
    
    TokenizerStats stats_;           ///< 统计信息
    mutable std::mutex statsMutex_;   ///< 保护统计信息的互斥锁
    
    void loadStopTokens(const std::string& configPath);  ///< 从配置文件加载停止tokens
};

// 检测模型类型辅助函数
ModelType detectModelType(const std::string& modelPath);

} // namespace cllm

#endif
