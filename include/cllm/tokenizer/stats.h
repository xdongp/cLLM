/**
 * @file stats.h
 * @brief Tokenizer统计信息类
 * @author cLLM Team
 * @date 2024-01-09
 */

#ifndef CLLM_TOKENIZER_STATS_H
#define CLLM_TOKENIZER_STATS_H

#include <atomic>
#include <mutex>

namespace cllm {

/**
 * @brief Tokenizer统计信息类
 * 
 * 记录tokenizer的运行统计信息，线程安全。
 */
class TokenizerStats {
public:
    /**
     * @brief 默认构造函数
     */
    TokenizerStats();
    
    /**
     * @brief 析构函数
     */
    ~TokenizerStats();
    
    /**
     * @brief 拷贝构造函数
     */
    TokenizerStats(const TokenizerStats& other);
    
    /**
     * @brief 拷贝赋值运算符
     */
    TokenizerStats& operator=(const TokenizerStats& other);
    
    void incrementEncodeCount();  ///< 递增编码次数
    void incrementDecodeCount();  ///< 递增解码次数
    void incrementGenerateCount();  ///< 递增生成次数
    void incrementStreamGenerateCount();  ///< 递增流式生成次数
    
    void addEncodeTime(float time);  ///< 添加编码时间
    void addDecodeTime(float time);  ///< 添加解码时间
    void addGenerateTime(float time);  ///< 添加生成时间
    void addStreamGenerateTime(float time);  ///< 添加流式生成时间
    
    void addGeneratedTokens(int count);  ///< 添加生成的tokens数
    
    long long getEncodeCount() const;  ///< 获取编码次数
    long long getDecodeCount() const;  ///< 获取解码次数
    long long getGenerateCount() const;  ///< 获取生成次数
    long long getStreamGenerateCount() const;  ///< 获取流式生成次数
    
    float getAverageEncodeTime() const;  ///< 获取平均编码时间
    float getAverageDecodeTime() const;  ///< 获取平均解码时间
    float getAverageGenerateTime() const;  ///< 获取平均生成时间
    float getAverageStreamGenerateTime() const;  ///< 获取平均流式生成时间
    
    long long getTotalGeneratedTokens() const;  ///< 获取总生成tokens数
    float getAverageTokensPerSecond() const;  ///< 获取平均每秒tokens数
    
    /**
     * @brief 重置所有统计信息
     */
    void reset();
    
private:
    long long encodeCount_{0};          ///< 编码次数
    long long decodeCount_{0};          ///< 解码次数
    long long generateCount_{0};        ///< 生成次数
    long long streamGenerateCount_{0};  ///< 流式生成次数
    
    float totalEncodeTime_{0.0f};          ///< 总编码时间
    float totalDecodeTime_{0.0f};          ///< 总解码时间
    float totalGenerateTime_{0.0f};        ///< 总生成时间
    float totalStreamGenerateTime_{0.0f};  ///< 总流式生成时间
    
    long long totalGeneratedTokens_{0}; ///< 总生成tokens数
    
    mutable std::mutex mutex_; ///< 保护统计信息的互斥锁
};

}

#endif
