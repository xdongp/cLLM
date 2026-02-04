/**
 * @file generate_endpoint.h
 * @brief 文本生成端点，处理/generate API请求
 * @author cLLM Team
 * @date 2024-01-09
 */

#ifndef CLLM_GENERATE_ENDPOINT_H
#define CLLM_GENERATE_ENDPOINT_H

#include "cllm/http/api_endpoint.h"
#include "cllm/http/request.h"
#include "cllm/http/response.h"
#include "cllm/http/handler.h"

namespace cllm {

class Scheduler;
class ITokenizer;

/**
 * @brief 文本生成端点类
 * 
 * 处理/generate API请求，支持流式和非流式生成。
 * 使用Scheduler调度生成任务，使用Tokenizer进行编解码。
 */
class GenerateEndpoint : public ApiEndpoint {
public:
    /**
     * @brief 构造函数
     * @param scheduler 调度器指针
     * @param tokenizer 分词器指针
     */
    GenerateEndpoint(Scheduler* scheduler, ITokenizer* tokenizer);
    
    /**
     * @brief 析构函数
     */
    ~GenerateEndpoint();
    
    /**
     * @brief 处理HTTP请求
     * @param request HTTP请求对象
     * @return HTTP响应对象
     */
    HttpResponse handle(const HttpRequest& request) override;
    
    /**
     * @brief 设置调度器
     * @param scheduler 调度器指针
     */
    void setScheduler(Scheduler* scheduler);
    
    /**
     * @brief 设置分词器
     * @param tokenizer 分词器指针
     */
    void setTokenizer(ITokenizer* tokenizer);
    
private:
    /**
     * @brief 生成请求结构
     */
    struct GenerateRequest {
        std::string prompt;          ///< 输入提示词
        int maxTokens;               ///< 最大生成token数
        float temperature;           ///< 温度参数
        float topP;                  ///< Top-P采样参数
        float repetitionPenalty;     ///< 重复惩罚参数
        bool stream;                 ///< 是否使用流式输出
    };
    
    GenerateRequest parseRequest(const HttpRequest& request);  ///< 解析生成请求
    std::string generateRequestId();  ///< 生成请求ID
    HttpResponse handleNonStreaming(const GenerateRequest& req);  ///< 处理非流式请求
    HttpResponse handleStreaming(const GenerateRequest& req);  ///< 处理流式请求（伪流式，兼容旧接口）
    
public:
    /**
     * @brief 真流式处理（边生成边发送）
     * 
     * 用于 HttpHandler::postStreaming 注册，每生成一个 token 立即通过回调发送。
     * 
     * @param request HTTP请求
     * @param writeCallback 流式写入回调
     */
    void handleStreamingCallback(const HttpRequest& request, StreamingWriteCallback writeCallback);
    
private:
    Scheduler* scheduler_;      ///< 调度器指针
    ITokenizer* tokenizer_;      ///< 分词器指针
};

}

#endif
