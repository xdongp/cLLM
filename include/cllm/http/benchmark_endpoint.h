/**
 * @file benchmark_endpoint.h
 * @brief Benchmark端点，用于服务器端性能测试
 * @author cLLM Team
 * @date 2026-01-20
 */

#ifndef CLLM_BENCHMARK_ENDPOINT_H
#define CLLM_BENCHMARK_ENDPOINT_H

#include <memory>
#include "cllm/http/api_endpoint.h"
#include "cllm/http/request.h"
#include "cllm/http/response.h"

namespace cllm {

class GenerateEndpoint;
class Scheduler;
class ITokenizer;
class ModelExecutor;

/**
 * @brief Benchmark端点类
 * 
 * 处理/benchmark API请求，在服务器端内部并发处理多个请求，
 * 消除网络传输和Python端开销，直接测试C++内部性能。
 * 
 * 🔥 优化版本：直接调用Scheduler和Tokenizer，绕过GenerateEndpoint和HTTP层开销。
 * 
 * 请求格式:
 * {
 *   "requests": 40,           // 总请求数
 *   "concurrency": 8,         // 并发数
 *   "max_tokens": 50,         // 每个请求的最大token数
 *   "prompt": "Hello, world", // 提示词（可选，默认使用）
 *   "temperature": 0.7       // 温度参数（可选）
 * }
 * 
 * 响应格式:
 * {
 *   "success": true,
 *   "data": {
 *     "total_requests": 40,
 *     "successful_requests": 38,
 *     "failed_requests": 2,
 *     "avg_response_time": 7.50,
 *     "min_response_time": 1.09,
 *     "max_response_time": 8.60,
 *     "avg_throughput": 49.13,
 *     "avg_tokens_per_second": 7.65,
 *     "total_tokens_processed": 2408,
 *     "avg_generated_tokens": 50.00,
 *     "total_time": 38.68
 *   }
 * }
 */
class BenchmarkEndpoint : public ApiEndpoint {
public:
    /**
     * @brief 构造函数（优化版本：直接使用Scheduler和Tokenizer）
     * @param scheduler Scheduler指针，用于直接调度请求
     * @param tokenizer Tokenizer指针，用于编码/解码
     */
    BenchmarkEndpoint(Scheduler* scheduler, ITokenizer* tokenizer);
    
    /**
     * @brief 构造函数（最优版本：使用独立的Scheduler实例，参考Stage 15）
     * @param modelExecutor ModelExecutor指针，用于创建独立的Scheduler
     * @param tokenizer Tokenizer指针，用于编码/解码
     * @param maxBatchSize 最大批处理大小（默认8，与Stage 15一致）
     * @param maxContextLength 最大上下文长度（默认2048，与Stage 15一致）
     */
    BenchmarkEndpoint(ModelExecutor* modelExecutor, ITokenizer* tokenizer, 
                      size_t maxBatchSize = 8, size_t maxContextLength = 2048);
    
    /**
     * @brief 构造函数（兼容版本：使用GenerateEndpoint）
     * @param generateEndpoint GenerateEndpoint指针，用于处理实际请求
     */
    explicit BenchmarkEndpoint(GenerateEndpoint* generateEndpoint);
    
    /**
     * @brief 析构函数
     */
    ~BenchmarkEndpoint();
    
    /**
     * @brief 处理HTTP请求
     * @param request HTTP请求对象
     * @return HTTP响应对象
     */
    HttpResponse handle(const HttpRequest& request) override;
    
    /**
     * @brief 设置GenerateEndpoint（兼容模式）
     * @param generateEndpoint GenerateEndpoint指针
     */
    void setGenerateEndpoint(GenerateEndpoint* generateEndpoint);
    
    /**
     * @brief 设置Scheduler和Tokenizer（优化模式）
     * @param scheduler Scheduler指针
     * @param tokenizer Tokenizer指针
     */
    void setSchedulerAndTokenizer(Scheduler* scheduler, ITokenizer* tokenizer);

private:
    /**
     * @brief Benchmark请求参数结构
     */
    struct BenchmarkRequest {
        int requests = 40;          ///< 总请求数
        int concurrency = 8;       ///< 并发数
        int maxTokens = 50;        ///< 每个请求的最大token数
        std::string prompt = "Hello, world! How are you today?";  ///< 提示词
        float temperature = 0.7f;  ///< 温度参数
    };
    
    /**
     * @brief 单个请求的结果
     */
    struct RequestResult {
        bool success = false;              ///< 是否成功
        double responseTime = 0.0;         ///< 响应时间（秒）
        size_t generatedTokens = 0;         ///< 生成的token数
        size_t totalTokens = 0;            ///< 总token数（prompt + generated）
        double tokensPerSecond = 0.0;      ///< tokens per second
        std::string errorMessage;          ///< 错误信息（如果有）
    };
    
    /**
     * @brief 统计结果
     */
    struct Statistics {
        int totalRequests = 0;
        int successfulRequests = 0;
        int failedRequests = 0;
        double avgResponseTime = 0.0;
        double minResponseTime = 0.0;
        double maxResponseTime = 0.0;
        double avgThroughput = 0.0;         ///< 平均吞吐量（tokens/sec）
        double avgTokensPerSecond = 0.0;   ///< 平均tokens per second
        size_t totalTokensProcessed = 0;
        double avgGeneratedTokens = 0.0;
        double totalTime = 0.0;            ///< 总测试时间
    };
    
    BenchmarkRequest parseRequest(const HttpRequest& request);  ///< 解析benchmark请求
    RequestResult executeSingleRequest(const BenchmarkRequest& params, int requestIndex);  ///< 执行单个请求（兼容模式：通过GenerateEndpoint）
    RequestResult executeSingleRequestDirect(const BenchmarkRequest& params, int requestIndex);  ///< 执行单个请求（优化模式：直接调用Scheduler）
    Statistics calculateStatistics(const std::vector<RequestResult>& results, double totalTime);  ///< 计算统计数据
    HttpResponse buildResponse(const Statistics& stats);  ///< 构建响应
    
    bool useDirectMode_;  ///< 是否使用直接模式（直接调用Scheduler）
    bool useIndependentScheduler_;  ///< 是否使用独立的Scheduler实例（最优模式）
    GenerateEndpoint* generateEndpoint_;  ///< GenerateEndpoint指针（兼容模式）
    Scheduler* scheduler_;  ///< Scheduler指针（优化模式，共享Scheduler）
    std::unique_ptr<Scheduler> independentScheduler_;  ///< 独立的Scheduler实例（最优模式）
    ITokenizer* tokenizer_;  ///< Tokenizer指针（优化模式）
    size_t maxBatchSize_;  ///< 最大批处理大小（用于创建独立Scheduler）
    size_t maxContextLength_;  ///< 最大上下文长度（用于创建独立Scheduler）
};

} // namespace cllm

#endif // CLLM_BENCHMARK_ENDPOINT_H
