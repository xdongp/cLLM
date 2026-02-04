#include "cllm/http/benchmark_endpoint.h"
#include "cllm/http/generate_endpoint.h"
#include "cllm/http/json_request_parser.h"
#include "cllm/http/response_builder.h"
#include "cllm/scheduler/scheduler.h"
#include "cllm/tokenizer/i_tokenizer.h"
#include "cllm/model/executor.h"
#include "cllm/common/request_state.h"
#include "cllm/common/logger.h"
#include "cllm/common/config.h"
#include <nlohmann/json.hpp>
#include <thread>
#include <mutex>
#include <vector>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <limits>
#include <memory>

namespace cllm {

BenchmarkEndpoint::BenchmarkEndpoint(Scheduler* scheduler, ITokenizer* tokenizer)
    : ApiEndpoint("benchmark", "/benchmark", "POST"),
      useDirectMode_(true),
      useIndependentScheduler_(false),
      generateEndpoint_(nullptr),
      scheduler_(scheduler),
      tokenizer_(tokenizer),
      maxBatchSize_(8),
      maxContextLength_(2048) {
}

BenchmarkEndpoint::BenchmarkEndpoint(ModelExecutor* modelExecutor, ITokenizer* tokenizer, 
                                     size_t maxBatchSize, size_t maxContextLength)
    : ApiEndpoint("benchmark", "/benchmark", "POST"),
      useDirectMode_(true),
      useIndependentScheduler_(true),
      generateEndpoint_(nullptr),
      scheduler_(nullptr),
      independentScheduler_(std::make_unique<Scheduler>(modelExecutor, maxBatchSize, maxContextLength)),
      tokenizer_(tokenizer),
      maxBatchSize_(maxBatchSize),
      maxContextLength_(maxContextLength) {
    // 🔥 优化：启动独立的Scheduler，确保独立运行（与Stage 15一致）
    independentScheduler_->start();
    std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 等待Scheduler启动（与Stage 15一致）
}

BenchmarkEndpoint::BenchmarkEndpoint(GenerateEndpoint* generateEndpoint)
    : ApiEndpoint("benchmark", "/benchmark", "POST"),
      useDirectMode_(false),
      useIndependentScheduler_(false),
      generateEndpoint_(generateEndpoint),
      scheduler_(nullptr),
      tokenizer_(nullptr),
      maxBatchSize_(8),
      maxContextLength_(2048) {
}

BenchmarkEndpoint::~BenchmarkEndpoint() {
    // 🔥 优化：停止独立的Scheduler（如果使用）
    if (useIndependentScheduler_ && independentScheduler_) {
        independentScheduler_->stop();
    }
}

void BenchmarkEndpoint::setGenerateEndpoint(GenerateEndpoint* generateEndpoint) {
    useDirectMode_ = false;
    generateEndpoint_ = generateEndpoint;
    scheduler_ = nullptr;
    tokenizer_ = nullptr;
}

void BenchmarkEndpoint::setSchedulerAndTokenizer(Scheduler* scheduler, ITokenizer* tokenizer) {
    useDirectMode_ = true;
    scheduler_ = scheduler;
    tokenizer_ = tokenizer;
    generateEndpoint_ = nullptr;
}

BenchmarkEndpoint::BenchmarkRequest BenchmarkEndpoint::parseRequest(const HttpRequest& request) {
    BenchmarkRequest req;
    
    nlohmann::json jsonBody;
    
    if (!JsonRequestParser::validateJson(request.getBody(), jsonBody)) {
        CLLM_WARN("Failed to parse JSON request body: %s, using default values", JsonRequestParser::getLastError().c_str());
        return req;
    }
    
    int defaultRequests = 40;
    int defaultConcurrency = 8;
    int defaultMaxTokens = 50;
    std::string defaultPrompt = "Hello, world! How are you today?";
    float defaultTemperature = 0.7f;
    
    JsonRequestParser::getFieldWithDefault(jsonBody, "requests", req.requests, defaultRequests);
    JsonRequestParser::getFieldWithDefault(jsonBody, "concurrency", req.concurrency, defaultConcurrency);
    JsonRequestParser::getFieldWithDefault(jsonBody, "max_tokens", req.maxTokens, defaultMaxTokens);
    JsonRequestParser::getFieldWithDefault(jsonBody, "prompt", req.prompt, defaultPrompt);
    JsonRequestParser::getFieldWithDefault(jsonBody, "temperature", req.temperature, defaultTemperature);
    
    // 参数验证
    if (req.requests <= 0) {
        req.requests = 40;
        CLLM_WARN("Invalid requests parameter, using default: 40");
    }
    if (req.concurrency <= 0) {
        req.concurrency = 8;
        CLLM_WARN("Invalid concurrency parameter, using default: 8");
    }
    if (req.maxTokens <= 0) {
        req.maxTokens = 50;
        CLLM_WARN("Invalid max_tokens parameter, using default: 50");
    }
    if (req.concurrency > req.requests) {
        req.concurrency = req.requests;
        CLLM_WARN("Concurrency exceeds requests, setting concurrency to requests: %d", req.concurrency);
    }
    
    return req;
}

BenchmarkEndpoint::RequestResult BenchmarkEndpoint::executeSingleRequest(
    const BenchmarkRequest& params, 
    int requestIndex
) {
    RequestResult result;
    
    if (!generateEndpoint_) {
        result.success = false;
        result.errorMessage = "GenerateEndpoint not initialized";
        return result;
    }
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    try {
        // 构建HttpRequest对象，模拟HTTP请求
        HttpRequest httpRequest;
        httpRequest.setMethod("POST");
        httpRequest.setPath("/generate");
        httpRequest.setHeader("Content-Type", "application/json");
        
        // 构建JSON请求体
        nlohmann::json requestJson;
        requestJson["prompt"] = params.prompt;
        requestJson["max_tokens"] = params.maxTokens;
        requestJson["temperature"] = params.temperature;
        requestJson["stream"] = false;
        httpRequest.setBody(requestJson.dump());
        
        // 直接调用GenerateEndpoint::handle()，避免HTTP开销
        HttpResponse httpResponse = generateEndpoint_->handle(httpRequest);
        
        auto endTime = std::chrono::high_resolution_clock::now();
        result.responseTime = std::chrono::duration<double>(endTime - startTime).count();
        
        // 解析响应
        if (httpResponse.getStatusCode() == 200) {
            try {
                nlohmann::json responseJson = nlohmann::json::parse(httpResponse.getBody());
                if (responseJson.contains("success") && responseJson["success"] == true) {
                    if (responseJson.contains("data")) {
                        auto data = responseJson["data"];
                        
                        // 提取tokens_per_second
                        if (data.contains("tokens_per_second")) {
                            result.tokensPerSecond = data["tokens_per_second"].get<float>();
                        }
                        
                        // 提取生成的文本和token数
                        if (data.contains("text")) {
                            std::string text = data["text"].get<std::string>();
                            // 简单估算token数（实际应该使用tokenizer）
                            if (result.tokensPerSecond > 0 && result.responseTime > 0) {
                                result.generatedTokens = static_cast<size_t>(result.tokensPerSecond * result.responseTime);
                            } else {
                                // 回退方案：根据文本长度估算
                                result.generatedTokens = text.length() / 4; // 粗略估算
                            }
                        }
                        
                        // 提取response_time（如果存在）
                        if (data.contains("response_time")) {
                            double responseTimeFromData = data["response_time"].get<double>();
                            if (responseTimeFromData > 0) {
                                result.responseTime = responseTimeFromData;
                            }
                        }
                        
                        result.totalTokens = params.prompt.length() / 4 + result.generatedTokens; // 粗略估算
                        result.success = true;
                    } else {
                        result.success = false;
                        result.errorMessage = "Response data field missing";
                    }
                } else {
                    result.success = false;
                    if (responseJson.contains("error")) {
                        result.errorMessage = responseJson["error"].get<std::string>();
                    } else {
                        result.errorMessage = "Request failed";
                    }
                }
            } catch (const std::exception& e) {
                result.success = false;
                result.errorMessage = std::string("Failed to parse response: ") + e.what();
            }
        } else {
            result.success = false;
            result.errorMessage = "HTTP " + std::to_string(httpResponse.getStatusCode());
        }
    } catch (const std::exception& e) {
        auto endTime = std::chrono::high_resolution_clock::now();
        result.responseTime = std::chrono::duration<double>(endTime - startTime).count();
        result.success = false;
        result.errorMessage = std::string("Exception: ") + e.what();
    }
    
    return result;
}

BenchmarkEndpoint::Statistics BenchmarkEndpoint::calculateStatistics(
    const std::vector<RequestResult>& results,
    double totalTime
) {
    Statistics stats;
    
    stats.totalRequests = static_cast<int>(results.size());
    stats.totalTime = totalTime;
    
    // 🔥 优化：单次遍历，避免多次拷贝和创建临时vector
    size_t successfulCount = 0;
    double totalResponseTime = 0.0;
    double minResponseTime = std::numeric_limits<double>::max();
    double maxResponseTime = 0.0;
    size_t totalGeneratedTokens = 0;
    size_t totalTokens = 0;
    double totalTokensPerSecond = 0.0;
    
    for (const auto& result : results) {
        if (result.success) {
            successfulCount++;
            totalResponseTime += result.responseTime;
            if (result.responseTime < minResponseTime) {
                minResponseTime = result.responseTime;
            }
            if (result.responseTime > maxResponseTime) {
                maxResponseTime = result.responseTime;
            }
            totalGeneratedTokens += result.generatedTokens;
            totalTokens += result.totalTokens;
            totalTokensPerSecond += result.tokensPerSecond;
        }
    }
    
    stats.successfulRequests = static_cast<int>(successfulCount);
    stats.failedRequests = stats.totalRequests - stats.successfulRequests;
    
    if (successfulCount == 0) {
        return stats;
    }
    
    // 计算统计
    stats.avgResponseTime = totalResponseTime / successfulCount;
    stats.minResponseTime = minResponseTime == std::numeric_limits<double>::max() ? 0.0 : minResponseTime;
    stats.maxResponseTime = maxResponseTime;
    
    stats.totalTokensProcessed = totalTokens;
    stats.avgGeneratedTokens = static_cast<double>(totalGeneratedTokens) / successfulCount;
    stats.avgTokensPerSecond = totalTokensPerSecond / successfulCount;
    
    // 计算平均吞吐量（总生成token数 / 总时间）
    if (totalTime > 0) {
        stats.avgThroughput = static_cast<double>(totalGeneratedTokens) / totalTime;
    }
    
    return stats;
}

HttpResponse BenchmarkEndpoint::buildResponse(const Statistics& stats) {
    nlohmann::json responseJson;
    responseJson["success"] = true;
    
    nlohmann::json dataJson;
    dataJson["total_requests"] = stats.totalRequests;
    dataJson["successful_requests"] = stats.successfulRequests;
    dataJson["failed_requests"] = stats.failedRequests;
    dataJson["avg_response_time"] = stats.avgResponseTime;
    dataJson["min_response_time"] = stats.minResponseTime;
    dataJson["max_response_time"] = stats.maxResponseTime;
    dataJson["avg_throughput"] = stats.avgThroughput;
    dataJson["avg_tokens_per_second"] = stats.avgTokensPerSecond;
    dataJson["total_tokens_processed"] = stats.totalTokensProcessed;
    dataJson["avg_generated_tokens"] = stats.avgGeneratedTokens;
    dataJson["total_time"] = stats.totalTime;
    
    responseJson["data"] = dataJson;
    
    return ResponseBuilder::json(responseJson, 200);
}

HttpResponse BenchmarkEndpoint::handle(const HttpRequest& request) {
    try {
        BenchmarkRequest params = parseRequest(request);
        
        // 🔥 优化：移除启动日志，减少开销
        // CLLM_INFO("Starting benchmark: requests=%d, concurrency=%d, max_tokens=%d",
        //           params.requests, params.concurrency, params.maxTokens);
        
        auto totalStartTime = std::chrono::high_resolution_clock::now();
        
        // 🔥 优化：参考Stage 15的实现，使用原子操作收集最小必要统计
        // 完全移除responseTimes收集，减少锁竞争
        std::atomic<size_t> completedRequests{0};
        std::atomic<size_t> totalGeneratedTokens{0};
        
        // 工作线程函数
        auto worker = [&](int startIndex, int count) {
            for (int i = 0; i < count; ++i) {
                int requestIndex = startIndex + i;
                if (requestIndex >= params.requests) {
                    break;
                }
                
                // 🔥 优化：直接调用Scheduler，不创建RequestResult对象
                // 优先使用独立的Scheduler实例（最优模式）
                Scheduler* activeScheduler = useIndependentScheduler_ ? independentScheduler_.get() : scheduler_;
                if (useDirectMode_ && activeScheduler && tokenizer_) {
                    try {
                        // 直接创建RequestState
                        RequestState requestState;
                        requestState.requestId = 0;
                        requestState.maxTokens = params.maxTokens;
                        requestState.temperature = params.temperature;
                        requestState.topP = 0.9f;
                        requestState.topK = 0;
                        requestState.repetitionPenalty = 1.1f; // 默认轻微惩罚重复
                        requestState.eosTokenId = tokenizer_->getEosId();
                        requestState.priority = 0;
                        requestState.arrivalTime = 0;
                        requestState.startTime = 0;
                        requestState.completionTime = 0;
                        requestState.isCompleted = false;
                        requestState.isRunning = false;
                        requestState.isFailed = false;
                        requestState.samplingStrategy = "";
                        requestState.errorMessage = "";
                        requestState.tokenizedPrompt = tokenizer_->encode(params.prompt, false);
                        
                        // 控制输入长度
                        const int maxInputTokens = cllm::Config::instance().httpMaxInputTokens();
                        if (maxInputTokens > 0 && requestState.tokenizedPrompt.size() > static_cast<size_t>(maxInputTokens)) {
                            requestState.tokenizedPrompt.resize(maxInputTokens);
                        }
                        
                        // 直接调用Scheduler（使用独立的或共享的）
                        size_t reqId = activeScheduler->addRequest(requestState);
                        const float timeoutMin = cllm::Config::instance().apiTimeoutMin();
                        const float timeoutMax = cllm::Config::instance().apiTimeoutMax();
                        const float tokenFactor = cllm::Config::instance().apiTimeoutTokenFactor();
                        const float timeoutSec = std::max(timeoutMin, std::min(timeoutMax, static_cast<float>(params.maxTokens) * tokenFactor));
                        
                        if (activeScheduler->waitForRequest(reqId, timeoutSec)) {
                            RequestState resultState = activeScheduler->getRequestResult(reqId);
                            if (!resultState.isTimeout && !resultState.isFailed && !resultState.generatedTokens.empty()) {
                                // 🔥 优化：直接更新原子变量，不创建RequestResult对象
                                completedRequests++;
                                totalGeneratedTokens += resultState.generatedTokens.size();
                            }
                        }
                    } catch (...) {
                        // 忽略错误，继续处理下一个请求
                    }
                } else {
                    // 回退到原有方式
                    RequestResult result = useDirectMode_ ? 
                        executeSingleRequestDirect(params, requestIndex) : 
                        executeSingleRequest(params, requestIndex);
                    if (result.success) {
                        completedRequests++;
                        totalGeneratedTokens += result.generatedTokens;
                    }
                }
            }
        };
        
        // 创建线程池
        std::vector<std::thread> threads;
        int requestsPerThread = params.requests / params.concurrency;
        int remainder = params.requests % params.concurrency;
        
        int currentIndex = 0;
        for (int i = 0; i < params.concurrency; ++i) {
            int threadRequests = requestsPerThread + (i < remainder ? 1 : 0);
            if (threadRequests > 0) {
                threads.emplace_back(worker, currentIndex, threadRequests);
                currentIndex += threadRequests;
            }
        }
        
        // 等待所有线程完成
        for (auto& thread : threads) {
            thread.join();
        }
        
        auto totalEndTime = std::chrono::high_resolution_clock::now();
        double totalTime = std::chrono::duration<double>(totalEndTime - totalStartTime).count();
        
        // 🔥 优化：直接从原子变量计算统计，完全无锁
        Statistics stats;
        stats.totalRequests = params.requests;
        stats.totalTime = totalTime;
        stats.successfulRequests = static_cast<int>(completedRequests.load());
        stats.failedRequests = stats.totalRequests - stats.successfulRequests;
        
        if (stats.successfulRequests > 0) {
            // 计算token统计
            size_t totalGenTokens = totalGeneratedTokens.load();
            stats.totalTokensProcessed = totalGenTokens; // 简化：只计算生成的token
            stats.avgGeneratedTokens = static_cast<double>(totalGenTokens) / stats.successfulRequests;
            
            // 计算平均吞吐量（主要指标）
            if (totalTime > 0) {
                stats.avgThroughput = static_cast<double>(totalGenTokens) / totalTime;
            }
            
            // 简化统计：使用总时间和总token数估算
            // 假设每个请求的平均响应时间 = 总时间 / 成功请求数
            stats.avgResponseTime = totalTime / stats.successfulRequests;
            stats.minResponseTime = 0.0;  // 简化：不收集
            stats.maxResponseTime = 0.0;  // 简化：不收集
            
            // 简化：使用平均吞吐量作为avg_tokens_per_second
            stats.avgTokensPerSecond = stats.avgThroughput;
        }
        
        // 🔥 优化：移除完成日志，减少开销（仅在DEBUG模式下输出）
        #ifdef CLLM_DEBUG_MODE
        CLLM_INFO("Benchmark completed: throughput=%.2f t/s, successful=%d/%d, total_time=%.2fs",
                  stats.avgThroughput, stats.successfulRequests, stats.totalRequests, stats.totalTime);
        #endif
        
        return buildResponse(stats);
        
    } catch (const std::exception& e) {
        CLLM_ERROR("Benchmark failed: %s", e.what());
        return ResponseBuilder::internalError(std::string("Benchmark error: ") + e.what());
    }
}

BenchmarkEndpoint::RequestResult BenchmarkEndpoint::executeSingleRequestDirect(
    const BenchmarkRequest& params, 
    int requestIndex
) {
    // 🔥 优化2: 使用返回值优化（RVO），避免不必要的拷贝
    RequestResult result;
    
    if (!scheduler_ || !tokenizer_) {
        result.success = false;
        result.errorMessage = "Scheduler or Tokenizer not initialized";
        return result;  // RVO优化
    }
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    try {
        // 🔥 优化：直接创建RequestState，无需JSON解析
        RequestState requestState;
        requestState.requestId = 0; // 由scheduler分配
        requestState.maxTokens = params.maxTokens;
        requestState.temperature = params.temperature;
        requestState.topP = 0.9f; // 使用默认值
        requestState.topK = 0; // 使用默认值
        requestState.repetitionPenalty = 1.1f; // 默认轻微惩罚重复
        requestState.eosTokenId = tokenizer_->getEosId();
        requestState.priority = 0;
        requestState.arrivalTime = 0;
        requestState.startTime = 0;
        requestState.completionTime = 0;
        requestState.isCompleted = false;
        requestState.isRunning = false;
        requestState.isFailed = false;
        requestState.samplingStrategy = "";
        requestState.errorMessage = "";
        
        // 🔥 优化：直接调用Tokenizer::encode()，无需JSON
        requestState.tokenizedPrompt = tokenizer_->encode(params.prompt, false);
        
        // 控制输入长度
        const int maxInputTokens = cllm::Config::instance().httpMaxInputTokens();
        if (maxInputTokens > 0) {
            const size_t MAX_INPUT_TOKENS = static_cast<size_t>(maxInputTokens);
            if (requestState.tokenizedPrompt.size() > MAX_INPUT_TOKENS) {
                CLLM_WARN("Input tokens (%zu) exceeds limit (%zu), truncating",
                          requestState.tokenizedPrompt.size(), MAX_INPUT_TOKENS);
                requestState.tokenizedPrompt.resize(MAX_INPUT_TOKENS);
            }
        }
        
        // 🔥 优化：直接调用Scheduler::addRequest()，无需GenerateEndpoint
        size_t reqId = scheduler_->addRequest(requestState);
        
        // 🔥 优化：直接调用Scheduler::waitForRequest()，无需HTTP层
        const float timeoutMin = cllm::Config::instance().apiTimeoutMin();
        const float timeoutMax = cllm::Config::instance().apiTimeoutMax();
        const float tokenFactor = cllm::Config::instance().apiTimeoutTokenFactor();
        const float timeoutSec = std::max(timeoutMin, std::min(timeoutMax, static_cast<float>(params.maxTokens) * tokenFactor));
        
        if (scheduler_->waitForRequest(reqId, timeoutSec)) {
            // 🔥 优化：直接调用Scheduler::getRequestResult()，无需JSON解析
            RequestState resultState = scheduler_->getRequestResult(reqId);
            
            if (resultState.isTimeout) {
                result.success = false;
                result.errorMessage = "Request timeout";
            } else if (resultState.isFailed) {
                result.success = false;
                result.errorMessage = resultState.errorMessage.empty() ? "Request failed" : resultState.errorMessage;
            } else {
                // 🔥 优化：直接使用RequestState中的token数，无需JSON解析
                result.generatedTokens = resultState.generatedTokens.size();
                result.totalTokens = requestState.tokenizedPrompt.size() + result.generatedTokens;
                
                // 计算tokens per second
                auto endTime = std::chrono::high_resolution_clock::now();
                result.responseTime = std::chrono::duration<double>(endTime - startTime).count();
                if (result.responseTime > 0) {
                    result.tokensPerSecond = static_cast<double>(result.generatedTokens) / result.responseTime;
                }
                
                result.success = true;
            }
        } else {
            result.success = false;
            result.errorMessage = "Request timeout (scheduler timeout)";
            auto endTime = std::chrono::high_resolution_clock::now();
            result.responseTime = std::chrono::duration<double>(endTime - startTime).count();
        }
    } catch (const SchedulerException& e) {
        auto endTime = std::chrono::high_resolution_clock::now();
        result.responseTime = std::chrono::duration<double>(endTime - startTime).count();
        result.success = false;
        result.errorMessage = std::string("Scheduler error: ") + e.what();
    } catch (const std::exception& e) {
        auto endTime = std::chrono::high_resolution_clock::now();
        result.responseTime = std::chrono::duration<double>(endTime - startTime).count();
        result.success = false;
        result.errorMessage = std::string("Exception: ") + e.what();
    }
    
    return result;
}

} // namespace cllm
