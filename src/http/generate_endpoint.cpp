#include "cllm/http/generate_endpoint.h"
#include "cllm/http/json_request_parser.h"
#include "cllm/http/response_builder.h"
#include "cllm/scheduler/scheduler.h"
#include "cllm/tokenizer/i_tokenizer.h"
#include "cllm/common/logger.h"
#include "cllm/common/config.h"
#include <nlohmann/json.hpp>
#include <sstream>
#include <chrono>
#include <random>
#include <algorithm>

namespace cllm {

GenerateEndpoint::GenerateEndpoint(Scheduler* scheduler, ITokenizer* tokenizer)
    : ApiEndpoint(cllm::Config::instance().apiEndpointGenerateName(), cllm::Config::instance().apiEndpointGeneratePath(), cllm::Config::instance().apiEndpointGenerateMethod()),
      scheduler_(scheduler),
      tokenizer_(tokenizer) {
}

GenerateEndpoint::~GenerateEndpoint() {
}

void GenerateEndpoint::setScheduler(Scheduler* scheduler) {
    scheduler_ = scheduler;
}

void GenerateEndpoint::setTokenizer(ITokenizer* tokenizer) {
    tokenizer_ = tokenizer;
}

GenerateEndpoint::GenerateRequest GenerateEndpoint::parseRequest(const HttpRequest& request) {
    GenerateRequest req;
    
    nlohmann::json jsonBody;
    
    if (!JsonRequestParser::validateJson(request.getBody(), jsonBody)) {
        CLLM_WARN("Failed to parse JSON request body: %s, using default values", JsonRequestParser::getLastError().c_str());
    }
    
    JsonRequestParser::getFieldWithDefault(jsonBody, "prompt", req.prompt, cllm::Config::instance().apiDefaultPrompt());
    JsonRequestParser::getFieldWithDefault(jsonBody, "max_tokens", req.maxTokens, cllm::Config::instance().apiDefaultMaxTokens());
    JsonRequestParser::getFieldWithDefault(jsonBody, "temperature", req.temperature, cllm::Config::instance().apiDefaultTemperature());
    JsonRequestParser::getFieldWithDefault(jsonBody, "top_p", req.topP, cllm::Config::instance().apiDefaultTopP());
    JsonRequestParser::getFieldWithDefault(jsonBody, "stream", req.stream, false);
    
    // 调试：打印解析后的参数
    CLLM_INFO("[GenerateEndpoint] Parsed request: prompt='%s', max_tokens=%d, temperature=%.4f, top_p=%.4f",
              req.prompt.c_str(), req.maxTokens, req.temperature, req.topP);
    
    return req;
}

HttpResponse GenerateEndpoint::handle(const HttpRequest& request) {
    try {
        GenerateRequest req = parseRequest(request);
        
        if (req.stream) {
            return handleStreaming(req);
        } else {
            return handleNonStreaming(req);
        }
    } catch (const std::exception& e) {
        return ResponseBuilder::internalError(std::string("Error handling request: ") + e.what());
    }
}

HttpResponse GenerateEndpoint::handleNonStreaming(const GenerateRequest& req) {
    // 🔥 优化：延迟开始时间测量，排除JSON解析等非核心开销
    // 在真正开始处理请求时才开始计时（与Stage 15对齐）
    std::string requestId = generateRequestId();
    std::string generatedText = "";
    size_t generatedTokenCount = 0;
    
    // 🔥 关键优化：在tokenization之前开始计时（与Stage 15对齐）
    auto startTime = std::chrono::high_resolution_clock::now();
    
    if (scheduler_ != nullptr && tokenizer_ != nullptr) {
        try {
            #ifdef CLLM_DEBUG_MODE
            CLLM_DEBUG("Starting non-streaming request processing");
            CLLM_DEBUG("Prompt: %s", req.prompt.c_str());
            CLLM_DEBUG("Max tokens: %d", req.maxTokens);
            CLLM_DEBUG("Temperature: %f", req.temperature);
            #endif
            
            // 创建请求状态
            RequestState requestState;
            requestState.requestId = 0; // 由scheduler分配
            requestState.maxTokens = req.maxTokens;
            requestState.temperature = req.temperature;
            requestState.topP = req.topP;
            requestState.topK = 0; // 使用默认值

            // 从 tokenizer 注入 EOS，确保调度/批处理能正确停止
            requestState.eosTokenId = tokenizer_->getEosId();

            requestState.priority = 0;
            requestState.arrivalTime = 0;
            requestState.startTime = 0;
            requestState.completionTime = 0;
            requestState.isCompleted = false; // 明确初始化
            requestState.isRunning = false;
            requestState.isFailed = false;
            requestState.samplingStrategy = "";
            requestState.errorMessage = "";
            
            // 编码prompt
            #ifdef CLLM_DEBUG_MODE
            CLLM_DEBUG("Starting tokenization...");
            #endif
            requestState.tokenizedPrompt = tokenizer_->encode(req.prompt, false);
            #ifdef CLLM_DEBUG_MODE
            CLLM_DEBUG("Tokenization completed, got %zu tokens", requestState.tokenizedPrompt.size());
            #endif
            
            // 控制输入长度：TorchScript trace 可能固化 seq_len（当前模型为 128），过长输入会导致推理开销变大
            // 这里做一个温和的上限，避免超长 prompt 把 CPU 推理拖垮；真正的裁剪/填充由后端按 traced seq_len 处理
            const int maxInputTokens = cllm::Config::instance().httpMaxInputTokens();
            if (maxInputTokens > 0) {
                const size_t MAX_INPUT_TOKENS = static_cast<size_t>(maxInputTokens);
                if (requestState.tokenizedPrompt.size() > MAX_INPUT_TOKENS) {
                    CLLM_WARN("Input tokens (%zu) exceeds limit (%zu), truncating",
                              requestState.tokenizedPrompt.size(), MAX_INPUT_TOKENS);
                    requestState.tokenizedPrompt.resize(MAX_INPUT_TOKENS);
                }
            }
            
            #ifdef CLLM_DEBUG_MODE
            if (!requestState.tokenizedPrompt.empty()) {
                CLLM_DEBUG("Token IDs: [");
                size_t showCount = std::min(requestState.tokenizedPrompt.size(), (size_t)10);
                std::stringstream tokenIds;
                for (size_t i = 0; i < showCount; ++i) {
                    tokenIds << " " << requestState.tokenizedPrompt[i];
                }
                if (requestState.tokenizedPrompt.size() > showCount) {
                    tokenIds << " ...";
                }
                tokenIds << " ]";
                CLLM_DEBUG("%s", tokenIds.str().c_str());
            }
            #endif
            
            // Phase 6: 检查并发限制
            size_t runningCount = scheduler_->getRunningCount();
            size_t maxConcurrent = scheduler_->getMaxConcurrentRequests();
            if (runningCount >= maxConcurrent) {
                #ifdef CLLM_DEBUG_MODE
                CLLM_WARN("Concurrent request limit reached: %zu/%zu, returning HTTP 429", runningCount, maxConcurrent);
                #endif
                nlohmann::json errorResp;
                errorResp["success"] = false;
                errorResp["error"] = "Too many concurrent requests";
                errorResp["message"] = "Server is currently at maximum capacity. Please try again later.";
                errorResp["retry_after"] = 5;  // 建议重试时间（秒）
                HttpResponse response = ResponseBuilder::json(errorResp, 429);
                response.setHeader("Retry-After", "5");
                return response;
            }
            
            // 添加请求到调度器
            #ifdef CLLM_DEBUG_MODE
            CLLM_DEBUG("Adding request to scheduler...");
            #endif
            size_t reqId = scheduler_->addRequest(requestState);
            #ifdef CLLM_DEBUG_MODE
            CLLM_DEBUG("Request added with ID: %zu", reqId);
            #endif
            
            // 等待请求完成
            const float timeoutMin = cllm::Config::instance().apiTimeoutMin();
            const float timeoutMax = cllm::Config::instance().apiTimeoutMax();
            const float tokenFactor = cllm::Config::instance().apiTimeoutTokenFactor();
            const float timeoutSec = std::max(timeoutMin, std::min(timeoutMax, static_cast<float>(req.maxTokens) * tokenFactor));
            #ifdef CLLM_DEBUG_MODE
            CLLM_DEBUG("Waiting for request completion (timeout=%.1fs)...", timeoutSec);
            #endif
            if (scheduler_->waitForRequest(reqId, timeoutSec)) {
                #ifdef CLLM_DEBUG_MODE
                CLLM_DEBUG("Request completed, retrieving result...");
                #endif
                RequestState result = scheduler_->getRequestResult(reqId);
                
                if (result.isTimeout) {
                    #ifdef CLLM_DEBUG_MODE
                    CLLM_WARN("Request timed out (scheduler timeout)");
                    #endif
                    nlohmann::json errorResp;
                    errorResp["success"] = false;
                    errorResp["error"] = "Request timeout";
                    errorResp["message"] = "Request timed out";
                    return ResponseBuilder::json(errorResp, 408);
                }
                
                #ifdef CLLM_DEBUG_MODE
                CLLM_DEBUG("Tokenized prompt in result: %zu tokens", result.tokenizedPrompt.size());
                CLLM_DEBUG("Generated tokens count: %zu", result.generatedTokens.size());
                CLLM_DEBUG("Request ID: %llu, isCompleted: %d, isFailed: %d, isTimeout: %d", 
                          result.requestId, result.isCompleted ? 1 : 0, result.isFailed ? 1 : 0, result.isTimeout ? 1 : 0);
                #endif
                
                if (!result.generatedTokens.empty()) {
                    #ifdef CLLM_DEBUG_MODE
                    CLLM_DEBUG("Generated tokens: [");
                    size_t showCount = std::min(result.generatedTokens.size(), (size_t)10);
                    std::stringstream generatedTokens;
                    for (size_t i = 0; i < showCount; ++i) {
                        generatedTokens << " " << result.generatedTokens[i];
                    }
                    if (result.generatedTokens.size() > showCount) {
                        generatedTokens << " ...";
                    }
                    generatedTokens << " ]";
                    CLLM_DEBUG("%s", generatedTokens.str().c_str());
                    #endif

                    // 解码前：按 EOS 截断，避免 EOS 后继续采样导致"乱码"
                    std::vector<int> toDecode = result.generatedTokens;
                    const int eosId = tokenizer_->getEosId();
                    if (eosId >= 0) {
                        for (size_t k = 0; k < toDecode.size(); ++k) {
                            if (toDecode[k] == eosId) {
                                toDecode.resize(k);
                                break;
                            }
                        }
                    }

                    generatedTokenCount = toDecode.size();

                    try {
                        generatedText = tokenizer_->decode(toDecode, true);
                        #ifdef CLLM_DEBUG_MODE
                        CLLM_DEBUG("Decoded text: [%s]", generatedText.c_str());
                        CLLM_DEBUG("Decoded text length: %zu", generatedText.length());
                        #endif
                    } catch (const std::exception& e) {
                        CLLM_ERROR("Exception during tokenizer decode: %s", e.what());
                        generatedText = "[Decode Error: " + std::string(e.what()) + "]";
                    }
                } else {
                    #ifdef CLLM_DEBUG_MODE
                    CLLM_WARN("No tokens generated!");
                    #endif
                    generatedText = "No tokens generated";
                }
            } else {
                CLLM_ERROR("Request timed out");
                nlohmann::json errorResp;
                errorResp["success"] = false;
                errorResp["error"] = "Request timeout";
                errorResp["message"] = "Request timed out";
                return ResponseBuilder::json(errorResp, 408);
            }
        } catch (const SchedulerException& e) {
            if (e.getError() == SchedulerError::REQUEST_QUEUE_FULL) {
                #ifdef CLLM_DEBUG_MODE
                CLLM_WARN("Request rejected: queue full");
                #endif
                nlohmann::json errorResp;
                errorResp["success"] = false;
                errorResp["error"] = "Request queue is full";
                errorResp["message"] = "Server is currently at maximum capacity. Please try again later.";
                errorResp["retry_after"] = 5;
                HttpResponse response = ResponseBuilder::json(errorResp, 429);
                response.setHeader("Retry-After", "5");
                return response;
            }
            CLLM_ERROR("Scheduler error: %s", e.what());
            generatedText = std::string("Error: ") + e.what();
        } catch (const std::exception& e) {
            CLLM_ERROR("Error processing request: %s", e.what());
            generatedText = std::string("Error: ") + e.what();
        }
    } else {
        CLLM_ERROR("Scheduler or tokenizer not initialized");
        generatedText = "Server not ready";
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    float responseTime = std::chrono::duration<float>(endTime - startTime).count();
    
    // TPS 使用实际生成 token 数量（req.maxTokens 只是上限）
    float tokensPerSecond = 0.0f;
    if (responseTime > 0.0f) {
        tokensPerSecond = static_cast<float>(generatedTokenCount) / responseTime;
    }
    
    // 用 JSON 库构造响应，确保 text 等字段正确转义（避免出现双引号导致 JSON 断裂）
    nlohmann::json resp;
    resp["id"] = requestId;
    resp["text"] = generatedText;
    resp["response_time"] = responseTime;
    resp["tokens_per_second"] = tokensPerSecond;

    return ResponseBuilder::success(resp);
}

HttpResponse GenerateEndpoint::handleStreaming(const GenerateRequest& req) {
    std::string requestId = generateRequestId();
    
    HttpResponse response;
    response.setStatusCode(200);
    response.enableStreaming();
    response.setContentType(cllm::Config::instance().apiResponseContentTypeStream());
    response.setHeader("Cache-Control", cllm::Config::instance().apiResponseHeaderCacheControl());
    response.setHeader("Connection", cllm::Config::instance().apiResponseHeaderConnection());
    
    if (scheduler_ == nullptr || tokenizer_ == nullptr) {
        nlohmann::json errorChunk;
        errorChunk["id"] = requestId;
        errorChunk["error"] = "Server not ready";
        errorChunk["done"] = true;
        std::ostringstream oss;
        oss << "data: " << errorChunk.dump() << "\n\n";
        response.addChunk(oss.str());
        return response;
    }
    
    try {
        // 创建请求状态
        RequestState requestState;
        requestState.requestId = 0;
        requestState.maxTokens = req.maxTokens;
        requestState.temperature = req.temperature;
        requestState.topP = req.topP;
        requestState.topK = 0;
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
        
        // 编码prompt
        requestState.tokenizedPrompt = tokenizer_->encode(req.prompt, false);
        
        // 控制输入长度
        const int maxInputTokens = cllm::Config::instance().httpMaxInputTokens();
        if (maxInputTokens > 0) {
            const size_t MAX_INPUT_TOKENS = static_cast<size_t>(maxInputTokens);
            if (requestState.tokenizedPrompt.size() > MAX_INPUT_TOKENS) {
                requestState.tokenizedPrompt.resize(MAX_INPUT_TOKENS);
            }
        }
        
        // 检查并发限制
        size_t runningCount = scheduler_->getRunningCount();
        size_t maxConcurrent = scheduler_->getMaxConcurrentRequests();
        if (runningCount >= maxConcurrent) {
            nlohmann::json errorChunk;
            errorChunk["id"] = requestId;
            errorChunk["error"] = "Too many concurrent requests";
            errorChunk["done"] = true;
            std::ostringstream oss;
            oss << "data: " << errorChunk.dump() << "\n\n";
            response.addChunk(oss.str());
            return response;
        }
        
        // 添加请求到调度器
        size_t reqId = scheduler_->addRequest(requestState);
        
        // 等待请求完成（流式场景下，每个 token 都需要从 scheduler 拉取）
        // 这里先实现简化版：等待完成后逐 token 返回（非真正实时流式）
        const float timeoutMin = cllm::Config::instance().apiTimeoutMin();
        const float timeoutMax = cllm::Config::instance().apiTimeoutMax();
        const float tokenFactor = cllm::Config::instance().apiTimeoutTokenFactor();
        const float timeoutSec = std::max(timeoutMin, std::min(timeoutMax, static_cast<float>(req.maxTokens) * tokenFactor));
        
        if (scheduler_->waitForRequest(reqId, timeoutSec)) {
            RequestState result = scheduler_->getRequestResult(reqId);
            
            if (result.isTimeout) {
                nlohmann::json errorChunk;
                errorChunk["id"] = requestId;
                errorChunk["error"] = "Request timeout";
                errorChunk["done"] = true;
                std::ostringstream oss;
                oss << "data: " << errorChunk.dump() << "\n\n";
                response.addChunk(oss.str());
                return response;
            }
            
            if (!result.generatedTokens.empty()) {
                // 按 EOS 截断
                std::vector<int> toDecode = result.generatedTokens;
                const int eosId = tokenizer_->getEosId();
                if (eosId >= 0) {
                    for (size_t k = 0; k < toDecode.size(); ++k) {
                        if (toDecode[k] == eosId) {
                            toDecode.resize(k);
                            break;
                        }
                    }
                }
                
                // 逐 token 解码并返回（模拟流式输出）
                for (size_t i = 0; i < toDecode.size(); ++i) {
                    std::string tokenText;
                    try {
                        tokenText = tokenizer_->decode({toDecode[i]}, false);
                    } catch (...) {
                        continue;
                    }
                    
                    nlohmann::json chunk;
                    chunk["id"] = requestId;
                    chunk["token"] = tokenText;
                    chunk["done"] = false;
                    
                    std::ostringstream oss;
                    oss << "data: " << chunk.dump() << "\n\n";
                    response.addChunk(oss.str());
                }
            }
        } else {
            nlohmann::json errorChunk;
            errorChunk["id"] = requestId;
            errorChunk["error"] = "Request timeout";
            errorChunk["done"] = true;
            std::ostringstream oss;
            oss << "data: " << errorChunk.dump() << "\n\n";
            response.addChunk(oss.str());
            return response;
        }
        
        // 发送完成消息
        nlohmann::json finalChunk;
        finalChunk["id"] = requestId;
        finalChunk["token"] = "";
        finalChunk["done"] = true;
        
        std::ostringstream finalOss;
        finalOss << "data: " << finalChunk.dump() << "\n\n";
        response.addChunk(finalOss.str());
        
    } catch (const SchedulerException& e) {
        if (e.getError() == SchedulerError::REQUEST_QUEUE_FULL) {
            nlohmann::json errorChunk;
            errorChunk["id"] = requestId;
            errorChunk["error"] = "Request queue is full";
            errorChunk["done"] = true;
            std::ostringstream oss;
            oss << "data: " << errorChunk.dump() << "\n\n";
            response.addChunk(oss.str());
            return response;
        }
        nlohmann::json errorChunk;
        errorChunk["id"] = requestId;
        errorChunk["error"] = std::string("Scheduler error: ") + e.what();
        errorChunk["done"] = true;
        std::ostringstream oss;
        oss << "data: " << errorChunk.dump() << "\n\n";
        response.addChunk(oss.str());
    } catch (const std::exception& e) {
        nlohmann::json errorChunk;
        errorChunk["id"] = requestId;
        errorChunk["error"] = std::string("Error: ") + e.what();
        errorChunk["done"] = true;
        std::ostringstream oss;
        oss << "data: " << errorChunk.dump() << "\n\n";
        response.addChunk(oss.str());
    }
    
    return response;
}

std::string GenerateEndpoint::generateRequestId() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, 15);
    
    const char hexChars[] = "0123456789abcdef";
    std::string id;
    
    for (int i = 0; i < 32; ++i) {
        id += hexChars[dis(gen)];
    }
    
    return id;
}

} // namespace cllm