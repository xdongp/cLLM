#include "cllm/http/generate_endpoint.h"
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
    
    std::string body = request.getBody();
    
    // 默认值
    req.prompt = cllm::Config::instance().apiDefaultPrompt();
    req.maxTokens = cllm::Config::instance().apiDefaultMaxTokens();
    req.temperature = cllm::Config::instance().apiDefaultTemperature();
    req.topP = cllm::Config::instance().apiDefaultTopP();
    req.stream = false;
    
    try {
        nlohmann::json jsonBody = nlohmann::json::parse(body);
        
        if (jsonBody.contains("prompt") && jsonBody["prompt"].is_string()) {
            req.prompt = jsonBody["prompt"].get<std::string>();
        }
        
        if (jsonBody.contains("max_tokens") && jsonBody["max_tokens"].is_number_integer()) {
            req.maxTokens = jsonBody["max_tokens"].get<int>();
        }
        
        if (jsonBody.contains("temperature") && jsonBody["temperature"].is_number()) {
            req.temperature = jsonBody["temperature"].get<float>();
        }
        
        if (jsonBody.contains("top_p") && jsonBody["top_p"].is_number()) {
            req.topP = jsonBody["top_p"].get<float>();
        }
        
        if (jsonBody.contains("stream") && jsonBody["stream"].is_boolean()) {
            req.stream = jsonBody["stream"].get<bool>();
        }
    } catch (const nlohmann::json::exception& e) {
        CLLM_WARN("Failed to parse JSON request body: %s, using default values", e.what());
    }
    
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
        HttpResponse response;
        response.setError(500, std::string("Error handling request: ") + e.what());
        return response;
    }
}

HttpResponse GenerateEndpoint::handleNonStreaming(const GenerateRequest& req) {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    std::string requestId = generateRequestId();
    std::string generatedText = "";
    size_t generatedTokenCount = 0;
    
    if (scheduler_ != nullptr && tokenizer_ != nullptr) {
        try {
            CLLM_DEBUG("Starting non-streaming request processing");
            CLLM_DEBUG("Prompt: %s", req.prompt.c_str());
            CLLM_DEBUG("Max tokens: %d", req.maxTokens);
            CLLM_DEBUG("Temperature: %f", req.temperature);
            
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
            CLLM_DEBUG("Starting tokenization...");
            requestState.tokenizedPrompt = tokenizer_->encode(req.prompt, true);
            CLLM_DEBUG("Tokenization completed, got %zu tokens", requestState.tokenizedPrompt.size());
            
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
            
            // 添加请求到调度器
            CLLM_DEBUG("Adding request to scheduler...");
            size_t reqId = scheduler_->addRequest(requestState);
            CLLM_DEBUG("Request added with ID: %zu", reqId);
            
            // 等待请求完成
            const float timeoutMin = cllm::Config::instance().apiTimeoutMin();
            const float timeoutMax = cllm::Config::instance().apiTimeoutMax();
            const float tokenFactor = cllm::Config::instance().apiTimeoutTokenFactor();
            const float timeoutSec = std::max(timeoutMin, std::min(timeoutMax, static_cast<float>(req.maxTokens) * tokenFactor));
            CLLM_DEBUG("Waiting for request completion (timeout=%.1fs)...", timeoutSec);
            if (scheduler_->waitForRequest(reqId, timeoutSec)) {
                CLLM_DEBUG("Request completed, retrieving result...");
                RequestState result = scheduler_->getRequestResult(reqId);
                
                CLLM_DEBUG("Tokenized prompt in result: %zu tokens", result.tokenizedPrompt.size());
                CLLM_DEBUG("Generated tokens count: %zu", result.generatedTokens.size());
                
                if (!result.generatedTokens.empty()) {
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

                    // 解码前：按 EOS 截断，避免 EOS 后继续采样导致“乱码”
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

                    CLLM_DEBUG("Decoding tokens...");
                    generatedText = tokenizer_->decode(toDecode, true);
                    CLLM_DEBUG("Decoded text: [%s]", generatedText.c_str());
                    CLLM_DEBUG("Decoded text length: %zu", generatedText.length());
                } else {
                    CLLM_WARN("No tokens generated!");
                    generatedText = "No tokens generated";
                }
            } else {
                CLLM_ERROR("Request timed out");
                generatedText = "Request timed out";
            }
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

    HttpResponse response;
    response.setStatusCode(200);
    response.setBody(resp.dump());
    response.setContentType(cllm::Config::instance().apiResponseContentTypeJson());
    
    return response;
}

HttpResponse GenerateEndpoint::handleStreaming(const GenerateRequest& req) {
    std::string requestId = generateRequestId();
    
    HttpResponse response;
    response.setStatusCode(200);
    response.enableStreaming();
    response.setContentType(cllm::Config::instance().apiResponseContentTypeStream());
    response.setHeader("Cache-Control", cllm::Config::instance().apiResponseHeaderCacheControl());
    response.setHeader("Connection", cllm::Config::instance().apiResponseHeaderConnection());
    
    if (tokenizer_ != nullptr) {
        std::vector<int> inputTokens = tokenizer_->encode(req.prompt, true);
        
        std::string generatedText;
        for (int i = 0; i < req.maxTokens; ++i) {
            // 测试英文生成 - 暂时只生成ASCII字符
            int nextToken;
            // 生成ASCII字符
            nextToken = 32 + (rand() % 95); // 32-126是可打印ASCII字符
            
            // ITokenizer 没有 isSpecialToken()，这里做最小“特殊 token”保护
            if (nextToken == tokenizer_->getEosId() ||
                nextToken == tokenizer_->getPadId() ||
                nextToken == tokenizer_->getBosId() ||
                nextToken == tokenizer_->getUnkId()) {
                break;
            }

            std::string tokenText;
            try {
                tokenText = tokenizer_->decode({nextToken}, false);
            } catch (...) {
                break;
            }
            generatedText += tokenText;
            
            nlohmann::json chunk;
            chunk["id"] = requestId;
            chunk["token"] = tokenText;
            chunk["done"] = false;

            std::ostringstream oss;
            oss << "data: " << chunk.dump() << "\n\n";
            response.addChunk(oss.str());
        }
        
        // Send final done message
        nlohmann::json finalChunk;
        finalChunk["id"] = requestId;
        finalChunk["token"] = "";
        finalChunk["done"] = true;

        std::ostringstream finalOss;
        finalOss << "data: " << finalChunk.dump() << "\n\n";
        response.addChunk(finalOss.str());
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