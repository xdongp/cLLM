/**
 * @file test_llama_cpp_direct.cpp
 * @brief 直接测试 llama.cpp 的性能上限
 * 
 * 绕过 cLLM 的调度器和 HTTP 层，直接使用 llama.cpp API 进行并发测试
 */

#include "llama.h"
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>
#include <mutex>
#include <string>
#include <cstring>
#include <random>

struct TestResult {
    size_t requestId;
    bool success;
    size_t tokensGenerated;
    double responseTime;
    std::string error;
};

class LlamaCppDirectBenchmark {
private:
    llama_model* model_;
    std::vector<llama_context*> contexts_;
    std::mutex contextMutex_;
    size_t numContexts_;
    int32_t nSeqMax_;
    int32_t nCtx_;
    int32_t nBatch_;
    int32_t nThreads_;
    
public:
    LlamaCppDirectBenchmark(
        const std::string& modelPath,
        size_t numContexts = 1,
        int32_t nSeqMax = 32,
        int32_t nCtx = 2048,
        int32_t nBatch = 512,
        int32_t nThreads = 8
    ) : model_(nullptr), numContexts_(numContexts), nSeqMax_(nSeqMax), 
        nCtx_(nCtx), nBatch_(nBatch), nThreads_(nThreads) {
        
        // 初始化模型参数
        llama_model_params modelParams = llama_model_default_params();
        modelParams.n_gpu_layers = 0;  // CPU only
        
        // 加载模型
        std::cout << "Loading model from: " << modelPath << std::endl;
        model_ = llama_model_load_from_file(modelPath.c_str(), modelParams);
        if (!model_) {
            throw std::runtime_error("Failed to load model");
        }
        std::cout << "Model loaded successfully" << std::endl;
        
        // 创建多个上下文（用于并发）
        contexts_.reserve(numContexts_);
        for (size_t i = 0; i < numContexts_; ++i) {
            llama_context_params ctxParams = llama_context_default_params();
            ctxParams.n_ctx = nCtx_;
            ctxParams.n_batch = nBatch_;
            ctxParams.n_threads = nThreads_;
            ctxParams.n_threads_batch = nThreads_;
            ctxParams.n_seq_max = nSeqMax_;
            
            llama_context* ctx = llama_new_context_with_model(model_, ctxParams);
            if (!ctx) {
                throw std::runtime_error("Failed to create context");
            }
            contexts_.push_back(ctx);
        }
        std::cout << "Created " << numContexts_ << " contexts" << std::endl;
    }
    
    ~LlamaCppDirectBenchmark() {
        for (auto* ctx : contexts_) {
            if (ctx) {
                llama_free(ctx);
            }
        }
        if (model_) {
            llama_model_free(model_);
        }
    }
    
    llama_context* getContext(size_t index) {
        return contexts_[index % contexts_.size()];
    }
    
    TestResult processRequest(
        size_t requestId,
        const std::string& prompt,
        size_t maxTokens = 50,
        size_t contextIndex = 0
    ) {
        TestResult result;
        result.requestId = requestId;
        result.success = false;
        result.tokensGenerated = 0;
        
        auto startTime = std::chrono::high_resolution_clock::now();
        
        try {
            llama_context* ctx = getContext(contextIndex);
            std::lock_guard<std::mutex> lock(contextMutex_);
            
            // Tokenize prompt
            std::vector<llama_token> tokens = llama_tokenize(ctx, prompt, true);
            if (tokens.empty()) {
                result.error = "Failed to tokenize prompt";
                return result;
            }
            
            // 分配序列ID（使用简单的映射：requestId % nSeqMax_）
            llama_seq_id seqId = static_cast<llama_seq_id>(requestId % nSeqMax_);
            
            // 准备 batch
            llama_batch batch = llama_batch_init(static_cast<int32_t>(tokens.size()), 0, 1);
            for (size_t i = 0; i < tokens.size(); ++i) {
                batch.token[i] = tokens[i];
                batch.pos[i] = static_cast<llama_pos>(i);
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = seqId;
                batch.logits[i] = (i == tokens.size() - 1);
            }
            batch.n_tokens = static_cast<int32_t>(tokens.size());
            
            // Decode prompt
                if (llama_decode(ctx, batch) != 0) {
                    result.error = "Failed to decode prompt";
                    llama_batch_free(batch);
                    llama_kv_cache_seq_rm(ctx, seqId, -1, -1);
                    return result;
                }
            
            // Generate tokens
            size_t nCur = tokens.size();
            while (nCur < tokens.size() + maxTokens) {
                // Get logits
                float* logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
                size_t vocabSize = llama_n_vocab(model_);
                
                // Simple greedy sampling
                llama_token newTokenId = 0;
                float maxLogit = logits[0];
                for (size_t i = 1; i < vocabSize; ++i) {
                    if (logits[i] > maxLogit) {
                        maxLogit = logits[i];
                        newTokenId = static_cast<llama_token>(i);
                    }
                }
                
                // Check for EOS
                if (newTokenId == llama_token_eos(model_)) {
                    break;
                }
                
                // Prepare next batch
                llama_batch_clear(&batch);
                batch.token[0] = newTokenId;
                batch.pos[0] = static_cast<llama_pos>(nCur);
                batch.n_seq_id[0] = 1;
                batch.seq_id[0][0] = seqId;
                batch.logits[0] = true;
                batch.n_tokens = 1;
                
                // Decode
                if (llama_decode(ctx, batch) != 0) {
                    result.error = "Failed to decode during generation";
                    break;
                }
                
                nCur++;
                result.tokensGenerated++;
            }
            
            // Cleanup
            llama_batch_free(batch);
            // Clear sequence from KV cache
            llama_kv_cache_seq_rm(ctx, seqId, -1, -1);
            
            result.success = true;
        } catch (const std::exception& e) {
            result.error = e.what();
        }
        
        auto endTime = std::chrono::high_resolution_clock::now();
        result.responseTime = std::chrono::duration<double>(endTime - startTime).count();
        
        return result;
    }
    
private:
    std::vector<llama_token> llama_tokenize(llama_context* ctx, const std::string& text, bool addBos) {
        std::vector<llama_token> tokens;
        int nTokens = llama_tokenize(ctx, text.c_str(), text.length(), nullptr, 0, addBos, false);
        if (nTokens < 0) {
            return tokens;
        }
        tokens.resize(nTokens);
        llama_tokenize(ctx, text.c_str(), text.length(), tokens.data(), tokens.size(), addBos, false);
        return tokens;
    }
};

void runConcurrentTest(
    LlamaCppDirectBenchmark& benchmark,
    const std::vector<std::string>& prompts,
    size_t numRequests,
    size_t concurrency,
    size_t maxTokens
) {
    std::vector<TestResult> results;
    results.reserve(numRequests);
    std::mutex resultsMutex;
    std::atomic<size_t> completedRequests(0);
    std::atomic<size_t> failedRequests(0);
    
    auto worker = [&](size_t workerId) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<size_t> promptDist(0, prompts.size() - 1);
        
        size_t requestsPerWorker = numRequests / concurrency;
        size_t startRequest = workerId * requestsPerWorker;
        size_t endRequest = (workerId == concurrency - 1) ? numRequests : startRequest + requestsPerWorker;
        
        for (size_t i = startRequest; i < endRequest; ++i) {
            const std::string& prompt = prompts[promptDist(gen)];
            TestResult result = benchmark.processRequest(i, prompt, maxTokens, workerId);
            
            {
                std::lock_guard<std::mutex> lock(resultsMutex);
                results.push_back(result);
            }
            
            if (result.success) {
                completedRequests++;
            } else {
                failedRequests++;
            }
            
            if ((i + 1) % 10 == 0) {
                std::cout << "  Request " << (i + 1) << "/" << numRequests 
                         << " completed (success: " << completedRequests 
                         << ", failed: " << failedRequests << ")" << std::endl;
            }
        }
    };
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    std::vector<std::thread> threads;
    for (size_t i = 0; i < concurrency; ++i) {
        threads.emplace_back(worker, i);
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    double totalTime = std::chrono::duration<double>(endTime - startTime).count();
    
    // Calculate statistics
    double totalResponseTime = 0.0;
    size_t totalTokens = 0;
    double minResponseTime = std::numeric_limits<double>::max();
    double maxResponseTime = 0.0;
    
    for (const auto& result : results) {
        if (result.success) {
            totalResponseTime += result.responseTime;
            totalTokens += result.tokensGenerated;
            minResponseTime = std::min(minResponseTime, result.responseTime);
            maxResponseTime = std::max(maxResponseTime, result.responseTime);
        }
    }
    
    size_t successfulRequests = completedRequests.load();
    double avgResponseTime = successfulRequests > 0 ? totalResponseTime / successfulRequests : 0.0;
    double avgThroughput = totalTime > 0 ? totalTokens / totalTime : 0.0;
    double avgTokensPerSecond = successfulRequests > 0 ? totalTokens / totalResponseTime : 0.0;
    
    // Print results
    std::cout << "\n========================================\n";
    std::cout << "llama.cpp Direct Performance Test Results\n";
    std::cout << "========================================\n";
    std::cout << "Total requests: " << numRequests << std::endl;
    std::cout << "Successful requests: " << successfulRequests 
              << " (" << (100.0 * successfulRequests / numRequests) << "%)" << std::endl;
    std::cout << "Failed requests: " << failedRequests.load() << std::endl;
    std::cout << "Avg response time: " << avgResponseTime << "s" << std::endl;
    std::cout << "Min response time: " << minResponseTime << "s" << std::endl;
    std::cout << "Max response time: " << maxResponseTime << "s" << std::endl;
    std::cout << "Total time: " << totalTime << "s" << std::endl;
    std::cout << "Avg throughput: " << avgThroughput << " tokens/sec" << std::endl;
    std::cout << "Avg tokens per second: " << avgTokensPerSecond << " tokens/sec" << std::endl;
    std::cout << "Total tokens processed: " << totalTokens << std::endl;
    std::cout << "Avg generated tokens: " << (successfulRequests > 0 ? totalTokens / successfulRequests : 0.0) << std::endl;
    std::cout << "========================================\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path> [num_requests] [concurrency] [max_tokens] [n_seq_max]" << std::endl;
        return 1;
    }
    
    std::string modelPath = argv[1];
    size_t numRequests = argc > 2 ? std::stoul(argv[2]) : 160;
    size_t concurrency = argc > 3 ? std::stoul(argv[3]) : 5;
    size_t maxTokens = argc > 4 ? std::stoul(argv[4]) : 50;
    int32_t nSeqMax = argc > 5 ? std::stoi(argv[5]) : 32;
    
    std::cout << "llama.cpp Direct Performance Test\n";
    std::cout << "Model: " << modelPath << std::endl;
    std::cout << "Requests: " << numRequests << std::endl;
    std::cout << "Concurrency: " << concurrency << std::endl;
    std::cout << "Max tokens: " << maxTokens << std::endl;
    std::cout << "n_seq_max: " << nSeqMax << std::endl;
    std::cout << std::endl;
    
    // Load prompts
    std::vector<std::string> prompts = {
        "Hello, how are you?",
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a short story about a robot.",
        "What are the benefits of exercise?",
    };
    
    try {
        LlamaCppDirectBenchmark benchmark(modelPath, concurrency, nSeqMax);
        runConcurrentTest(benchmark, prompts, numRequests, concurrency, maxTokens);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
