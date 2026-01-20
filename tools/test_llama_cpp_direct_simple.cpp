/**
 * @file test_llama_cpp_direct_simple.cpp
 * @brief 直接测试 llama.cpp 的性能上限（简化版）
 * 
 * 基于 llama.cpp 的 batched 示例，测试直接使用 llama.cpp 的并发性能
 */

#include "llama.h"
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>
#include <mutex>
#include <string>
#include <random>
#include <cstring>

struct TestResult {
    size_t requestId;
    bool success;
    size_t tokensGenerated;
    double responseTime;
    std::string error;
};

// 简化的 tokenize 函数
std::vector<llama_token> tokenize(const llama_model* model, const std::string& text, bool addBos) {
    const llama_vocab* vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens;
    
    // 第一次调用：获取需要的 token 数量
    int nTokens = llama_tokenize(vocab, text.c_str(), text.length(), nullptr, 0, addBos, false);
    if (nTokens < 0) {
        // 如果返回负数，可能是错误，尝试分配一个合理的缓冲区大小
        nTokens = text.length() + 10;  // 粗略估计
    }
    
    // 分配缓冲区
    tokens.resize(nTokens);
    
    // 第二次调用：实际 tokenize
    int actualTokens = llama_tokenize(vocab, text.c_str(), text.length(), tokens.data(), tokens.size(), addBos, false);
    if (actualTokens < 0) {
        tokens.clear();
        return tokens;
    }
    
    // 调整大小到实际 token 数量
    tokens.resize(actualTokens);
    return tokens;
}

// 简化的生成函数
TestResult generateTokens(
    llama_context* ctx,
    const llama_model* model,
    size_t requestId,
    const std::string& prompt,
    size_t maxTokens = 50,
    llama_seq_id seqId = 0
) {
    TestResult result;
    result.requestId = requestId;
    result.success = false;
    result.tokensGenerated = 0;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    try {
        // Tokenize prompt
        std::vector<llama_token> tokens = tokenize(model, prompt, true);
        if (tokens.empty()) {
            result.error = "Failed to tokenize prompt";
            return result;
        }
        
        // Prepare initial batch
        llama_batch batch = llama_batch_init(static_cast<int32_t>(tokens.size()), 0, 1);
        for (size_t i = 0; i < tokens.size(); ++i) {
            batch.token[i] = tokens[i];
            batch.pos[i] = static_cast<llama_pos>(i);
            batch.n_seq_id[i] = 1;
            batch.seq_id[i] = new llama_seq_id[1];
            batch.seq_id[i][0] = seqId;
            batch.logits[i] = (i == tokens.size() - 1);
        }
        batch.n_tokens = static_cast<int32_t>(tokens.size());
        
        // Decode prompt
        int decodeResult = llama_decode(ctx, batch);
        if (decodeResult != 0) {
            result.error = "Failed to decode prompt (code: " + std::to_string(decodeResult) + ")";
            for (int32_t i = 0; i < batch.n_tokens; ++i) {
                delete[] batch.seq_id[i];
            }
            llama_batch_free(batch);
            return result;
        }
        
        // Generate tokens
        size_t nCur = tokens.size();
        while (nCur < tokens.size() + maxTokens) {
            // Get logits
            float* logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
            const llama_vocab* vocab = llama_model_get_vocab(model);
            size_t vocabSize = llama_vocab_n_tokens(vocab);
            
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
            if (newTokenId == llama_vocab_eos(vocab)) {
                break;
            }
            
            // Cleanup old batch
            for (int32_t i = 0; i < batch.n_tokens; ++i) {
                delete[] batch.seq_id[i];
            }
            llama_batch_free(batch);
            
            // Prepare next batch
            batch = llama_batch_init(1, 0, 1);
            batch.token[0] = newTokenId;
            batch.pos[0] = static_cast<llama_pos>(nCur);
            batch.n_seq_id[0] = 1;
            batch.seq_id[0] = new llama_seq_id[1];
            batch.seq_id[0][0] = seqId;
            batch.logits[0] = true;
            batch.n_tokens = 1;
            
            // Decode
            int decodeResult = llama_decode(ctx, batch);
            if (decodeResult != 0) {
                result.error = "Failed to decode during generation (code: " + std::to_string(decodeResult) + ")";
                break;
            }
            
            nCur++;
            result.tokensGenerated++;
        }
        
        // Cleanup
        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            delete[] batch.seq_id[i];
        }
        llama_batch_free(batch);
        
        // Clear sequence from KV cache (optional, will be reused)
        // Note: In a real scenario, we might want to clear the sequence,
        // but for performance testing, we can skip this to avoid overhead
        
        result.success = true;
    } catch (const std::exception& e) {
        result.error = e.what();
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    result.responseTime = std::chrono::duration<double>(endTime - startTime).count();
    
    return result;
}

void runConcurrentTest(
    llama_model* model,
    const std::vector<std::string>& prompts,
    size_t numRequests,
    size_t concurrency,
    size_t maxTokens,
    int32_t nSeqMax,
    int32_t nCtx,
    int32_t nBatch,
    int32_t nThreads
) {
    std::vector<TestResult> results;
    results.reserve(numRequests);
    std::mutex resultsMutex;
    std::atomic<size_t> completedRequests(0);
    std::atomic<size_t> failedRequests(0);
    
    // Create contexts for each worker
    std::vector<llama_context*> contexts;
    for (size_t i = 0; i < concurrency; ++i) {
        llama_context_params ctxParams = llama_context_default_params();
        ctxParams.n_ctx = nCtx;
        ctxParams.n_batch = nBatch;
        ctxParams.n_threads = nThreads;
        ctxParams.n_threads_batch = nThreads;
        ctxParams.n_seq_max = nSeqMax;
        
        llama_context* ctx = llama_init_from_model(model, ctxParams);
        if (!ctx) {
            std::cerr << "Failed to create context " << i << std::endl;
            return;
        }
        contexts.push_back(ctx);
    }
    std::cout << "Created " << contexts.size() << " contexts" << std::endl;
    
    auto worker = [&](size_t workerId) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<size_t> promptDist(0, prompts.size() - 1);
        
        size_t requestsPerWorker = numRequests / concurrency;
        size_t startRequest = workerId * requestsPerWorker;
        size_t endRequest = (workerId == concurrency - 1) ? numRequests : startRequest + requestsPerWorker;
        
        llama_context* ctx = contexts[workerId];
        
        for (size_t i = startRequest; i < endRequest; ++i) {
            const std::string& prompt = prompts[promptDist(gen)];
            llama_seq_id seqId = static_cast<llama_seq_id>(i % nSeqMax);
            TestResult result = generateTokens(ctx, model, i, prompt, maxTokens, seqId);
            
            {
                std::lock_guard<std::mutex> lock(resultsMutex);
                results.push_back(result);
            }
            
            if (result.success) {
                completedRequests++;
            } else {
                failedRequests++;
                if (failedRequests <= 5) {
                    std::cerr << "Request " << i << " failed: " << result.error << std::endl;
                }
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
    
    // Cleanup contexts
    for (auto* ctx : contexts) {
        llama_free(ctx);
    }
    
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
        std::cerr << "Usage: " << argv[0] << " <model_path> [num_requests] [concurrency] [max_tokens] [n_seq_max] [n_ctx] [n_batch] [n_threads]" << std::endl;
        return 1;
    }
    
    std::string modelPath = argv[1];
    size_t numRequests = argc > 2 ? std::stoul(argv[2]) : 160;
    size_t concurrency = argc > 3 ? std::stoul(argv[3]) : 5;
    size_t maxTokens = argc > 4 ? std::stoul(argv[4]) : 50;
    int32_t nSeqMax = argc > 5 ? std::stoi(argv[5]) : 32;
    int32_t nCtx = argc > 6 ? std::stoi(argv[6]) : 2048;
    int32_t nBatch = argc > 7 ? std::stoi(argv[7]) : 512;
    int32_t nThreads = argc > 8 ? std::stoi(argv[8]) : 8;
    
    std::cout << "llama.cpp Direct Performance Test\n";
    std::cout << "Model: " << modelPath << std::endl;
    std::cout << "Requests: " << numRequests << std::endl;
    std::cout << "Concurrency: " << concurrency << std::endl;
    std::cout << "Max tokens: " << maxTokens << std::endl;
    std::cout << "n_seq_max: " << nSeqMax << std::endl;
    std::cout << "n_ctx: " << nCtx << std::endl;
    std::cout << "n_batch: " << nBatch << std::endl;
    std::cout << "n_threads: " << nThreads << std::endl;
    std::cout << std::endl;
    
    // Initialize llama.cpp
    llama_backend_init();
    
    // Load model
    llama_model_params modelParams = llama_model_default_params();
    modelParams.n_gpu_layers = 0;  // CPU only
    
    std::cout << "Loading model..." << std::endl;
    llama_model* model = llama_model_load_from_file(modelPath.c_str(), modelParams);
    if (!model) {
        std::cerr << "Failed to load model" << std::endl;
        llama_backend_free();
        return 1;
    }
    std::cout << "Model loaded successfully" << std::endl;
    
    // Load prompts
    std::vector<std::string> prompts = {
        "Hello, how are you?",
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a short story about a robot.",
        "What are the benefits of exercise?",
    };
    
    try {
        runConcurrentTest(model, prompts, numRequests, concurrency, maxTokens, 
                         nSeqMax, nCtx, nBatch, nThreads);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    
    llama_model_free(model);
    llama_backend_free();
    
    return 0;
}
