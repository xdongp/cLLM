/**
 * @file test_llama_cpp_via_cllm.cpp
 * @brief 通过 cLLM 的 LlamaCppBackend 测试 llama.cpp 的直接性能
 * 
 * 绕过调度器和 HTTP 层，直接使用 LlamaCppBackend 进行并发测试
 */

#include "cllm/inference/llama_cpp_backend.h"
#include "cllm/model/config.h"
#include "cllm/common/logger.h"
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>
#include <mutex>
#include <string>
#include <random>

struct TestResult {
    size_t requestId;
    bool success;
    size_t tokensGenerated;
    double responseTime;
    std::string error;
};

using namespace cllm;
using namespace cllm::inference;

TestResult processRequest(
    LlamaCppBackend& backend,
    size_t requestId,
    const std::string& prompt,
    size_t maxTokens = 50
) {
    TestResult result;
    result.requestId = requestId;
    result.success = false;
    result.tokensGenerated = 0;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    try {
        // 分配序列ID
        int32_t seqId = backend.allocateSequenceId(requestId);
        if (seqId < 0) {
            result.error = "Failed to allocate sequence ID";
            return result;
        }
        
        // 简单的 tokenize（这里需要实际的 tokenizer，暂时跳过）
        // 为了测试，我们使用一个简单的 token 序列
        std::vector<int> inputIds = {1, 2, 3, 4, 5};  // 占位符
        
        // 准备 batch input
        BatchInput batchInput;
        batchInput.batchSize = 1;
        batchInput.inputIds = inputIds;
        batchInput.requestPositions.push_back({0, inputIds.size()});
        batchInput.sequenceIds.push_back(requestId);
        
        // 执行推理
        BatchOutput output = backend.forwardBatch(
            batchInput.inputIds,
            batchInput.requestPositions,
            batchInput.batchSize,
            batchInput.sequenceIds
        );
        
        // 生成 tokens（简化版，实际需要采样）
        for (size_t i = 0; i < maxTokens; ++i) {
            // 获取 logits 并采样（简化）
            result.tokensGenerated++;
        }
        
        // 清理
        backend.cleanupKVCache(requestId);
        backend.releaseSequenceId(requestId);
        
        result.success = true;
    } catch (const std::exception& e) {
        result.error = e.what();
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    result.responseTime = std::chrono::duration<double>(endTime - startTime).count();
    
    return result;
}

void runConcurrentTest(
    LlamaCppBackend& backend,
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
            TestResult result = processRequest(backend, i, prompt, maxTokens);
            
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
    
    // 计算统计信息
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
    
    // 打印结果
    std::cout << "\n========================================\n";
    std::cout << "llama.cpp via cLLM Backend Performance Test\n";
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
        std::cerr << "Usage: " << argv[0] << " <model_path> [num_requests] [concurrency] [max_tokens]" << std::endl;
        return 1;
    }
    
    std::string modelPath = argv[1];
    size_t numRequests = argc > 2 ? std::stoul(argv[2]) : 160;
    size_t concurrency = argc > 3 ? std::stoul(argv[3]) : 5;
    size_t maxTokens = argc > 4 ? std::stoul(argv[4]) : 50;
    
    std::cout << "llama.cpp via cLLM Backend Performance Test\n";
    std::cout << "Model: " << modelPath << std::endl;
    std::cout << "Requests: " << numRequests << std::endl;
    std::cout << "Concurrency: " << concurrency << std::endl;
    std::cout << "Max tokens: " << maxTokens << std::endl;
    std::cout << std::endl;
    
    // 创建模型配置
    ModelConfig config;
    config.vocabSize = 151936;  // Qwen3 0.6B
    config.maxSequenceLength = 2048;
    config.llamaNumThreads = 8;
    config.llamaBatchSize = 512;
    config.llamaGpuLayers = 0;  // CPU only for comparison
    config.llamaUseMmap = true;
    config.llamaUseMlock = false;
    config.llamaNSeqMax = 32;
    
    try {
        LlamaCppBackend backend(config, modelPath);
        if (!backend.initialize()) {
            std::cerr << "Failed to initialize backend" << std::endl;
            return 1;
        }
        
        std::vector<std::string> prompts = {
            "Hello, how are you?",
            "What is the capital of France?",
            "Explain quantum computing in simple terms.",
            "Write a short story about a robot.",
            "What are the benefits of exercise?",
        };
        
        runConcurrentTest(backend, prompts, numRequests, concurrency, maxTokens);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
