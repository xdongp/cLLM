/**
 * @file direct_benchmark.cpp
 * @brief 直接性能测试程序 - 参考 llama-bench 实现
 * 
 * 目标：绕过 Scheduler、BatchManager 等中间层，直接测试底层 API 性能
 * 参考：llama-bench 的简单直接实现方式
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <algorithm>

#include "cllm/common/config.h"
#include "cllm/inference/llama_cpp_backend.h"
#include "cllm/model/config.h"
#include "cllm/common/logger.h"
#include <numeric>

using namespace cllm;
using namespace cllm::inference;

// 工具函数
static uint64_t get_time_ns() {
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::nanoseconds(clock::now().time_since_epoch()).count();
}

static double get_time_sec() {
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

// 测试参数
struct BenchParams {
    std::string model_path;
    int n_prompt = 32;      // prompt tokens
    int n_gen = 50;         // generation tokens per request
    int n_requests = 40;    // total requests
    int n_concurrent = 5;   // concurrent requests
    int n_reps = 1;         // repetitions
    int n_batch = 512;      // llama.cpp batch size
    int n_ubatch = 512;     // llama.cpp ubatch size
    int n_seq_max = 64;     // max sequences
    int n_gpu_layers = 99;  // GPU layers
    bool verbose = false;
};

// 参考 llama-bench 的 test_gen 函数
static bool test_gen_direct(
    LlamaCppBackend& backend,
    std::mutex& backendMutex,  // 🔥 保护 llama_decode 调用
    int n_gen,
    size_t requestId,
    int32_t /*seqId*/,  // 未使用，forwardBatch 会自动分配
    std::vector<int>& promptTokens,
    std::vector<int>& generatedTokens
) {
    generatedTokens.clear();
    generatedTokens.reserve(n_gen);
    
    // 🔥 不要预先分配序列ID，让 forwardBatch 在首次调用时自动分配
    // 这样可以确保新请求从位置0开始，已存在的请求从正确的位置继续
    
    // 初始化：如果有 prompt，先处理 prompt
    if (!promptTokens.empty()) {
        std::vector<int> flatInputIds = promptTokens;
        std::vector<std::pair<size_t, size_t>> requestPositions = {{0, promptTokens.size()}};
        std::vector<size_t> sequenceIds = {requestId};
        
        Tensor logits;
        {
            std::lock_guard<std::mutex> lock(backendMutex);
            logits = backend.forwardBatch(flatInputIds, requestPositions, 1, sequenceIds);
        }
        
        // 采样第一个 token（简单随机采样）
        const float* logitsPtr = logits.data() + (promptTokens.size() - 1) * logits.shape()[1];
        size_t vocabSize = logits.shape()[1];
        int nextToken = std::rand() % vocabSize;
        generatedTokens.push_back(nextToken);
    } else {
        // 如果没有 prompt，生成一个初始 token
        int32_t bosToken = 151643;  // Qwen3 BOS token
        std::vector<int> flatInputIds = {bosToken};
        std::vector<std::pair<size_t, size_t>> requestPositions = {{0, 1}};
        std::vector<size_t> sequenceIds = {requestId};
        
        Tensor logits;
        {
            std::lock_guard<std::mutex> lock(backendMutex);
            logits = backend.forwardBatch(flatInputIds, requestPositions, 1, sequenceIds);
        }
        
        const float* logitsPtr = logits.data();
        size_t vocabSize = logits.shape()[1];
        int nextToken = std::rand() % vocabSize;
        generatedTokens.push_back(nextToken);
    }
    
    // 生成 tokens
    for (int i = generatedTokens.size(); i < n_gen; ++i) {
        // 🔥 参考 llama-bench: 只送最后一个 token（增量生成）
        std::vector<int> flatInputIds = {generatedTokens.back()};
        std::vector<std::pair<size_t, size_t>> requestPositions = {{0, 1}};
        std::vector<size_t> sequenceIds = {requestId};
        
        Tensor logits;
        {
            std::lock_guard<std::mutex> lock(backendMutex);
            logits = backend.forwardBatch(flatInputIds, requestPositions, 1, sequenceIds);
        }
        
        // 采样下一个 token（简单随机采样）
        const float* logitsPtr = logits.data();
        size_t vocabSize = logits.shape()[1];
        int nextToken = std::rand() % vocabSize;
        generatedTokens.push_back(nextToken);
    }
    
    // 🔥 释放序列ID
    try {
        backend.releaseSequenceId(requestId);
    } catch (...) {
        // 忽略释放错误
    }
    
    return true;
}

// 并发测试：参考 llama-bench 的简单直接方式
static void run_concurrent_test(const BenchParams& params) {
    std::cout << "=== Direct Benchmark Test (参考 llama-bench) ===" << std::endl;
    std::cout << "Model: " << params.model_path << std::endl;
    std::cout << "Requests: " << params.n_requests << std::endl;
    std::cout << "Concurrent: " << params.n_concurrent << std::endl;
    std::cout << "Prompt tokens: " << params.n_prompt << std::endl;
    std::cout << "Gen tokens per request: " << params.n_gen << std::endl;
    std::cout << std::endl;
    
    // 🔥 初始化 Config（LlamaCppBackend 需要从 Config 读取 n_seq_max 和 n_ubatch）
    try {
        Config::instance().load("config/config.yaml");
    } catch (...) {
        // 如果配置文件不存在，使用默认配置
        std::cerr << "Warning: Failed to load config file, using defaults" << std::endl;
    }
    
    // 创建后端配置
    ModelConfig modelConfig;
    modelConfig.vocabSize = 151936;
    modelConfig.maxSequenceLength = 2048;
    modelConfig.llamaBatchSize = params.n_batch;
    modelConfig.llamaGpuLayers = params.n_gpu_layers;
    
    LlamaCppBackend backend(modelConfig, params.model_path);
    
    if (!backend.initialize()) {
        std::cerr << "Failed to initialize backend" << std::endl;
        return;
    }
    
    std::cout << "Backend initialized successfully" << std::endl;
    
    // 准备 prompt tokens（随机）
    std::vector<int> promptTokens;
    promptTokens.reserve(params.n_prompt);
    for (int i = 0; i < params.n_prompt; ++i) {
        promptTokens.push_back(std::rand() % 1000);  // 简单随机 tokens
    }
    
    // 并发请求队列
    std::queue<size_t> requestQueue;
    std::mutex queueMutex;
    std::condition_variable queueCondition;
    std::atomic<size_t> completedRequests{0};
    std::atomic<size_t> totalTokens{0};
    std::vector<double> requestTimes;
    std::mutex timesMutex;
    
    // 🔥 llama_decode 不是线程安全的，需要互斥锁保护
    std::mutex backendMutex;
    
    // 初始化请求队列
    for (size_t i = 0; i < params.n_requests; ++i) {
        requestQueue.push(i);
    }
    
    // 工作线程函数
    auto worker = [&](int workerId) {
        while (true) {
            size_t requestId;
            {
                std::unique_lock<std::mutex> lock(queueMutex);
                if (requestQueue.empty()) {
                    break;
                }
                requestId = requestQueue.front();
                requestQueue.pop();
            }
            
            // 测试单个请求
            double startTime = get_time_sec();
            std::vector<int> generatedTokens;
            
            bool success = test_gen_direct(
                backend,
                backendMutex,
                params.n_gen,
                requestId,
                0,  // seqId 会在 forwardBatch 中自动分配
                promptTokens,
                generatedTokens
            );
            
            double endTime = get_time_sec();
            double elapsed = endTime - startTime;
            
            if (success && !generatedTokens.empty()) {
                size_t tokens = generatedTokens.size();
                totalTokens += tokens;
                completedRequests++;
                
                {
                    std::lock_guard<std::mutex> lock(timesMutex);
                    requestTimes.push_back(elapsed);
                }
                
                if (params.verbose) {
                    std::cout << "Worker " << workerId << ": Request " << requestId 
                              << " completed in " << elapsed << "s, generated " << tokens << " tokens" << std::endl;
                }
            } else {
                std::cerr << "Worker " << workerId << ": Request " << requestId << " failed!" << std::endl;
                // test_gen_direct 已经在函数内部释放了序列ID（无论成功或失败）
            }
        }
    };
    
    // 启动并发测试
    double testStart = get_time_sec();
    
    std::vector<std::thread> workers;
    for (int i = 0; i < params.n_concurrent; ++i) {
        workers.emplace_back(worker, i);
    }
    
    // 等待所有工作线程完成
    for (auto& w : workers) {
        w.join();
    }
    
    double testEnd = get_time_sec();
    double totalTime = testEnd - testStart;
    
    // 计算统计信息
    size_t successful = completedRequests.load();
    size_t totalGenTokens = totalTokens.load();
    
    std::sort(requestTimes.begin(), requestTimes.end());
    double avgTime = requestTimes.empty() ? 0.0 : 
                     std::accumulate(requestTimes.begin(), requestTimes.end(), 0.0) / requestTimes.size();
    double p50Time = requestTimes.empty() ? 0.0 : requestTimes[requestTimes.size() / 2];
    double p99Time = requestTimes.empty() ? 0.0 : requestTimes[static_cast<size_t>(requestTimes.size() * 0.99)];
    double minTime = requestTimes.empty() ? 0.0 : requestTimes.front();
    double maxTime = requestTimes.empty() ? 0.0 : requestTimes.back();
    
    double throughput = totalTime > 0 ? (totalGenTokens / totalTime) : 0.0;
    
    // 输出结果
    std::cout << std::endl;
    std::cout << "=== Test Results ===" << std::endl;
    std::cout << "Total time: " << totalTime << "s" << std::endl;
    std::cout << "Successful requests: " << successful << "/" << params.n_requests << std::endl;
    std::cout << "Total generated tokens: " << totalGenTokens << std::endl;
    std::cout << "Average throughput: " << throughput << " tokens/sec" << std::endl;
    std::cout << std::endl;
    std::cout << "Response time stats:" << std::endl;
    std::cout << "  Min: " << minTime << "s" << std::endl;
    std::cout << "  Max: " << maxTime << "s" << std::endl;
    std::cout << "  Avg: " << avgTime << "s" << std::endl;
    std::cout << "  P50: " << p50Time << "s" << std::endl;
    std::cout << "  P99: " << p99Time << "s" << std::endl;
    std::cout << std::endl;
    
    // 目标检查
    double target = 80.0;
    if (throughput >= target) {
        std::cout << "✅ 已达到第一阶段目标: " << throughput << " >= " << target << " tokens/sec" << std::endl;
    } else {
        std::cout << "❌ 未达到第一阶段目标: " << throughput << " < " << target << " tokens/sec" << std::endl;
        std::cout << "   差距: " << (target - throughput) << " tokens/sec (" 
                  << ((target - throughput) / target * 100) << "%)" << std::endl;
    }
}

// 解析命令行参数
static BenchParams parse_args(int argc, char** argv) {
    BenchParams params;
    
    // 默认值
    params.model_path = "/Users/dannypan/PycharmProjects/xllm/cpp/cLLM/model/Qwen/qwen3-0.6b-q8_0.gguf";
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            params.model_path = argv[++i];
        } else if (strcmp(argv[i], "--n-prompt") == 0 && i + 1 < argc) {
            params.n_prompt = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--n-gen") == 0 && i + 1 < argc) {
            params.n_gen = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--n-requests") == 0 && i + 1 < argc) {
            params.n_requests = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--n-concurrent") == 0 && i + 1 < argc) {
            params.n_concurrent = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--n-batch") == 0 && i + 1 < argc) {
            params.n_batch = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--n-ubatch") == 0 && i + 1 < argc) {
            params.n_ubatch = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--n-seq-max") == 0 && i + 1 < argc) {
            params.n_seq_max = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--n-gpu-layers") == 0 && i + 1 < argc) {
            params.n_gpu_layers = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--verbose") == 0 || strcmp(argv[i], "-v") == 0) {
            params.verbose = true;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --model <path>           Model path (default: " << params.model_path << ")" << std::endl;
            std::cout << "  --n-prompt <n>           Prompt tokens (default: " << params.n_prompt << ")" << std::endl;
            std::cout << "  --n-gen <n>              Generation tokens per request (default: " << params.n_gen << ")" << std::endl;
            std::cout << "  --n-requests <n>         Total requests (default: " << params.n_requests << ")" << std::endl;
            std::cout << "  --n-concurrent <n>       Concurrent requests (default: " << params.n_concurrent << ")" << std::endl;
            std::cout << "  --n-batch <n>            Batch size (default: " << params.n_batch << ")" << std::endl;
            std::cout << "  --n-ubatch <n>           Ubatch size (default: " << params.n_ubatch << ")" << std::endl;
            std::cout << "  --n-seq-max <n>          Max sequences (default: " << params.n_seq_max << ")" << std::endl;
            std::cout << "  --n-gpu-layers <n>       GPU layers (default: " << params.n_gpu_layers << ")" << std::endl;
            std::cout << "  --verbose, -v            Verbose output" << std::endl;
            std::cout << "  --help, -h               Show this help" << std::endl;
            exit(0);
        }
    }
    
    return params;
}

int main(int argc, char** argv) {
    BenchParams params = parse_args(argc, argv);
    
    // 运行测试
    run_concurrent_test(params);
    
    return 0;
}
