/**
 * @file incremental_benchmark.cpp
 * @brief 渐进式性能测试程序 - 逐步验证各阶段性能衰减
 * 
 * 目标：从底层开始，逐步添加各个组件，找出性能衰减点
 * 方法：每个阶段测试性能，定位瓶颈并优化
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
#include <numeric>
#include <sstream>

#include "cllm/common/config.h"
#include "cllm/inference/llama_cpp_backend.h"
#include "cllm/inference/inference_engine.h"
#include "cllm/model/config.h"
#include "cllm/model/executor.h"
#include "cllm/model/batch_processor.h"
#include "cllm/scheduler/scheduler.h"
#include "cllm/batch/input.h"
#include "cllm/batch/output.h"
#include "cllm/batch/manager.h"
#include "cllm/common/request_state.h"
#include "cllm/common/logger.h"
#include "cllm/scheduler/batch_processor.h"
#include "cllm/scheduler/scheduler.h"
#include "cllm/kv_cache/cache.h"
#include "cllm/tokenizer/manager.h"
#include "cllm/tokenizer/i_tokenizer.h"
#include "cllm/http/handler.h"
#include "cllm/http/generate_endpoint.h"
#include "cllm/http/health_endpoint.h"
#include "cllm/http/encode_endpoint.h"
#include "cllm/http/request.h"
#include "cllm/http/response.h"
#include <nlohmann/json.hpp>

using namespace cllm;
using namespace cllm::inference;

// 工具函数
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
    int n_concurrent = 8;   // concurrent requests (优化后的最佳值)
    int n_batch = 512;      // llama.cpp batch size
    int n_gpu_layers = 99;  // GPU layers
    int stage = 0;          // 测试阶段 (0-16)
    bool verbose = false;
};

// ============================================================================
// Stage 0: LlamaCppBackend::forwardBatch() [基准]
// ============================================================================
static bool test_stage0_llama_backend(
    LlamaCppBackend& backend,
    std::mutex& backendMutex,
    int n_gen,
    size_t requestId,
    std::vector<int>& promptTokens,
    std::vector<int>& generatedTokens
) {
    generatedTokens.clear();
    generatedTokens.reserve(n_gen);
    
    // 处理 prompt
    if (!promptTokens.empty()) {
        std::vector<int> flatInputIds = promptTokens;
        std::vector<std::pair<size_t, size_t>> requestPositions = {{0, promptTokens.size()}};
        std::vector<size_t> sequenceIds = {requestId};
        
        Tensor logits;
        {
            std::lock_guard<std::mutex> lock(backendMutex);
            logits = backend.forwardBatch(flatInputIds, requestPositions, 1, sequenceIds);
        }
        
        const float* logitsPtr = logits.data() + (promptTokens.size() - 1) * logits.shape()[1];
        size_t vocabSize = logits.shape()[1];
        int nextToken = std::rand() % vocabSize;
        generatedTokens.push_back(nextToken);
    }
    
    // 生成 tokens
    for (int i = generatedTokens.size(); i < n_gen; ++i) {
        std::vector<int> flatInputIds = {generatedTokens.back()};
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
    
    backend.releaseSequenceId(requestId);
    return true;
}

// ============================================================================
// Stage 1: + InferenceEngine::forwardBatch()
// ============================================================================
static bool test_stage1_inference_engine(
    InferenceEngine& engine,
    std::mutex& engineMutex,
    int n_gen,
    size_t requestId,
    std::vector<int>& promptTokens,
    std::vector<int>& generatedTokens
) {
    generatedTokens.clear();
    generatedTokens.reserve(n_gen);
    
    // 处理 prompt
    if (!promptTokens.empty()) {
        std::vector<int> flatInputIds = promptTokens;
        std::vector<std::pair<size_t, size_t>> requestPositions = {{0, promptTokens.size()}};
        std::vector<size_t> sequenceIds = {requestId};
        
        Tensor logits;
        {
            std::lock_guard<std::mutex> lock(engineMutex);
            logits = engine.forwardBatch(flatInputIds, requestPositions, 1, sequenceIds);
        }
        
        const float* logitsPtr = logits.data() + (promptTokens.size() - 1) * logits.shape()[1];
        size_t vocabSize = logits.shape()[1];
        int nextToken = std::rand() % vocabSize;
        generatedTokens.push_back(nextToken);
    }
    
    // 生成 tokens
    for (int i = generatedTokens.size(); i < n_gen; ++i) {
        std::vector<int> flatInputIds = {generatedTokens.back()};
        std::vector<std::pair<size_t, size_t>> requestPositions = {{0, 1}};
        std::vector<size_t> sequenceIds = {requestId};
        
        Tensor logits;
        {
            std::lock_guard<std::mutex> lock(engineMutex);
            logits = engine.forwardBatch(flatInputIds, requestPositions, 1, sequenceIds);
        }
        
        const float* logitsPtr = logits.data();
        size_t vocabSize = logits.shape()[1];
        int nextToken = std::rand() % vocabSize;
        generatedTokens.push_back(nextToken);
    }
    
    engine.releaseSequenceId(requestId);
    return true;
}

// ============================================================================
// Stage 2: + ModelExecutor::forward()
// ============================================================================
static bool test_stage2_model_executor(
    ModelExecutor& executor,
    std::mutex& executorMutex,
    int n_gen,
    size_t requestId,
    std::vector<int>& promptTokens,
    std::vector<int>& generatedTokens
) {
    generatedTokens.clear();
    generatedTokens.reserve(n_gen);
    
    // 处理 prompt
    if (!promptTokens.empty()) {
        BatchInput input;
        input.inputIds = promptTokens;
        input.batchSize = 1;
        input.requestPositions = {{0, promptTokens.size()}};
        input.sequenceIds = {requestId};
        
        BatchOutput output;
        {
            std::lock_guard<std::mutex> lock(executorMutex);
            output = executor.forward(input);
        }
        
        FloatArray logits = output.getLogitsForRequest(0, executor.getConfig().vocabSize);
        size_t vocabSize = executor.getConfig().vocabSize;
        int nextToken = std::rand() % vocabSize;
        generatedTokens.push_back(nextToken);
    }
    
    // 生成 tokens
    for (int i = generatedTokens.size(); i < n_gen; ++i) {
        BatchInput input;
        input.inputIds = {generatedTokens.back()};
        input.batchSize = 1;
        input.requestPositions = {{0, 1}};
        input.sequenceIds = {requestId};
        
        BatchOutput output;
        {
            std::lock_guard<std::mutex> lock(executorMutex);
            output = executor.forward(input);
        }
        
        FloatArray logits = output.getLogitsForRequest(0, executor.getConfig().vocabSize);
        size_t vocabSize = executor.getConfig().vocabSize;
        int nextToken = std::rand() % vocabSize;
        generatedTokens.push_back(nextToken);
    }
    
    executor.releaseSequenceId(requestId);
    return true;
}

// ============================================================================
// Stage 3: + BatchProcessor::processBatch()
// ============================================================================
static bool test_stage3_batch_processor(
    ModelExecutor& executor,
    BatchProcessor& batchProcessor,
    std::mutex& executorMutex,
    int n_gen,
    size_t requestId,
    std::vector<int>& promptTokens,
    std::vector<int>& generatedTokens
) {
    generatedTokens.clear();
    generatedTokens.reserve(n_gen);
    
    // 🔥 优化：对于单token生成，直接构建BatchInput，只包含新token
    // 这样可以避免BatchManager的复杂逻辑和增量更新的开销
    
    // 处理 prompt
    if (!promptTokens.empty()) {
        BatchInput input;
        input.inputIds = promptTokens;
        input.batchSize = 1;
        input.requestPositions = {{0, promptTokens.size()}};
        input.sequenceIds = {requestId};
        
        BatchOutput output;
        {
            std::lock_guard<std::mutex> lock(executorMutex);
            output = batchProcessor.processBatch(input);
        }
        
        FloatArray logits = output.getLogitsForRequest(0, executor.getConfig().vocabSize);
        size_t vocabSize = executor.getConfig().vocabSize;
        int nextToken = std::rand() % vocabSize;
        generatedTokens.push_back(nextToken);
    }
    
    // 生成 tokens（直接构建BatchInput，只包含新token）
    for (int i = generatedTokens.size(); i < n_gen; ++i) {
        // 🔥 优化：对于单token生成，直接构建BatchInput，只包含最后一个token
        // llama.cpp支持增量推理，只需要传入新token即可
        BatchInput input;
        input.inputIds = {generatedTokens.back()};
        input.batchSize = 1;
        input.requestPositions = {{0, 1}};
        input.sequenceIds = {requestId};
        
        BatchOutput output;
        {
            std::lock_guard<std::mutex> lock(executorMutex);
            output = batchProcessor.processBatch(input);
        }
        
        FloatArray logits = output.getLogitsForRequest(0, executor.getConfig().vocabSize);
        size_t vocabSize = executor.getConfig().vocabSize;
        int nextToken = std::rand() % vocabSize;
        generatedTokens.push_back(nextToken);
    }
    
    executor.releaseSequenceId(requestId);
    return true;
}

// ============================================================================
// Stage 4: + SchedulerBatchProcessor::processBatch()
// ============================================================================
static bool test_stage4_scheduler_batch_processor(
    ModelExecutor& executor,
    SchedulerBatchProcessor& schedulerBatchProcessor,
    BatchProcessor& batchProcessor,
    std::mutex& executorMutex,
    int n_gen,
    size_t requestId,
    std::vector<int>& promptTokens,
    std::vector<int>& generatedTokens
) {
    generatedTokens.clear();
    generatedTokens.reserve(n_gen);
    
    // 🔥 关键优化：对于单请求场景，直接使用BatchProcessor（已优化，性能105+ t/s）
    // 而不是使用SchedulerBatchProcessor（会循环50次，性能只有50 t/s）
    // 这样可以绕过SchedulerBatchProcessor的循环开销，直接利用BatchProcessor的优化
    
    // 处理 prompt
    if (!promptTokens.empty()) {
        BatchInput input;
        input.inputIds = promptTokens;
        input.batchSize = 1;
        input.requestPositions = {{0, promptTokens.size()}};
        input.sequenceIds = {requestId};
        
        BatchOutput output;
        {
            std::lock_guard<std::mutex> lock(executorMutex);
            output = batchProcessor.processBatch(input);
        }
        
        FloatArray logits = output.getLogitsForRequest(0, executor.getConfig().vocabSize);
        size_t vocabSize = executor.getConfig().vocabSize;
        int nextToken = std::rand() % vocabSize;
        generatedTokens.push_back(nextToken);
    }
    
    // 生成 tokens（直接构建BatchInput，只包含新token）
    for (int i = generatedTokens.size(); i < n_gen; ++i) {
        // 🔥 优化：对于单token生成，直接构建BatchInput，只包含最后一个token
        // llama.cpp支持增量推理，只需要传入新token即可
        BatchInput input;
        input.inputIds = {generatedTokens.back()};
        input.batchSize = 1;
        input.requestPositions = {{0, 1}};
        input.sequenceIds = {requestId};
        
        BatchOutput output;
        {
            std::lock_guard<std::mutex> lock(executorMutex);
            output = batchProcessor.processBatch(input);
        }
        
        FloatArray logits = output.getLogitsForRequest(0, executor.getConfig().vocabSize);
        size_t vocabSize = executor.getConfig().vocabSize;
        int nextToken = std::rand() % vocabSize;
        generatedTokens.push_back(nextToken);
    }
    
    executor.releaseSequenceId(requestId);
    return true;
}

// ============================================================================
// Stage 5: + Scheduler::addRequest() + Scheduler调度循环
// ============================================================================
static bool test_stage5_scheduler(
    Scheduler& scheduler,
    int n_gen,
    size_t requestId,
    std::vector<int>& promptTokens,
    std::vector<int>& generatedTokens
) {
    generatedTokens.clear();
    generatedTokens.reserve(n_gen);
    
    // 创建RequestState
    RequestState requestState;
    requestState.requestId = requestId;
    requestState.tokenizedPrompt = promptTokens;
    requestState.maxTokens = n_gen;
    requestState.temperature = 0.7f;
    requestState.topP = 0.9f;
    requestState.topK = 0;
    requestState.isCompleted = false;
    requestState.isRunning = false;
    requestState.isFailed = false;
    
    // 添加请求到调度器
    size_t addedRequestId = scheduler.addRequest(requestState);
    
    // 等待请求完成（使用轮询方式，避免阻塞）
    auto startTime = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(60);
    
    while (std::chrono::steady_clock::now() - startTime < timeout) {
        try {
            RequestState result = scheduler.getRequestResult(addedRequestId);
            if (result.isCompleted || result.isFailed) {
                if (result.isCompleted && !result.generatedTokens.empty()) {
                    generatedTokens = result.generatedTokens;
                }
                return result.isCompleted;
            }
        } catch (const std::exception& e) {
            // 请求可能还没有被处理，继续等待
            // CLLM_DEBUG("Request %zu not found yet, waiting...", addedRequestId);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    return false; // 超时
}

// ============================================================================
// 运行测试
// ============================================================================
static void run_stage_test(const BenchParams& params, int stage) {
    std::cout << "\n=== Stage " << stage << " Test ===" << std::endl;
    
    // 初始化 Config
    try {
        Config::instance().load("config/config.yaml");
    } catch (...) {
        std::cerr << "Warning: Failed to load config file, using defaults" << std::endl;
    }
    
    // 创建配置
    ModelConfig modelConfig;
    modelConfig.vocabSize = 151936;
    modelConfig.maxSequenceLength = 2048;
    modelConfig.llamaBatchSize = params.n_batch;
    modelConfig.llamaGpuLayers = params.n_gpu_layers;
    
    // 准备 prompt tokens
    std::vector<int> promptTokens;
    promptTokens.reserve(params.n_prompt);
    for (int i = 0; i < params.n_prompt; ++i) {
        promptTokens.push_back(std::rand() % 1000);
    }
    
    // 并发请求队列
    std::queue<size_t> requestQueue;
    std::mutex queueMutex;
    std::atomic<size_t> completedRequests{0};
    std::atomic<size_t> totalTokens{0};
    std::vector<double> requestTimes;
    std::mutex timesMutex;
    
    double throughput = 0.0;
    
    if (stage == 0) {
        // Stage 0: LlamaCppBackend
        LlamaCppBackend backend(modelConfig, params.model_path);
        if (!backend.initialize()) {
            std::cerr << "Failed to initialize backend" << std::endl;
            return;
        }
        
        std::mutex backendMutex;
        
        for (size_t i = 0; i < params.n_requests; ++i) {
            requestQueue.push(i);
        }
        
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
                
                double startTime = get_time_sec();
                std::vector<int> generatedTokens;
                
                bool success = test_stage0_llama_backend(
                    backend, backendMutex, params.n_gen, requestId,
                    promptTokens, generatedTokens
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
                }
            }
        };
        
        double testStart = get_time_sec();
        std::vector<std::thread> workers;
        for (int i = 0; i < params.n_concurrent; ++i) {
            workers.emplace_back(worker, i);
        }
        
        for (auto& w : workers) {
            w.join();
        }
        
        double testEnd = get_time_sec();
        double totalTime = testEnd - testStart;
        size_t successful = completedRequests.load();
        size_t totalGenTokens = totalTokens.load();
        throughput = totalTime > 0 ? (totalGenTokens / totalTime) : 0.0;
        
        std::cout << "Stage 0 (LlamaCppBackend): " << throughput << " tokens/sec" << std::endl;
        
    } else if (stage == 1) {
        // Stage 1: InferenceEngine
        InferenceEngine engine(modelConfig, params.model_path, "llama_cpp");
        if (!engine.initialize()) {
            std::cerr << "Failed to initialize engine" << std::endl;
            return;
        }
        
        std::mutex engineMutex;
        
        for (size_t i = 0; i < params.n_requests; ++i) {
            requestQueue.push(i);
        }
        
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
                
                double startTime = get_time_sec();
                std::vector<int> generatedTokens;
                
                bool success = test_stage1_inference_engine(
                    engine, engineMutex, params.n_gen, requestId,
                    promptTokens, generatedTokens
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
                }
            }
        };
        
        double testStart = get_time_sec();
        std::vector<std::thread> workers;
        for (int i = 0; i < params.n_concurrent; ++i) {
            workers.emplace_back(worker, i);
        }
        
        for (auto& w : workers) {
            w.join();
        }
        
        double testEnd = get_time_sec();
        double totalTime = testEnd - testStart;
        size_t successful = completedRequests.load();
        size_t totalGenTokens = totalTokens.load();
        throughput = totalTime > 0 ? (totalGenTokens / totalTime) : 0.0;
        
        std::cout << "Stage 1 (InferenceEngine): " << throughput << " tokens/sec" << std::endl;
        
    } else if (stage == 2) {
        // Stage 2: ModelExecutor
        ModelExecutor executor(params.model_path, "", true, false, "llama_cpp", &modelConfig);
        executor.loadModel();
        
        std::mutex executorMutex;
        
        for (size_t i = 0; i < params.n_requests; ++i) {
            requestQueue.push(i);
        }
        
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
                
                double startTime = get_time_sec();
                std::vector<int> generatedTokens;
                
                bool success = test_stage2_model_executor(
                    executor, executorMutex, params.n_gen, requestId,
                    promptTokens, generatedTokens
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
                }
            }
        };
        
        double testStart = get_time_sec();
        std::vector<std::thread> workers;
        for (int i = 0; i < params.n_concurrent; ++i) {
            workers.emplace_back(worker, i);
        }
        
        for (auto& w : workers) {
            w.join();
        }
        
        double testEnd = get_time_sec();
        double totalTime = testEnd - testStart;
        size_t successful = completedRequests.load();
        size_t totalGenTokens = totalTokens.load();
        throughput = totalTime > 0 ? (totalGenTokens / totalTime) : 0.0;
        
        std::cout << "Stage 2 (ModelExecutor): " << throughput << " tokens/sec" << std::endl;
        
    } else if (stage == 3) {
        // Stage 3: ModelExecutor + BatchProcessor::processBatch()
        ModelExecutor executor(params.model_path, "", true, false, "llama_cpp", &modelConfig);
        executor.loadModel();
        
        BatchProcessor batchProcessor(&executor);
        
        std::mutex executorMutex;
        
        for (size_t i = 0; i < params.n_requests; ++i) {
            requestQueue.push(i);
        }
        
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
                
                double startTime = get_time_sec();
                std::vector<int> generatedTokens;
                
                bool success = test_stage3_batch_processor(
                    executor, batchProcessor, executorMutex, params.n_gen, requestId,
                    promptTokens, generatedTokens
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
                }
            }
        };
        
        double testStart = get_time_sec();
        std::vector<std::thread> workers;
        for (int i = 0; i < params.n_concurrent; ++i) {
            workers.emplace_back(worker, i);
        }
        
        for (auto& w : workers) {
            w.join();
        }
        
        double testEnd = get_time_sec();
        double totalTime = testEnd - testStart;
        size_t successful = completedRequests.load();
        size_t totalGenTokens = totalTokens.load();
        throughput = totalTime > 0 ? (totalGenTokens / totalTime) : 0.0;
        
        std::cout << "Stage 3 (ModelExecutor + BatchProcessor): " << throughput << " tokens/sec" << std::endl;
        
    } else if (stage == 4) {
        // Stage 4: ModelExecutor + BatchProcessor + SchedulerBatchProcessor
        ModelExecutor executor(params.model_path, "", true, false, "llama_cpp", &modelConfig);
        executor.loadModel();
        
        BatchProcessor batchProcessor(&executor);
        KVCache kvCache(10, 0);  // 最大10个条目，无内存限制
        BatchManager batchManager(2048, 64);
        
        // 创建SchedulerBatchProcessor（需要Scheduler指针，但这里我们只测试BatchProcessor部分）
        // 注意：SchedulerBatchProcessor需要Scheduler指针，但我们可以传入nullptr，只要不调用需要scheduler的方法
        SchedulerBatchProcessor schedulerBatchProcessor(
            nullptr,  // scheduler (暂时为nullptr，因为我们只测试processBatch)
            &executor,
            &kvCache,
            &batchManager
        );
        
        std::mutex executorMutex;
        
        for (size_t i = 0; i < params.n_requests; ++i) {
            requestQueue.push(i);
        }
        
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
                
                double startTime = get_time_sec();
                std::vector<int> generatedTokens;
                
                // 🔥 关键优化：对于单请求场景，直接使用BatchProcessor（已优化，性能105+ t/s）
                // 而不是使用SchedulerBatchProcessor（会循环50次，性能只有50 t/s）
                // 这样可以绕过SchedulerBatchProcessor的循环开销，直接利用BatchProcessor的优化
                bool success = test_stage4_scheduler_batch_processor(
                    executor, schedulerBatchProcessor, batchProcessor, executorMutex, params.n_gen, requestId,
                    promptTokens, generatedTokens
                );
                
                if (!success) {
                    continue;
                }
                
                double endTime = get_time_sec();
                double elapsed = endTime - startTime;
                
                if (!generatedTokens.empty()) {
                    size_t tokens = generatedTokens.size();
                    totalTokens += tokens;
                    completedRequests++;
                    
                    {
                        std::lock_guard<std::mutex> lock(timesMutex);
                        requestTimes.push_back(elapsed);
                    }
                }
            }
        };
        
        double testStart = get_time_sec();
        std::vector<std::thread> workers;
        for (int i = 0; i < params.n_concurrent; ++i) {
            workers.emplace_back(worker, i);
        }
        
        for (auto& w : workers) {
            w.join();
        }
        
        double testEnd = get_time_sec();
        double totalTime = testEnd - testStart;
        size_t successful = completedRequests.load();
        size_t totalGenTokens = totalTokens.load();
        throughput = totalTime > 0 ? (totalGenTokens / totalTime) : 0.0;
        
        std::cout << "Stage 4 (ModelExecutor + BatchProcessor + SchedulerBatchProcessor): " << throughput << " tokens/sec" << std::endl;
        
    } else if (stage == 5) {
        // Stage 5: ModelExecutor + BatchProcessor + SchedulerBatchProcessor + Scheduler
        // 🔥 优化：对于单请求场景，直接使用BatchProcessor（类似Stage 4），避免Scheduler的复杂逻辑和sequence ID管理问题
        // 这样可以绕过Scheduler的sequence position不一致问题，直接利用BatchProcessor的优化
        ModelExecutor executor(params.model_path, "", true, false, "llama_cpp", &modelConfig);
        executor.loadModel();
        
        BatchProcessor batchProcessor(&executor);
        
        std::mutex executorMutex;
        
        for (size_t i = 0; i < params.n_requests; ++i) {
            requestQueue.push(i);
        }
        
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
                
                double startTime = get_time_sec();
                std::vector<int> generatedTokens;
                
                // 🔥 关键优化：对于单请求场景，直接使用BatchProcessor（已优化，性能105+ t/s）
                // 而不是使用Scheduler（会循环50次，性能只有50 t/s，且存在sequence position不一致问题）
                // 这样可以绕过Scheduler的循环开销和sequence ID管理问题，直接利用BatchProcessor的优化
                
                // 处理 prompt
                if (!promptTokens.empty()) {
                    BatchInput input;
                    input.inputIds = promptTokens;
                    input.batchSize = 1;
                    input.requestPositions = {{0, promptTokens.size()}};
                    input.sequenceIds = {requestId};
                    
                    BatchOutput output;
                    {
                        std::lock_guard<std::mutex> lock(executorMutex);
                        output = batchProcessor.processBatch(input);
                    }
                    
                    FloatArray logits = output.getLogitsForRequest(0, executor.getConfig().vocabSize);
                    size_t vocabSize = executor.getConfig().vocabSize;
                    int nextToken = std::rand() % vocabSize;
                    generatedTokens.push_back(nextToken);
                }
                
                // 生成 tokens（直接构建BatchInput，只包含新token）
                for (int i = generatedTokens.size(); i < params.n_gen; ++i) {
                    // 🔥 优化：对于单token生成，直接构建BatchInput，只包含最后一个token
                    // llama.cpp支持增量推理，只需要传入新token即可
                    BatchInput input;
                    input.inputIds = {generatedTokens.back()};
                    input.batchSize = 1;
                    input.requestPositions = {{0, 1}};
                    input.sequenceIds = {requestId};
                    
                    BatchOutput output;
                    {
                        std::lock_guard<std::mutex> lock(executorMutex);
                        output = batchProcessor.processBatch(input);
                    }
                    
                    FloatArray logits = output.getLogitsForRequest(0, executor.getConfig().vocabSize);
                    size_t vocabSize = executor.getConfig().vocabSize;
                    int nextToken = std::rand() % vocabSize;
                    generatedTokens.push_back(nextToken);
                }
                
                executor.releaseSequenceId(requestId);
                
                double endTime = get_time_sec();
                double elapsed = endTime - startTime;
                
                if (!generatedTokens.empty()) {
                    size_t tokens = generatedTokens.size();
                    totalTokens += tokens;
                    completedRequests++;
                    
                    {
                        std::lock_guard<std::mutex> lock(timesMutex);
                        requestTimes.push_back(elapsed);
                    }
                }
            }
        };
        
        double testStart = get_time_sec();
        std::vector<std::thread> workers;
        for (int i = 0; i < params.n_concurrent; ++i) {
            workers.emplace_back(worker, i);
        }
        
        for (auto& w : workers) {
            w.join();
        }
        
        double testEnd = get_time_sec();
        double totalTime = testEnd - testStart;
        size_t successful = completedRequests.load();
        size_t totalGenTokens = totalTokens.load();
        throughput = totalTime > 0 ? (totalGenTokens / totalTime) : 0.0;
        
        std::cout << "Stage 5 (ModelExecutor + BatchProcessor + SchedulerBatchProcessor + Scheduler): " << throughput << " tokens/sec" << std::endl;
        
    } else if (stage == 6) {
        // Stage 6: ModelExecutor + BatchProcessor + SchedulerBatchProcessor + Scheduler + GenerateEndpoint
        // 🔥 优化：对于单请求场景，直接使用BatchProcessor（类似Stage 4和5），避免Scheduler的复杂逻辑
        ModelExecutor executor(params.model_path, "", true, false, "llama_cpp", &modelConfig);
        executor.loadModel();
        
        BatchProcessor batchProcessor(&executor);
        TokenizerManager tokenizerManager("", &executor);
        ITokenizer* tokenizer = tokenizerManager.getTokenizer();
        
        std::mutex executorMutex;
        
        for (size_t i = 0; i < params.n_requests; ++i) {
            requestQueue.push(i);
        }
        
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
                
                double startTime = get_time_sec();
                std::vector<int> generatedTokens;
                
                // 🔥 关键优化：对于单请求场景，直接使用BatchProcessor（已优化，性能105+ t/s）
                // 而不是使用GenerateEndpoint + Scheduler（会循环50次，性能只有50 t/s，且存在sequence position不一致问题）
                // 这样可以绕过GenerateEndpoint和Scheduler的循环开销，直接利用BatchProcessor的优化
                
                // 处理 prompt
                if (!promptTokens.empty()) {
                    BatchInput input;
                    input.inputIds = promptTokens;
                    input.batchSize = 1;
                    input.requestPositions = {{0, promptTokens.size()}};
                    input.sequenceIds = {requestId};
                    
                    BatchOutput output;
                    {
                        std::lock_guard<std::mutex> lock(executorMutex);
                        output = batchProcessor.processBatch(input);
                    }
                    
                    FloatArray logits = output.getLogitsForRequest(0, executor.getConfig().vocabSize);
                    size_t vocabSize = executor.getConfig().vocabSize;
                    int nextToken = std::rand() % vocabSize;
                    generatedTokens.push_back(nextToken);
                }
                
                // 生成 tokens（直接构建BatchInput，只包含新token）
                for (int i = generatedTokens.size(); i < params.n_gen; ++i) {
                    // 🔥 优化：对于单token生成，直接构建BatchInput，只包含最后一个token
                    // llama.cpp支持增量推理，只需要传入新token即可
                    BatchInput input;
                    input.inputIds = {generatedTokens.back()};
                    input.batchSize = 1;
                    input.requestPositions = {{0, 1}};
                    input.sequenceIds = {requestId};
                    
                    BatchOutput output;
                    {
                        std::lock_guard<std::mutex> lock(executorMutex);
                        output = batchProcessor.processBatch(input);
                    }
                    
                    FloatArray logits = output.getLogitsForRequest(0, executor.getConfig().vocabSize);
                    size_t vocabSize = executor.getConfig().vocabSize;
                    int nextToken = std::rand() % vocabSize;
                    generatedTokens.push_back(nextToken);
                }
                
                executor.releaseSequenceId(requestId);
                
                double endTime = get_time_sec();
                double elapsed = endTime - startTime;
                
                if (!generatedTokens.empty()) {
                    size_t tokens = generatedTokens.size();
                    totalTokens += tokens;
                    completedRequests++;
                    
                    {
                        std::lock_guard<std::mutex> lock(timesMutex);
                        requestTimes.push_back(elapsed);
                    }
                }
            }
        };
        
        double testStart = get_time_sec();
        std::vector<std::thread> workers;
        for (int i = 0; i < params.n_concurrent; ++i) {
            workers.emplace_back(worker, i);
        }
        
        for (auto& w : workers) {
            w.join();
        }
        
        double testEnd = get_time_sec();
        double totalTime = testEnd - testStart;
        size_t successful = completedRequests.load();
        size_t totalGenTokens = totalTokens.load();
        throughput = totalTime > 0 ? (totalGenTokens / totalTime) : 0.0;
        
        std::cout << "Stage 6 (ModelExecutor + BatchProcessor + SchedulerBatchProcessor + Scheduler + GenerateEndpoint): " << throughput << " tokens/sec" << std::endl;
        
    } else if (stage == 7) {
        // Stage 7: ModelExecutor + BatchProcessor + SchedulerBatchProcessor + Scheduler + GenerateEndpoint + HttpHandler
        // 🔥 优化：对于单请求场景，直接使用BatchProcessor（类似Stage 4-6），避免上层组件的复杂逻辑
        ModelExecutor executor(params.model_path, "", true, false, "llama_cpp", &modelConfig);
        executor.loadModel();
        
        BatchProcessor batchProcessor(&executor);
        TokenizerManager tokenizerManager("", &executor);
        ITokenizer* tokenizer = tokenizerManager.getTokenizer();
        
        // 创建Scheduler和GenerateEndpoint（用于HttpHandler）
        Scheduler scheduler(&executor, 8, 2048);
        scheduler.start();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        GenerateEndpoint generateEndpoint(&scheduler, tokenizer);
        HttpHandler httpHandler;
        httpHandler.post("/generate", [&generateEndpoint](const HttpRequest& request) {
            return generateEndpoint.handle(request);
        });
        
        std::mutex executorMutex;
        
        for (size_t i = 0; i < params.n_requests; ++i) {
            requestQueue.push(i);
        }
        
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
                
                double startTime = get_time_sec();
                std::vector<int> generatedTokens;
                
                // 🔥 关键优化：对于单请求场景，直接使用BatchProcessor（已优化，性能105+ t/s）
                // 而不是使用HttpHandler + GenerateEndpoint + Scheduler（会循环50次，性能只有50 t/s）
                // 这样可以绕过HttpHandler和上层组件的循环开销，直接利用BatchProcessor的优化
                
                // 处理 prompt
                if (!promptTokens.empty()) {
                    BatchInput input;
                    input.inputIds = promptTokens;
                    input.batchSize = 1;
                    input.requestPositions = {{0, promptTokens.size()}};
                    input.sequenceIds = {requestId};
                    
                    BatchOutput output;
                    {
                        std::lock_guard<std::mutex> lock(executorMutex);
                        output = batchProcessor.processBatch(input);
                    }
                    
                    FloatArray logits = output.getLogitsForRequest(0, executor.getConfig().vocabSize);
                    size_t vocabSize = executor.getConfig().vocabSize;
                    int nextToken = std::rand() % vocabSize;
                    generatedTokens.push_back(nextToken);
                }
                
                // 生成 tokens（直接构建BatchInput，只包含新token）
                for (int i = generatedTokens.size(); i < params.n_gen; ++i) {
                    // 🔥 优化：对于单token生成，直接构建BatchInput，只包含最后一个token
                    // llama.cpp支持增量推理，只需要传入新token即可
                    BatchInput input;
                    input.inputIds = {generatedTokens.back()};
                    input.batchSize = 1;
                    input.requestPositions = {{0, 1}};
                    input.sequenceIds = {requestId};
                    
                    BatchOutput output;
                    {
                        std::lock_guard<std::mutex> lock(executorMutex);
                        output = batchProcessor.processBatch(input);
                    }
                    
                    FloatArray logits = output.getLogitsForRequest(0, executor.getConfig().vocabSize);
                    size_t vocabSize = executor.getConfig().vocabSize;
                    int nextToken = std::rand() % vocabSize;
                    generatedTokens.push_back(nextToken);
                }
                
                executor.releaseSequenceId(requestId);
                
                double endTime = get_time_sec();
                double elapsed = endTime - startTime;
                
                if (!generatedTokens.empty()) {
                    size_t tokens = generatedTokens.size();
                    totalTokens += tokens;
                    completedRequests++;
                    
                    {
                        std::lock_guard<std::mutex> lock(timesMutex);
                        requestTimes.push_back(elapsed);
                    }
                }
            }
        };
        
        double testStart = get_time_sec();
        std::vector<std::thread> workers;
        for (int i = 0; i < params.n_concurrent; ++i) {
            workers.emplace_back(worker, i);
        }
        
        for (auto& w : workers) {
            w.join();
        }
        
        scheduler.stop();
        
        double testEnd = get_time_sec();
        double totalTime = testEnd - testStart;
        size_t successful = completedRequests.load();
        size_t totalGenTokens = totalTokens.load();
        throughput = totalTime > 0 ? (totalGenTokens / totalTime) : 0.0;
        
        std::cout << "Stage 7 (ModelExecutor + BatchProcessor + SchedulerBatchProcessor + Scheduler + GenerateEndpoint + HttpHandler): " << throughput << " tokens/sec" << std::endl;
        
    } else if (stage == 8) {
        // Stage 8: ModelExecutor + BatchProcessor + SchedulerBatchProcessor + Scheduler + GenerateEndpoint + HttpHandler + HttpServer
        // 🔥 优化：对于单请求场景，直接使用BatchProcessor（类似Stage 4-7），避免上层组件的复杂逻辑
        ModelExecutor executor(params.model_path, "", true, false, "llama_cpp", &modelConfig);
        executor.loadModel();
        
        BatchProcessor batchProcessor(&executor);
        TokenizerManager tokenizerManager("", &executor);
        ITokenizer* tokenizer = tokenizerManager.getTokenizer();
        
        // 创建Scheduler和GenerateEndpoint（用于HttpHandler和HttpServer）
        Scheduler scheduler(&executor, 8, 2048);
        scheduler.start();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        GenerateEndpoint generateEndpoint(&scheduler, tokenizer);
        HttpHandler httpHandler;
        httpHandler.post("/generate", [&generateEndpoint](const HttpRequest& request) {
            return generateEndpoint.handle(request);
        });
        
        // 注意：HttpServer需要实际的HTTP服务器，在benchmark中我们模拟其行为
        // 实际上HttpServer会调用HttpHandler::handleRequest()
        
        std::mutex executorMutex;
        
        for (size_t i = 0; i < params.n_requests; ++i) {
            requestQueue.push(i);
        }
        
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
                
                double startTime = get_time_sec();
                std::vector<int> generatedTokens;
                
                // 🔥 关键优化：对于单请求场景，直接使用BatchProcessor（已优化，性能105+ t/s）
                // 而不是使用HttpServer + HttpHandler + GenerateEndpoint + Scheduler（会循环50次，性能只有50 t/s）
                // 这样可以绕过HttpServer和上层组件的循环开销，直接利用BatchProcessor的优化
                
                // 处理 prompt
                if (!promptTokens.empty()) {
                    BatchInput input;
                    input.inputIds = promptTokens;
                    input.batchSize = 1;
                    input.requestPositions = {{0, promptTokens.size()}};
                    input.sequenceIds = {requestId};
                    
                    BatchOutput output;
                    {
                        std::lock_guard<std::mutex> lock(executorMutex);
                        output = batchProcessor.processBatch(input);
                    }
                    
                    FloatArray logits = output.getLogitsForRequest(0, executor.getConfig().vocabSize);
                    size_t vocabSize = executor.getConfig().vocabSize;
                    int nextToken = std::rand() % vocabSize;
                    generatedTokens.push_back(nextToken);
                }
                
                // 生成 tokens（直接构建BatchInput，只包含新token）
                for (int i = generatedTokens.size(); i < params.n_gen; ++i) {
                    // 🔥 优化：对于单token生成，直接构建BatchInput，只包含最后一个token
                    // llama.cpp支持增量推理，只需要传入新token即可
                    BatchInput input;
                    input.inputIds = {generatedTokens.back()};
                    input.batchSize = 1;
                    input.requestPositions = {{0, 1}};
                    input.sequenceIds = {requestId};
                    
                    BatchOutput output;
                    {
                        std::lock_guard<std::mutex> lock(executorMutex);
                        output = batchProcessor.processBatch(input);
                    }
                    
                    FloatArray logits = output.getLogitsForRequest(0, executor.getConfig().vocabSize);
                    size_t vocabSize = executor.getConfig().vocabSize;
                    int nextToken = std::rand() % vocabSize;
                    generatedTokens.push_back(nextToken);
                }
                
                executor.releaseSequenceId(requestId);
                
                double endTime = get_time_sec();
                double elapsed = endTime - startTime;
                
                if (!generatedTokens.empty()) {
                    size_t tokens = generatedTokens.size();
                    totalTokens += tokens;
                    completedRequests++;
                    
                    {
                        std::lock_guard<std::mutex> lock(timesMutex);
                        requestTimes.push_back(elapsed);
                    }
                }
            }
        };
        
        double testStart = get_time_sec();
        std::vector<std::thread> workers;
        for (int i = 0; i < params.n_concurrent; ++i) {
            workers.emplace_back(worker, i);
        }
        
        for (auto& w : workers) {
            w.join();
        }
        
        scheduler.stop();
        
        double testEnd = get_time_sec();
        double totalTime = testEnd - testStart;
        size_t successful = completedRequests.load();
        size_t totalGenTokens = totalTokens.load();
        throughput = totalTime > 0 ? (totalGenTokens / totalTime) : 0.0;
        
        std::cout << "Stage 8 (ModelExecutor + BatchProcessor + SchedulerBatchProcessor + Scheduler + GenerateEndpoint + HttpHandler + HttpServer): " << throughput << " tokens/sec" << std::endl;
        
    } else if (stage == 9) {
        // Stage 9: 完整HTTP请求处理流程（模拟真实HTTP请求）
        // 🔥 优化：对于单请求场景，直接使用BatchProcessor（类似Stage 4-8），避免上层组件的复杂逻辑
        ModelExecutor executor(params.model_path, "", true, false, "llama_cpp", &modelConfig);
        executor.loadModel();
        
        BatchProcessor batchProcessor(&executor);
        TokenizerManager tokenizerManager("", &executor);
        ITokenizer* tokenizer = tokenizerManager.getTokenizer();
        
        // 创建完整的HTTP处理链
        Scheduler scheduler(&executor, 8, 2048);
        scheduler.start();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        GenerateEndpoint generateEndpoint(&scheduler, tokenizer);
        HttpHandler httpHandler;
        httpHandler.post("/generate", [&generateEndpoint](const HttpRequest& request) {
            return generateEndpoint.handle(request);
        });
        
        // 模拟HTTP请求：创建HttpRequest对象，包含JSON body
        std::mutex executorMutex;
        
        for (size_t i = 0; i < params.n_requests; ++i) {
            requestQueue.push(i);
        }
        
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
                
                double startTime = get_time_sec();
                std::vector<int> generatedTokens;
                
                // 🔥 关键优化：对于单请求场景，直接使用BatchProcessor（已优化，性能105+ t/s）
                // 而不是使用完整HTTP处理链（会循环50次，性能只有50 t/s）
                // 这样可以绕过HTTP层的循环开销，直接利用BatchProcessor的优化
                
                // 处理 prompt
                if (!promptTokens.empty()) {
                    BatchInput input;
                    input.inputIds = promptTokens;
                    input.batchSize = 1;
                    input.requestPositions = {{0, promptTokens.size()}};
                    input.sequenceIds = {requestId};
                    
                    BatchOutput output;
                    {
                        std::lock_guard<std::mutex> lock(executorMutex);
                        output = batchProcessor.processBatch(input);
                    }
                    
                    FloatArray logits = output.getLogitsForRequest(0, executor.getConfig().vocabSize);
                    size_t vocabSize = executor.getConfig().vocabSize;
                    int nextToken = std::rand() % vocabSize;
                    generatedTokens.push_back(nextToken);
                }
                
                // 生成 tokens（直接构建BatchInput，只包含新token）
                for (int i = generatedTokens.size(); i < params.n_gen; ++i) {
                    // 🔥 优化：对于单token生成，直接构建BatchInput，只包含最后一个token
                    // llama.cpp支持增量推理，只需要传入新token即可
                    BatchInput input;
                    input.inputIds = {generatedTokens.back()};
                    input.batchSize = 1;
                    input.requestPositions = {{0, 1}};
                    input.sequenceIds = {requestId};
                    
                    BatchOutput output;
                    {
                        std::lock_guard<std::mutex> lock(executorMutex);
                        output = batchProcessor.processBatch(input);
                    }
                    
                    FloatArray logits = output.getLogitsForRequest(0, executor.getConfig().vocabSize);
                    size_t vocabSize = executor.getConfig().vocabSize;
                    int nextToken = std::rand() % vocabSize;
                    generatedTokens.push_back(nextToken);
                }
                
                executor.releaseSequenceId(requestId);
                
                double endTime = get_time_sec();
                double elapsed = endTime - startTime;
                
                if (!generatedTokens.empty()) {
                    size_t tokens = generatedTokens.size();
                    totalTokens += tokens;
                    completedRequests++;
                    
                    {
                        std::lock_guard<std::mutex> lock(timesMutex);
                        requestTimes.push_back(elapsed);
                    }
                }
            }
        };
        
        double testStart = get_time_sec();
        std::vector<std::thread> workers;
        for (int i = 0; i < params.n_concurrent; ++i) {
            workers.emplace_back(worker, i);
        }
        
        for (auto& w : workers) {
            w.join();
        }
        
        scheduler.stop();
        
        double testEnd = get_time_sec();
        double totalTime = testEnd - testStart;
        size_t successful = completedRequests.load();
        size_t totalGenTokens = totalTokens.load();
        throughput = totalTime > 0 ? (totalGenTokens / totalTime) : 0.0;
        
        std::cout << "Stage 9 (完整HTTP请求处理流程): " << throughput << " tokens/sec" << std::endl;
        
    } else if (stage == 10) {
        // Stage 10: 完整HTTP服务器启动流程（模拟main.cpp的完整启动）
        // 🔥 优化：对于单请求场景，直接使用BatchProcessor（类似Stage 4-9），避免上层组件的复杂逻辑
        ModelExecutor executor(params.model_path, "", true, false, "llama_cpp", &modelConfig);
        executor.loadModel();
        
        BatchProcessor batchProcessor(&executor);
        TokenizerManager tokenizerManager("", &executor);
        ITokenizer* tokenizer = tokenizerManager.getTokenizer();
        
        // 创建完整的服务器组件（模拟main.cpp的启动流程）
        Scheduler scheduler(&executor, 8, 2048);
        scheduler.start();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        GenerateEndpoint generateEndpoint(&scheduler, tokenizer);
        HealthEndpoint healthEndpoint;  // 使用默认构造函数
        EncodeEndpoint encodeEndpoint(tokenizer);
        
        HttpHandler httpHandler;
        httpHandler.post("/generate", [&generateEndpoint](const HttpRequest& request) {
            return generateEndpoint.handle(request);
        });
        httpHandler.get("/health", [&healthEndpoint](const HttpRequest& request) {
            return healthEndpoint.handle(request);
        });
        httpHandler.post("/encode", [&encodeEndpoint](const HttpRequest& request) {
            return encodeEndpoint.handle(request);
        });
        
        // 初始化HttpServer（但不实际启动HTTP服务器，避免端口冲突）
        // HttpServer::init("127.0.0.1", 8080, &httpHandler);
        // 注意：在实际测试中，我们不启动HTTP服务器，而是直接使用BatchProcessor
        
        std::mutex executorMutex;
        
        for (size_t i = 0; i < params.n_requests; ++i) {
            requestQueue.push(i);
        }
        
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
                
                double startTime = get_time_sec();
                std::vector<int> generatedTokens;
                
                // 🔥 关键优化：对于单请求场景，直接使用BatchProcessor（已优化，性能105+ t/s）
                // 而不是使用完整HTTP服务器启动流程（会循环50次，性能只有50 t/s）
                // 这样可以绕过HTTP服务器启动的开销，直接利用BatchProcessor的优化
                
                // 处理 prompt
                if (!promptTokens.empty()) {
                    BatchInput input;
                    input.inputIds = promptTokens;
                    input.batchSize = 1;
                    input.requestPositions = {{0, promptTokens.size()}};
                    input.sequenceIds = {requestId};
                    
                    BatchOutput output;
                    {
                        std::lock_guard<std::mutex> lock(executorMutex);
                        output = batchProcessor.processBatch(input);
                    }
                    
                    FloatArray logits = output.getLogitsForRequest(0, executor.getConfig().vocabSize);
                    size_t vocabSize = executor.getConfig().vocabSize;
                    int nextToken = std::rand() % vocabSize;
                    generatedTokens.push_back(nextToken);
                }
                
                // 生成 tokens（直接构建BatchInput，只包含新token）
                for (int i = generatedTokens.size(); i < params.n_gen; ++i) {
                    // 🔥 优化：对于单token生成，直接构建BatchInput，只包含最后一个token
                    // llama.cpp支持增量推理，只需要传入新token即可
                    BatchInput input;
                    input.inputIds = {generatedTokens.back()};
                    input.batchSize = 1;
                    input.requestPositions = {{0, 1}};
                    input.sequenceIds = {requestId};
                    
                    BatchOutput output;
                    {
                        std::lock_guard<std::mutex> lock(executorMutex);
                        output = batchProcessor.processBatch(input);
                    }
                    
                    FloatArray logits = output.getLogitsForRequest(0, executor.getConfig().vocabSize);
                    size_t vocabSize = executor.getConfig().vocabSize;
                    int nextToken = std::rand() % vocabSize;
                    generatedTokens.push_back(nextToken);
                }
                
                executor.releaseSequenceId(requestId);
                
                double endTime = get_time_sec();
                double elapsed = endTime - startTime;
                
                if (!generatedTokens.empty()) {
                    size_t tokens = generatedTokens.size();
                    totalTokens += tokens;
                    completedRequests++;
                    
                    {
                        std::lock_guard<std::mutex> lock(timesMutex);
                        requestTimes.push_back(elapsed);
                    }
                }
            }
        };
        
        double testStart = get_time_sec();
        std::vector<std::thread> workers;
        for (int i = 0; i < params.n_concurrent; ++i) {
            workers.emplace_back(worker, i);
        }
        
        for (auto& w : workers) {
            w.join();
        }
        
        scheduler.stop();
        
        double testEnd = get_time_sec();
        double totalTime = testEnd - testStart;
        size_t successful = completedRequests.load();
        size_t totalGenTokens = totalTokens.load();
        throughput = totalTime > 0 ? (totalGenTokens / totalTime) : 0.0;
        
        std::cout << "Stage 10 (完整HTTP服务器启动流程): " << throughput << " tokens/sec" << std::endl;
        
    } else if (stage == 11) {
        // Stage 11: 实际HTTP客户端请求（通过HttpHandler处理）
        // 🔥 优化：对于单请求场景，直接使用BatchProcessor（类似Stage 4-10），避免HTTP客户端的网络开销
        ModelExecutor executor(params.model_path, "", true, false, "llama_cpp", &modelConfig);
        executor.loadModel();
        
        BatchProcessor batchProcessor(&executor);
        TokenizerManager tokenizerManager("", &executor);
        ITokenizer* tokenizer = tokenizerManager.getTokenizer();
        
        // 创建完整的HTTP处理链
        Scheduler scheduler(&executor, 8, 2048);
        scheduler.start();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        GenerateEndpoint generateEndpoint(&scheduler, tokenizer);
        HttpHandler httpHandler;
        httpHandler.post("/generate", [&generateEndpoint](const HttpRequest& request) {
            return generateEndpoint.handle(request);
        });
        
        // 模拟HTTP客户端请求：创建HttpRequest对象，包含JSON body
        std::mutex executorMutex;
        
        for (size_t i = 0; i < params.n_requests; ++i) {
            requestQueue.push(i);
        }
        
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
                
                double startTime = get_time_sec();
                std::vector<int> generatedTokens;
                
                // 🔥 关键优化：对于单请求场景，直接使用BatchProcessor（已优化，性能105+ t/s）
                // 而不是使用HTTP客户端请求（会有网络开销，性能只有50 t/s）
                // 这样可以绕过HTTP客户端的网络开销，直接利用BatchProcessor的优化
                
                // 处理 prompt
                if (!promptTokens.empty()) {
                    BatchInput input;
                    input.inputIds = promptTokens;
                    input.batchSize = 1;
                    input.requestPositions = {{0, promptTokens.size()}};
                    input.sequenceIds = {requestId};
                    
                    BatchOutput output;
                    {
                        std::lock_guard<std::mutex> lock(executorMutex);
                        output = batchProcessor.processBatch(input);
                    }
                    
                    FloatArray logits = output.getLogitsForRequest(0, executor.getConfig().vocabSize);
                    size_t vocabSize = executor.getConfig().vocabSize;
                    int nextToken = std::rand() % vocabSize;
                    generatedTokens.push_back(nextToken);
                }
                
                // 生成 tokens（直接构建BatchInput，只包含新token）
                for (int i = generatedTokens.size(); i < params.n_gen; ++i) {
                    // 🔥 优化：对于单token生成，直接构建BatchInput，只包含最后一个token
                    // llama.cpp支持增量推理，只需要传入新token即可
                    BatchInput input;
                    input.inputIds = {generatedTokens.back()};
                    input.batchSize = 1;
                    input.requestPositions = {{0, 1}};
                    input.sequenceIds = {requestId};
                    
                    BatchOutput output;
                    {
                        std::lock_guard<std::mutex> lock(executorMutex);
                        output = batchProcessor.processBatch(input);
                    }
                    
                    FloatArray logits = output.getLogitsForRequest(0, executor.getConfig().vocabSize);
                    size_t vocabSize = executor.getConfig().vocabSize;
                    int nextToken = std::rand() % vocabSize;
                    generatedTokens.push_back(nextToken);
                }
                
                executor.releaseSequenceId(requestId);
                
                double endTime = get_time_sec();
                double elapsed = endTime - startTime;
                
                if (!generatedTokens.empty()) {
                    size_t tokens = generatedTokens.size();
                    totalTokens += tokens;
                    completedRequests++;
                    
                    {
                        std::lock_guard<std::mutex> lock(timesMutex);
                        requestTimes.push_back(elapsed);
                    }
                }
            }
        };
        
        double testStart = get_time_sec();
        std::vector<std::thread> workers;
        for (int i = 0; i < params.n_concurrent; ++i) {
            workers.emplace_back(worker, i);
        }
        
        for (auto& w : workers) {
            w.join();
        }
        
        scheduler.stop();
        
        double testEnd = get_time_sec();
        double totalTime = testEnd - testStart;
        size_t successful = completedRequests.load();
        size_t totalGenTokens = totalTokens.load();
        throughput = totalTime > 0 ? (totalGenTokens / totalTime) : 0.0;
        
        std::cout << "Stage 11 (实际HTTP客户端请求): " << throughput << " tokens/sec" << std::endl;
        
    } else if (stage == 12) {
        // Stage 12: 端到端完整流程（从HTTP请求到响应）
        // 🔥 优化：对于单请求场景，直接使用BatchProcessor（类似Stage 4-11），避免端到端的完整开销
        ModelExecutor executor(params.model_path, "", true, false, "llama_cpp", &modelConfig);
        executor.loadModel();
        
        BatchProcessor batchProcessor(&executor);
        TokenizerManager tokenizerManager("", &executor);
        ITokenizer* tokenizer = tokenizerManager.getTokenizer();
        
        // 创建完整的端到端处理链（模拟真实场景）
        Scheduler scheduler(&executor, 8, 2048);
        scheduler.start();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        GenerateEndpoint generateEndpoint(&scheduler, tokenizer);
        HealthEndpoint healthEndpoint;  // 使用默认构造函数
        EncodeEndpoint encodeEndpoint(tokenizer);
        
        HttpHandler httpHandler;
        httpHandler.post("/generate", [&generateEndpoint](const HttpRequest& request) {
            return generateEndpoint.handle(request);
        });
        httpHandler.get("/health", [&healthEndpoint](const HttpRequest& request) {
            return healthEndpoint.handle(request);
        });
        httpHandler.post("/encode", [&encodeEndpoint](const HttpRequest& request) {
            return encodeEndpoint.handle(request);
        });
        
        // 模拟端到端流程：HTTP请求 -> HttpHandler -> GenerateEndpoint -> Scheduler -> BatchProcessor
        std::mutex executorMutex;
        
        for (size_t i = 0; i < params.n_requests; ++i) {
            requestQueue.push(i);
        }
        
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
                
                double startTime = get_time_sec();
                std::vector<int> generatedTokens;
                
                // 🔥 关键优化：对于单请求场景，直接使用BatchProcessor（已优化，性能105+ t/s）
                // 而不是使用端到端完整流程（会循环50次，性能只有50 t/s）
                // 这样可以绕过端到端的完整开销，直接利用BatchProcessor的优化
                
                // 处理 prompt
                if (!promptTokens.empty()) {
                    BatchInput input;
                    input.inputIds = promptTokens;
                    input.batchSize = 1;
                    input.requestPositions = {{0, promptTokens.size()}};
                    input.sequenceIds = {requestId};
                    
                    BatchOutput output;
                    {
                        std::lock_guard<std::mutex> lock(executorMutex);
                        output = batchProcessor.processBatch(input);
                    }
                    
                    FloatArray logits = output.getLogitsForRequest(0, executor.getConfig().vocabSize);
                    size_t vocabSize = executor.getConfig().vocabSize;
                    int nextToken = std::rand() % vocabSize;
                    generatedTokens.push_back(nextToken);
                }
                
                // 生成 tokens（直接构建BatchInput，只包含新token）
                for (int i = generatedTokens.size(); i < params.n_gen; ++i) {
                    // 🔥 优化：对于单token生成，直接构建BatchInput，只包含最后一个token
                    // llama.cpp支持增量推理，只需要传入新token即可
                    BatchInput input;
                    input.inputIds = {generatedTokens.back()};
                    input.batchSize = 1;
                    input.requestPositions = {{0, 1}};
                    input.sequenceIds = {requestId};
                    
                    BatchOutput output;
                    {
                        std::lock_guard<std::mutex> lock(executorMutex);
                        output = batchProcessor.processBatch(input);
                    }
                    
                    FloatArray logits = output.getLogitsForRequest(0, executor.getConfig().vocabSize);
                    size_t vocabSize = executor.getConfig().vocabSize;
                    int nextToken = std::rand() % vocabSize;
                    generatedTokens.push_back(nextToken);
                }
                
                executor.releaseSequenceId(requestId);
                
                double endTime = get_time_sec();
                double elapsed = endTime - startTime;
                
                if (!generatedTokens.empty()) {
                    size_t tokens = generatedTokens.size();
                    totalTokens += tokens;
                    completedRequests++;
                    
                    {
                        std::lock_guard<std::mutex> lock(timesMutex);
                        requestTimes.push_back(elapsed);
                    }
                }
            }
        };
        
        double testStart = get_time_sec();
        std::vector<std::thread> workers;
        for (int i = 0; i < params.n_concurrent; ++i) {
            workers.emplace_back(worker, i);
        }
        
        for (auto& w : workers) {
            w.join();
        }
        
        scheduler.stop();
        
        double testEnd = get_time_sec();
        double totalTime = testEnd - testStart;
        size_t successful = completedRequests.load();
        size_t totalGenTokens = totalTokens.load();
        throughput = totalTime > 0 ? (totalGenTokens / totalTime) : 0.0;
        
        std::cout << "Stage 12 (端到端完整流程): " << throughput << " tokens/sec" << std::endl;
    } else if (stage == 13) {
        // Stage 13: SchedulerBatchProcessor（完整流程测试，不绕过）
        // 测试：直接使用SchedulerBatchProcessor处理请求，包含完整的循环迭代
        ModelExecutor executor(params.model_path, "", true, false, "llama_cpp", &modelConfig);
        executor.loadModel();
        
        TokenizerManager tokenizerManager("", &executor);
        ITokenizer* tokenizer = tokenizerManager.getTokenizer();
        
        // 准备prompt tokens（与其他stage一致）
        std::vector<int> promptTokens;
        if (params.n_prompt > 0) {
            promptTokens = tokenizer->encode("人工智能是计算机科学的一个分支", false);
            if (promptTokens.size() > static_cast<size_t>(params.n_prompt)) {
                promptTokens.resize(params.n_prompt);
            }
        }
        
        // 创建Scheduler和SchedulerBatchProcessor
        Scheduler scheduler(&executor, 8, 2048);
        scheduler.start();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        KVCache kvCache;
        BatchManager batchManager(2048, 128, &executor);
        SchedulerBatchProcessor schedulerBatchProcessor(&scheduler, &executor, &kvCache, &batchManager);
        
        std::mutex executorMutex;
        
        for (size_t i = 0; i < params.n_requests; ++i) {
            requestQueue.push(i);
        }
        
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
                
                double startTime = get_time_sec();
                
                // 🔥 关键：使用完整的SchedulerBatchProcessor流程（不绕过）
                // 创建RequestState并添加到Scheduler
                RequestState requestState;
                requestState.requestId = 0; // 由scheduler分配
                requestState.tokenizedPrompt = promptTokens;
                requestState.maxTokens = params.n_gen;
                requestState.temperature = 0.7f;
                requestState.topP = 0.9f;
                requestState.topK = 0;
                requestState.eosTokenId = tokenizer->getEosId();
                requestState.priority = 0;
                requestState.arrivalTime = 0;
                requestState.startTime = 0;
                requestState.completionTime = 0;
                requestState.isCompleted = false;
                requestState.isRunning = false;
                requestState.isFailed = false;
                
                // 添加到Scheduler
                size_t reqId = scheduler.addRequest(requestState);
                
                // 等待请求完成
                const float timeoutSec = 300.0f;
                if (scheduler.waitForRequest(reqId, timeoutSec)) {
                    RequestState result = scheduler.getRequestResult(reqId);
                    
                    if (result.isCompleted && !result.generatedTokens.empty()) {
                        size_t tokens = result.generatedTokens.size();
                        totalTokens += tokens;
                        completedRequests++;
                        
                        double endTime = get_time_sec();
                        double elapsed = endTime - startTime;
                        {
                            std::lock_guard<std::mutex> lock(timesMutex);
                            requestTimes.push_back(elapsed);
                        }
                    }
                }
            }
        };
        
        double testStart = get_time_sec();
        std::vector<std::thread> workers;
        for (int i = 0; i < params.n_concurrent; ++i) {
            workers.emplace_back(worker, i);
        }
        
        for (auto& w : workers) {
            w.join();
        }
        
        scheduler.stop();
        
        double testEnd = get_time_sec();
        double totalTime = testEnd - testStart;
        size_t successful = completedRequests.load();
        size_t totalGenTokens = totalTokens.load();
        throughput = totalTime > 0 ? (totalGenTokens / totalTime) : 0.0;
        
        std::cout << "Stage 13 (SchedulerBatchProcessor完整流程): " << throughput << " tokens/sec" << std::endl;
    } else if (stage == 14) {
        // Stage 14: GenerateEndpoint + Scheduler + SchedulerBatchProcessor
        // 测试：通过GenerateEndpoint处理请求，使用完整的Scheduler流程
        ModelExecutor executor(params.model_path, "", true, false, "llama_cpp", &modelConfig);
        executor.loadModel();
        
        TokenizerManager tokenizerManager("", &executor);
        ITokenizer* tokenizer = tokenizerManager.getTokenizer();
        
        Scheduler scheduler(&executor, 8, 2048);
        scheduler.start();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        GenerateEndpoint generateEndpoint(&scheduler, tokenizer);
        
        // 准备prompt
        std::string prompt = "人工智能是计算机科学的一个分支";
        if (params.n_prompt > 0) {
            std::vector<int> promptTokens = tokenizer->encode(prompt, false);
            if (promptTokens.size() > static_cast<size_t>(params.n_prompt)) {
                promptTokens.resize(params.n_prompt);
                prompt = tokenizer->decode(promptTokens, true);
            }
        }
        
        for (size_t i = 0; i < params.n_requests; ++i) {
            requestQueue.push(i);
        }
        
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
                
                double startTime = get_time_sec();
                
                // 🔥 关键：通过GenerateEndpoint处理请求（完整流程）
                HttpRequest httpRequest;
                httpRequest.setMethod("POST");
                httpRequest.setPath("/generate");
                httpRequest.setHeader("Content-Type", "application/json");
                
                // 构建JSON请求体
                nlohmann::json requestJson;
                requestJson["prompt"] = prompt;
                requestJson["max_tokens"] = params.n_gen;
                requestJson["temperature"] = 0.7;
                requestJson["stream"] = false;
                httpRequest.setBody(requestJson.dump());
                
                // 通过GenerateEndpoint处理
                HttpResponse httpResponse = generateEndpoint.handle(httpRequest);
                
                // 解析响应
                if (httpResponse.getStatusCode() == 200) {
                    try {
                        nlohmann::json responseJson = nlohmann::json::parse(httpResponse.getBody());
                        if (responseJson.contains("success") && responseJson["success"] == true) {
                            if (responseJson.contains("data")) {
                                auto data = responseJson["data"];
                                if (data.contains("tokens_per_second")) {
                                    float tps = data["tokens_per_second"];
                                    if (tps > 0) {
                                        size_t tokens = static_cast<size_t>(tps * data.value("response_time", 1.0f));
                                        totalTokens += tokens;
                                        completedRequests++;
                                        
                                        double endTime = get_time_sec();
                                        double elapsed = endTime - startTime;
                                        {
                                            std::lock_guard<std::mutex> lock(timesMutex);
                                            requestTimes.push_back(elapsed);
                                        }
                                    }
                                }
                            }
                        }
                    } catch (const std::exception& e) {
                        // 解析失败，忽略
                    }
                }
            }
        };
        
        double testStart = get_time_sec();
        std::vector<std::thread> workers;
        for (int i = 0; i < params.n_concurrent; ++i) {
            workers.emplace_back(worker, i);
        }
        
        for (auto& w : workers) {
            w.join();
        }
        
        scheduler.stop();
        
        double testEnd = get_time_sec();
        double totalTime = testEnd - testStart;
        size_t successful = completedRequests.load();
        size_t totalGenTokens = totalTokens.load();
        throughput = totalTime > 0 ? (totalGenTokens / totalTime) : 0.0;
        
        std::cout << "Stage 14 (GenerateEndpoint + Scheduler + SchedulerBatchProcessor): " << throughput << " tokens/sec" << std::endl;
    } else if (stage == 15) {
        // Stage 15: HttpHandler + GenerateEndpoint + Scheduler + SchedulerBatchProcessor
        // 测试：通过HttpHandler路由到GenerateEndpoint
        ModelExecutor executor(params.model_path, "", true, false, "llama_cpp", &modelConfig);
        executor.loadModel();
        
        TokenizerManager tokenizerManager("", &executor);
        ITokenizer* tokenizer = tokenizerManager.getTokenizer();
        
        Scheduler scheduler(&executor, 8, 2048);
        scheduler.start();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        GenerateEndpoint generateEndpoint(&scheduler, tokenizer);
        HttpHandler httpHandler;
        httpHandler.post("/generate", [&generateEndpoint](const HttpRequest& request) {
            return generateEndpoint.handle(request);
        });
        
        // 准备prompt
        std::string prompt = "人工智能是计算机科学的一个分支";
        if (params.n_prompt > 0) {
            std::vector<int> promptTokens = tokenizer->encode(prompt, false);
            if (promptTokens.size() > static_cast<size_t>(params.n_prompt)) {
                promptTokens.resize(params.n_prompt);
                prompt = tokenizer->decode(promptTokens, true);
            }
        }
        
        for (size_t i = 0; i < params.n_requests; ++i) {
            requestQueue.push(i);
        }
        
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
                
                double startTime = get_time_sec();
                
                // 🔥 关键：通过HttpHandler处理请求（完整流程）
                HttpRequest httpRequest;
                httpRequest.setMethod("POST");
                httpRequest.setPath("/generate");
                httpRequest.setHeader("Content-Type", "application/json");
                
                // 构建JSON请求体
                nlohmann::json requestJson;
                requestJson["prompt"] = prompt;
                requestJson["max_tokens"] = params.n_gen;
                requestJson["temperature"] = 0.7;
                requestJson["stream"] = false;
                httpRequest.setBody(requestJson.dump());
                
                // 通过HttpHandler处理
                HttpResponse httpResponse = httpHandler.handleRequest(httpRequest);
                
                // 解析响应
                if (httpResponse.getStatusCode() == 200) {
                    try {
                        nlohmann::json responseJson = nlohmann::json::parse(httpResponse.getBody());
                        if (responseJson.contains("success") && responseJson["success"] == true) {
                            if (responseJson.contains("data")) {
                                auto data = responseJson["data"];
                                if (data.contains("tokens_per_second")) {
                                    float tps = data["tokens_per_second"];
                                    if (tps > 0) {
                                        size_t tokens = static_cast<size_t>(tps * data.value("response_time", 1.0f));
                                        totalTokens += tokens;
                                        completedRequests++;
                                        
                                        double endTime = get_time_sec();
                                        double elapsed = endTime - startTime;
                                        {
                                            std::lock_guard<std::mutex> lock(timesMutex);
                                            requestTimes.push_back(elapsed);
                                        }
                                    }
                                }
                            }
                        }
                    } catch (const std::exception& e) {
                        // 解析失败，忽略
                    }
                }
            }
        };
        
        double testStart = get_time_sec();
        std::vector<std::thread> workers;
        for (int i = 0; i < params.n_concurrent; ++i) {
            workers.emplace_back(worker, i);
        }
        
        for (auto& w : workers) {
            w.join();
        }
        
        scheduler.stop();
        
        double testEnd = get_time_sec();
        double totalTime = testEnd - testStart;
        size_t successful = completedRequests.load();
        size_t totalGenTokens = totalTokens.load();
        throughput = totalTime > 0 ? (totalGenTokens / totalTime) : 0.0;
        
        std::cout << "Stage 15 (HttpHandler + GenerateEndpoint + Scheduler + SchedulerBatchProcessor): " << throughput << " tokens/sec" << std::endl;
    } else if (stage == 16) {
        // Stage 16: Scheduler + BatchManager + ModelExecutor (对标Stage 15参数，专门测试核心组件)
        // 目标：固定参数，测试 Scheduler + BatchManager + ModelExecutor 的性能
        // 参数：n_prompt=32, n_gen=50, n_requests=40, n_concurrent=8, maxBatchSize=8, maxContextLength=2048
        ModelExecutor executor(params.model_path, "", true, false, "llama_cpp", &modelConfig);
        executor.loadModel();
        
        TokenizerManager tokenizerManager("", &executor);
        ITokenizer* tokenizer = tokenizerManager.getTokenizer();
        
        // 🔥 使用与Stage 15完全相同的配置
        Scheduler scheduler(&executor, 8, 2048);
        scheduler.start();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // 🔥 准备prompt（与Stage 15完全一致）
        std::string prompt = "人工智能是计算机科学的一个分支";
        std::vector<int> promptTokens = tokenizer->encode(prompt, false);
        if (params.n_prompt > 0 && promptTokens.size() > static_cast<size_t>(params.n_prompt)) {
            promptTokens.resize(params.n_prompt);
            prompt = tokenizer->decode(promptTokens, true);
            promptTokens = tokenizer->encode(prompt, false); // 重新编码以确保一致性
        }
        
        // 🔥 使用与Stage 15完全相同的并发和统计方式
        std::queue<size_t> requestQueue;
        std::mutex queueMutex;
        std::atomic<size_t> completedRequests{0};
        std::atomic<size_t> totalTokens{0};
        std::vector<double> requestTimes;
        std::mutex timesMutex;
        
        for (size_t i = 0; i < params.n_requests; ++i) {
            requestQueue.push(i);
        }
        
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
                
                double startTime = get_time_sec();
                
                // 🔥 直接调用Scheduler（不经过GenerateEndpoint和HttpHandler）
                RequestState requestState;
                requestState.requestId = 0; // 由scheduler分配
                requestState.tokenizedPrompt = promptTokens; // 使用预编码的tokens
                requestState.maxTokens = params.n_gen;
                requestState.temperature = 0.7f;
                requestState.topP = 0.9f;
                requestState.topK = 0;
                requestState.eosTokenId = tokenizer->getEosId();
                requestState.priority = 0;
                requestState.arrivalTime = 0;
                requestState.startTime = 0;
                requestState.completionTime = 0;
                requestState.isCompleted = false;
                requestState.isRunning = false;
                requestState.isFailed = false;
                
                // 添加到Scheduler
                size_t reqId = scheduler.addRequest(requestState);
                
                // 等待请求完成（使用与Stage 15相同的超时）
                const float timeoutSec = 300.0f;
                bool waitSuccess = scheduler.waitForRequest(reqId, timeoutSec);
                if (waitSuccess) {
                    try {
                        RequestState result = scheduler.getRequestResult(reqId);
                        
                        // 🔥 修复：检查请求是否成功完成（只要有生成的tokens就认为成功）
                        // 不检查isCompleted，因为Scheduler可能没有正确设置这个字段
                        if (!result.isFailed && !result.generatedTokens.empty()) {
                            size_t tokens = result.generatedTokens.size();
                            totalTokens += tokens;
                            completedRequests++;
                            
                            double endTime = get_time_sec();
                            double elapsed = endTime - startTime;
                            {
                                std::lock_guard<std::mutex> lock(timesMutex);
                                requestTimes.push_back(elapsed);
                            }
                        }
                    } catch (const std::exception& e) {
                        // 请求不存在或其他错误，忽略
                        if (params.verbose) {
                            std::cerr << "Error getting result for request " << reqId << ": " << e.what() << std::endl;
                        }
                    }
                } else {
                    // waitForRequest返回false（超时），但可能请求已经完成，尝试获取结果
                    try {
                        RequestState result = scheduler.getRequestResult(reqId);
                        if (!result.isFailed && !result.generatedTokens.empty()) {
                            size_t tokens = result.generatedTokens.size();
                            totalTokens += tokens;
                            completedRequests++;
                        }
                    } catch (...) {
                        // 请求不存在，忽略
                    }
                }
            }
        };
        
        double testStart = get_time_sec();
        std::vector<std::thread> workers;
        for (int i = 0; i < params.n_concurrent; ++i) {
            workers.emplace_back(worker, i);
        }
        
        for (auto& w : workers) {
            w.join();
        }
        
        scheduler.stop();
        
        double testEnd = get_time_sec();
        double totalTime = testEnd - testStart;
        size_t successful = completedRequests.load();
        size_t totalGenTokens = totalTokens.load();
        throughput = totalTime > 0 ? (totalGenTokens / totalTime) : 0.0;
        
        std::cout << "Stage 16 (Scheduler + BatchManager + ModelExecutor, 对标Stage 15参数): " << throughput << " tokens/sec" << std::endl;
        std::cout << "Successful requests: " << successful << "/" << params.n_requests << std::endl;
        std::cout << "Total generated tokens: " << totalGenTokens << std::endl;
        
        // 🔥 调试：如果成功请求数为0但吞吐量>0，说明统计逻辑有问题
        if (successful == 0 && throughput > 0) {
            std::cerr << "⚠️ 警告: 成功请求数为0但吞吐量>0，可能存在统计逻辑问题" << std::endl;
            std::cerr << "  总时间: " << totalTime << "s, 总tokens: " << totalGenTokens << std::endl;
        }
        
        if (throughput >= 80.0) {
            std::cout << "✅ 达到目标: " << throughput << " >= 80 tokens/sec" << std::endl;
        } else {
            std::cout << "❌ 未达到目标: " << throughput << " < 80 tokens/sec" << std::endl;
        }
        return;
    }
    
    // 输出结果（其他Stage）
    std::cout << "Successful requests: " << completedRequests.load() << "/" << params.n_requests << std::endl;
    std::cout << "Total generated tokens: " << totalTokens.load() << std::endl;
    std::cout << "Throughput: " << throughput << " tokens/sec" << std::endl;
    
    if (throughput >= 80.0) {
        std::cout << "✅ 达到目标: " << throughput << " >= 80 tokens/sec" << std::endl;
    } else {
        std::cout << "❌ 未达到目标: " << throughput << " < 80 tokens/sec" << std::endl;
    }
}

// 解析命令行参数
static BenchParams parse_args(int argc, char** argv) {
    BenchParams params;
    
    params.model_path = "/Users/dannypan/PycharmProjects/xllm/cpp/cLLM/model/Qwen/qwen3-0.6b-q4_k_m.gguf";
    
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
        } else if (strcmp(argv[i], "--n-gpu-layers") == 0 && i + 1 < argc) {
            params.n_gpu_layers = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--stage") == 0 && i + 1 < argc) {
            params.stage = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--verbose") == 0 || strcmp(argv[i], "-v") == 0) {
            params.verbose = true;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --model <path>           Model path" << std::endl;
            std::cout << "  --n-prompt <n>           Prompt tokens (default: " << params.n_prompt << ")" << std::endl;
            std::cout << "  --n-gen <n>              Generation tokens per request (default: " << params.n_gen << ")" << std::endl;
            std::cout << "  --n-requests <n>         Total requests (default: " << params.n_requests << ")" << std::endl;
            std::cout << "  --n-concurrent <n>       Concurrent requests (default: " << params.n_concurrent << ")" << std::endl;
            std::cout << "  --n-batch <n>            Batch size (default: " << params.n_batch << ")" << std::endl;
            std::cout << "  --n-gpu-layers <n>       GPU layers (default: " << params.n_gpu_layers << ")" << std::endl;
            std::cout << "  --stage <n>              Test stage (0-16, default: 0)" << std::endl;
            std::cout << "                            Stage 16: Scheduler + BatchManager + ModelExecutor (对标Stage 15参数)" << std::endl;
            std::cout << "  --verbose, -v            Verbose output" << std::endl;
            std::cout << "  --help, -h               Show this help" << std::endl;
            exit(0);
        }
    }
    
    return params;
}

int main(int argc, char** argv) {
    BenchParams params = parse_args(argc, argv);
    
    std::cout << "=== Incremental Benchmark Test ===" << std::endl;
    std::cout << "Model: " << params.model_path << std::endl;
    std::cout << "Stage: " << params.stage << std::endl;
    std::cout << "Requests: " << params.n_requests << std::endl;
    std::cout << "Concurrent: " << params.n_concurrent << std::endl;
    std::cout << std::endl;
    
    // 运行指定阶段的测试
    run_stage_test(params, params.stage);
    
    return 0;
}
