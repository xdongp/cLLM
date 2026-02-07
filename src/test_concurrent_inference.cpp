#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <memory>
#include <mutex>
#include <atomic>
#include <iomanip>

#include "cllm/kylin/hf/transformer.h"
#include "cllm/kylin/hf/ggml_backend.h"
#include "cllm/kylin/hf/config.h"

using namespace cllm::kylin;

struct RequestResult {
    size_t requestId;
    double durationMs;
    size_t tokensGenerated;
    bool success;
    std::string errorMessage;
};

class ConcurrentInferenceTester {
public:
    ConcurrentInferenceTester(const std::string& modelPath, int numLayers = 28)
        : modelPath_(modelPath), numLayers_(numLayers) {}

    bool initialize() {
        try {
            model_ = std::make_unique<HFTransformerModel>(modelPath_, DeviceType::Metal, QuantType::FP16);
            
            if (!model_->isLoaded()) {
                std::cerr << "Failed to load model" << std::endl;
                return false;
            }
            
            std::cout << "Model loaded successfully!" << std::endl;
            std::cout << "Vocab size: " << model_->vocabSize() << std::endl;
            std::cout << "Hidden size: " << model_->hiddenSize() << std::endl;
            std::cout << "Num layers: " << numLayers_ << std::endl;
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Exception during initialization: " << e.what() << std::endl;
            return false;
        }
    }

    std::vector<RequestResult> runConcurrentTest(int concurrency, int tokensPerRequest = 10) {
        std::cout << "\n=== Running Concurrent Test (Concurrency: " << concurrency << ") ===" << std::endl;
        std::cout << "Tokens per request: " << tokensPerRequest << std::endl;
        
        std::vector<RequestResult> results(concurrency);
        std::vector<std::thread> threads;
        std::atomic<int> completed(0);
        
        auto startTime = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < concurrency; ++i) {
            threads.emplace_back([this, i, tokensPerRequest, &results, &completed]() {
                try {
                    auto reqStart = std::chrono::high_resolution_clock::now();
                    
                    // 生成 tokensPerRequest 个 token
                    std::vector<int32_t> inputIds = {1};  // 初始 token
                    size_t requestId = i;
                    
                    for (int j = 0; j < tokensPerRequest; ++j) {
                        auto result = model_->forwardWithRequestId(inputIds, requestId);
                        if (result.empty()) {
                            results[i] = {requestId, 0, 0, false, "Empty result"};
                            return;
                        }
                        
                        // 简单的采样：选择概率最高的 token
                        int maxToken = 0;
                        float maxProb = result[0];
                        for (size_t k = 1; k < result.size(); ++k) {
                            if (result[k] > maxProb) {
                                maxProb = result[k];
                                maxToken = static_cast<int>(k);
                            }
                        }
                        
                        inputIds = {static_cast<int32_t>(maxToken)};
                    }
                    
                    auto reqEnd = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(reqEnd - reqStart);
                    
                    results[i] = {
                        requestId,
                        static_cast<double>(duration.count()),
                        static_cast<size_t>(tokensPerRequest),
                        true,
                        ""
                    };
                    
                    completed++;
                    
                } catch (const std::exception& e) {
                    auto reqEnd = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(reqEnd - std::chrono::high_resolution_clock::now());
                    
                    results[i] = {
                        static_cast<size_t>(i),
                        0.0,
                        0,
                        false,
                        e.what()
                    };
                    
                    completed++;
                }
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
        
        auto endTime = std::chrono::high_resolution_clock::now();
        auto totalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        
        std::cout << "All requests completed in: " << totalDuration.count() << " ms" << std::endl;
        std::cout << "Completed: " << completed.load() << "/" << concurrency << std::endl;
        
        return results;
    }

    void printResults(const std::vector<RequestResult>& results) {
        if (results.empty()) {
            std::cout << "No results to display" << std::endl;
            return;
        }
        
        std::cout << "\n=== Results ===" << std::endl;
        
        double totalDuration = 0.0;
        size_t totalTokens = 0;
        int successCount = 0;
        
        for (const auto& result : results) {
            std::cout << "Request " << result.requestId << ": ";
            if (result.success) {
                std::cout << "✅ " << result.durationMs << " ms, " 
                          << result.tokensGenerated << " tokens, "
                          << std::fixed << std::setprecision(2)
                          << (result.tokensGenerated * 1000.0 / result.durationMs) << " TPS";
                totalDuration += result.durationMs;
                totalTokens += result.tokensGenerated;
                successCount++;
            } else {
                std::cout << "❌ Failed: " << result.errorMessage;
            }
            std::cout << std::endl;
        }
        
        std::cout << "\n=== Summary ===" << std::endl;
        std::cout << "Success rate: " << successCount << "/" << results.size() 
                  << " (" << (100.0 * successCount / results.size()) << "%)" << std::endl;
        
        if (successCount > 0) {
            double avgDuration = totalDuration / successCount;
            double avgTPS = (totalTokens * 1000.0) / totalDuration;
            double maxDuration = 0.0;
            double minDuration = std::numeric_limits<double>::max();
            
            for (const auto& result : results) {
                if (result.success) {
                    maxDuration = std::max(maxDuration, result.durationMs);
                    minDuration = std::min(minDuration, result.durationMs);
                }
            }
            
            std::cout << "Average duration: " << avgDuration << " ms" << std::endl;
            std::cout << "Min duration: " << minDuration << " ms" << std::endl;
            std::cout << "Max duration: " << maxDuration << " ms" << std::endl;
            std::cout << "Total tokens: " << totalTokens << std::endl;
            std::cout << "Overall TPS: " << avgTPS << " tokens/s" << std::endl;
            std::cout << "Average TPS per request: " << (totalTokens * 1000.0 / totalDuration) << " tokens/s" << std::endl;
        }
    }

private:
    std::string modelPath_;
    int numLayers_;
    std::unique_ptr<HFTransformerModel> model_;
};

int main() {
    std::string modelPath = "/Users/dannypan/PycharmProjects/cLLM/model/Qwen/Qwen3-0.6B";
    
    ConcurrentInferenceTester tester(modelPath);
    
    if (!tester.initialize()) {
        std::cerr << "Failed to initialize tester" << std::endl;
        return 1;
    }
    
    // 测试 2 个并发
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "TEST 1: 2 Concurrent Requests" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    auto results2 = tester.runConcurrentTest(2, 10);
    tester.printResults(results2);
    
    // 等待一段时间，让系统稳定
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    // 测试 4 个并发
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "TEST 2: 4 Concurrent Requests" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    auto results4 = tester.runConcurrentTest(4, 10);
    tester.printResults(results4);
    
    std::cout << "\n✅ All concurrent tests completed!" << std::endl;
    
    return 0;
}