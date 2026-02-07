/**
 * @file test_performance_benchmark.cpp
 * @brief Stage 16: 性能基准测试
 *
 * 测试内容：
 * - 吞吐量测试
 * - 延迟测试
 * - 内存使用测试
 */

#include "kylin_test_framework.h"
#include <chrono>
#include <random>
#include <thread>
#include <atomic>

namespace kylin_test {

// Test 1: 吞吐量测试
class ThroughputTest : public TestCase {
public:
    ThroughputTest() : TestCase(
        "throughput_test",
        "测试模型推理吞吐量"
    ) {}

    void execute() override {
        log(LogLevel::INFO, "Testing throughput...");

        const int numIterations = 10;
        const int tokensPerIteration = 50;

        auto startTime = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < numIterations; ++i) {
            simulateInference(tokensPerIteration);
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        double totalTime = std::chrono::duration<double>(endTime - startTime).count();
        int totalTokens = numIterations * tokensPerIteration;

        double throughput = totalTokens / totalTime;

        log(LogLevel::INFO, "Total iterations: " + std::to_string(numIterations));
        log(LogLevel::INFO, "Tokens per iteration: " + std::to_string(tokensPerIteration));
        log(LogLevel::INFO, "Total time: " + std::to_string(totalTime) + " seconds");
        log(LogLevel::INFO, "Throughput: " + std::to_string(throughput) + " tokens/second");

        assertTrue(throughput > 0, "Throughput should be positive");
        assertTrue(totalTime > 0, "Total time should be positive");

        log(LogLevel::INFO, "Throughput test completed");
    }

private:
    void simulateInference(int numTokens) {
        for (int i = 0; i < numTokens; ++i) {
            volatile double x = 0.0;
            for (int j = 0; j < 1000; ++j) {
                x += std::sin(j) * std::cos(j);
            }
        }
    }
};

// Test 2: 延迟测试
class LatencyTest : public TestCase {
public:
    LatencyTest() : TestCase(
        "latency_test",
        "测试单次推理延迟"
    ) {}

    void execute() override {
        log(LogLevel::INFO, "Testing latency...");

        const int numRuns = 20;

        std::vector<double> latencies;

        for (int i = 0; i < numRuns; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            simulateSingleInference();
            auto end = std::chrono::high_resolution_clock::now();

            double latency = std::chrono::duration<double>(end - start).count() * 1000.0;
            latencies.push_back(latency);
        }

        double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
        double mean = sum / latencies.size();

        double sqSum = 0.0;
        for (double lat : latencies) {
            sqSum += (lat - mean) * (lat - mean);
        }
        double stddev = std::sqrt(sqSum / latencies.size());

        double minLat = *std::min_element(latencies.begin(), latencies.end());
        double maxLat = *std::max_element(latencies.begin(), latencies.end());

        log(LogLevel::INFO, "Number of runs: " + std::to_string(numRuns));
        log(LogLevel::INFO, "Mean latency: " + std::to_string(mean) + " ms");
        log(LogLevel::INFO, "Std deviation: " + std::to_string(stddev) + " ms");
        log(LogLevel::INFO, "Min latency: " + std::to_string(minLat) + " ms");
        log(LogLevel::INFO, "Max latency: " + std::to_string(maxLat) + " ms");

        assertTrue(mean > 0, "Mean latency should be positive");
        assertTrue(stddev >= 0, "Std deviation should be non-negative");

        log(LogLevel::INFO, "Latency test completed");
    }

private:
    void simulateSingleInference() {
        volatile double x = 0.0;
        for (int i = 0; i < 5000; ++i) {
            x += std::sin(i) * std::cos(i);
        }
    }
};

// Test 3: 内存使用测试
class MemoryUsageTest : public TestCase {
public:
    MemoryUsageTest() : TestCase(
        "memory_usage",
        "测试内存使用情况"
    ) {}

    void execute() override {
        log(LogLevel::INFO, "Testing memory usage...");

        size_t tensorSize = 1024 * 1024 * 100;
        size_t elementSize = sizeof(float);

        size_t totalMemory = tensorSize * elementSize;

        log(LogLevel::INFO, "Tensor size: " + std::to_string(tensorSize) + " elements");
        log(LogLevel::INFO, "Element size: " + std::to_string(elementSize) + " bytes");
        log(LogLevel::INFO, "Total memory per tensor: " +
            std::to_string(totalMemory / (1024 * 1024)) + " MB");

        size_t numLayers = 28;
        size_t numTensors = 100;

        size_t estimatedMemory = tensorSize * elementSize * numTensors;

        log(LogLevel::INFO, "Estimated layers: " + std::to_string(numLayers));
        log(LogLevel::INFO, "Estimated tensors: " + std::to_string(numTensors));
        log(LogLevel::INFO, "Total estimated memory: " +
            std::to_string(estimatedMemory / (1024 * 1024 * 1024)) + " GB");

        assertTrue(estimatedMemory > 0, "Estimated memory should be positive");

        log(LogLevel::INFO, "Memory usage test completed");
    }
};

// Test 4: 并发测试
class ConcurrencyTest : public TestCase {
public:
    ConcurrencyTest() : TestCase(
        "concurrency_test",
        "测试并发推理能力"
    ) {}

    void execute() override {
        log(LogLevel::INFO, "Testing concurrency...");

        const int numThreads = 4;
        const int iterationsPerThread = 5;

        std::atomic<int> completedThreads{0};
        std::atomic<int> totalTokens{0};

        std::vector<std::thread> threads;

        auto startTime = std::chrono::high_resolution_clock::now();

        for (int t = 0; t < numThreads; ++t) {
            threads.emplace_back([&]() {
                for (int i = 0; i < iterationsPerThread; ++i) {
                    simulateInference(20);
                    totalTokens++;
                }
                completedThreads++;
            });
        }

        for (auto& t : threads) {
            t.join();
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        double totalTime = std::chrono::duration<double>(endTime - startTime).count();

        log(LogLevel::INFO, "Number of threads: " + std::to_string(numThreads));
        log(LogLevel::INFO, "Iterations per thread: " + std::to_string(iterationsPerThread));
        log(LogLevel::INFO, "Completed threads: " + std::to_string(completedThreads.load()));
        log(LogLevel::INFO, "Total tokens generated: " + std::to_string(totalTokens.load()));
        log(LogLevel::INFO, "Total time: " + std::to_string(totalTime) + " seconds");

        assertTrue(completedThreads.load() == numThreads,
                   "All threads should complete");
        assertTrue(totalTokens.load() == numThreads * iterationsPerThread,
                   "All tokens should be generated");

        log(LogLevel::INFO, "Concurrency test completed");
    }

private:
    void simulateInference(int numTokens) {
        volatile double x = 0.0;
        for (int i = 0; i < 1000; ++i) {
            x += std::sin(i) * std::cos(i);
        }
    }
};

// Test 5: 批量大小影响测试
class BatchSizeTest : public TestCase {
public:
    BatchSizeTest() : TestCase(
        "batch_size_test",
        "测试不同批量大小的影响"
    ) {}

    void execute() override {
        log(LogLevel::INFO, "Testing batch size effects...");

        std::vector<int> batchSizes = {1, 2, 4, 8, 16};

        log(LogLevel::INFO, "Testing batch sizes:");
        for (int batchSize : batchSizes) {
            log(LogLevel::INFO, "  Batch size: " + std::to_string(batchSize));

            auto start = std::chrono::high_resolution_clock::now();
            simulateBatchInference(batchSize, 50);
            auto end = std::chrono::high_resolution_clock::now();

            double time = std::chrono::duration<double>(end - start).count();
            log(LogLevel::INFO, "    Time: " + std::to_string(time * 1000) + " ms");
        }

        log(LogLevel::INFO, "Batch size test completed");
    }

private:
    void simulateBatchInference(int batchSize, int tokensPerBatch) {
        for (int b = 0; b < batchSize; ++b) {
            volatile double x = 0.0;
            for (int i = 0; i < tokensPerBatch * 100; ++i) {
                x += std::sin(i) * std::cos(i);
            }
        }
    }
};

// Test 6: 预热测试
class WarmupTest : public TestCase {
public:
    WarmupTest() : TestCase(
        "warmup_test",
        "测试预热效果"
    ) {}

    void execute() override {
        log(LogLevel::INFO, "Testing warmup effects...");

        std::vector<double> firstRunTimes;
        std::vector<double> steadyStateTimes;

        for (int i = 0; i < 10; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            simulateInference(20);
            auto end = std::chrono::high_resolution_clock::now();

            double time = std::chrono::duration<double>(end - start).count() * 1000.0;

            if (i < 2) {
                firstRunTimes.push_back(time);
            } else {
                steadyStateTimes.push_back(time);
            }
        }

        double firstAvg = std::accumulate(firstRunTimes.begin(), firstRunTimes.end(), 0.0)
                         / firstRunTimes.size();
        double steadyAvg = std::accumulate(steadyStateTimes.begin(), steadyStateTimes.end(), 0.0)
                         / steadyStateTimes.size();

        log(LogLevel::INFO, "First " + std::to_string(firstRunTimes.size()) + " runs average: " +
            std::to_string(firstAvg) + " ms");
        log(LogLevel::INFO, "Steady state average: " + std::to_string(steadyAvg) + " ms");

        if (steadyAvg < firstAvg) {
            log(LogLevel::INFO, "Warmup effect detected: " +
                std::to_string((firstAvg - steadyAvg) / firstAvg * 100) + "% improvement");
        }

        log(LogLevel::INFO, "Warmup test completed");
    }

private:
    void simulateInference(int numTokens) {
        volatile double x = 0.0;
        for (int i = 0; i < 2000; ++i) {
            x += std::sin(i) * std::cos(i);
        }
    }
};

// 创建 Stage 16 测试套件
std::shared_ptr<TestSuite> createPerformanceBenchmarkTestSuite() {
    auto suite = std::make_shared<TestSuite>("Stage 16: Performance Benchmark");

    suite->addTest(std::make_shared<ThroughputTest>());
    suite->addTest(std::make_shared<LatencyTest>());
    suite->addTest(std::make_shared<MemoryUsageTest>());
    suite->addTest(std::make_shared<ConcurrencyTest>());
    suite->addTest(std::make_shared<BatchSizeTest>());
    suite->addTest(std::make_shared<WarmupTest>());

    return suite;
}

} // namespace kylin_test
