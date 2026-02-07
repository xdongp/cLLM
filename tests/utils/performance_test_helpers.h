#pragma once

#include <chrono>
#include <string>
#include <iostream>
#include <functional>

namespace cllm {
namespace test {

/**
 * @brief 性能测试辅助工具类
 * 提供性能测量和基准测试的便捷方法
 */
class PerformanceTestHelpers {
public:
    /**
     * @brief 时间计时器类
     */
    class Timer {
    public:
        Timer() : start_(std::chrono::high_resolution_clock::now()) {}
        
        /**
         * @brief 获取已经过的时间（毫秒）
         */
        double elapsedMs() const {
            auto end = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::milli>(end - start_).count();
        }
        
        /**
         * @brief 获取已经过的时间（微秒）
         */
        double elapsedUs() const {
            auto end = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::micro>(end - start_).count();
        }
        
        /**
         * @brief 获取已经过的时间（秒）
         */
        double elapsedSeconds() const {
            auto end = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double>(end - start_).count();
        }
        
        /**
         * @brief 重置计时器
         */
        void reset() {
            start_ = std::chrono::high_resolution_clock::now();
        }
        
    private:
        std::chrono::time_point<std::chrono::high_resolution_clock> start_;
    };
    
    /**
     * @brief 性能统计数据
     */
    struct PerformanceStats {
        double minTime = 0.0;      // 最小执行时间（ms）
        double maxTime = 0.0;      // 最大执行时间（ms）
        double avgTime = 0.0;      // 平均执行时间（ms）
        double totalTime = 0.0;    // 总执行时间（ms）
        size_t iterations = 0;     // 迭代次数
        double throughput = 0.0;   // 吞吐量（ops/sec）
        
        void print(const std::string& testName) const {
            std::cout << "\n=== Performance Stats: " << testName << " ===" << std::endl;
            std::cout << "Iterations: " << iterations << std::endl;
            std::cout << "Total Time: " << totalTime << " ms" << std::endl;
            std::cout << "Average Time: " << avgTime << " ms" << std::endl;
            std::cout << "Min Time: " << minTime << " ms" << std::endl;
            std::cout << "Max Time: " << maxTime << " ms" << std::endl;
            std::cout << "Throughput: " << throughput << " ops/sec" << std::endl;
            std::cout << "====================================\n" << std::endl;
        }
    };
    
    /**
     * @brief 运行基准测试
     * @param testFunc 测试函数
     * @param iterations 迭代次数
     * @param testName 测试名称
     * @return 性能统计数据
     */
    static PerformanceStats benchmark(
        std::function<void()> testFunc,
        size_t iterations,
        const std::string& testName = "Benchmark") {
        
        PerformanceStats stats;
        stats.iterations = iterations;
        stats.minTime = std::numeric_limits<double>::max();
        stats.maxTime = 0.0;
        
        Timer totalTimer;
        
        for (size_t i = 0; i < iterations; ++i) {
            Timer timer;
            testFunc();
            double elapsed = timer.elapsedMs();
            
            stats.minTime = std::min(stats.minTime, elapsed);
            stats.maxTime = std::max(stats.maxTime, elapsed);
        }
        
        stats.totalTime = totalTimer.elapsedMs();
        stats.avgTime = stats.totalTime / iterations;
        stats.throughput = (iterations * 1000.0) / stats.totalTime;
        
        return stats;
    }
    
    /**
     * @brief 运行并发基准测试
     * @param testFunc 测试函数
     * @param numThreads 线程数
     * @param iterationsPerThread 每个线程的迭代次数
     * @param testName 测试名称
     * @return 性能统计数据
     */
    static PerformanceStats concurrentBenchmark(
        std::function<void()> testFunc,
        size_t numThreads,
        size_t iterationsPerThread,
        const std::string& testName = "Concurrent Benchmark") {
        
        PerformanceStats stats;
        stats.iterations = numThreads * iterationsPerThread;
        
        Timer totalTimer;
        
        std::vector<std::thread> threads;
        for (size_t t = 0; t < numThreads; ++t) {
            threads.emplace_back([&testFunc, iterationsPerThread]() {
                for (size_t i = 0; i < iterationsPerThread; ++i) {
                    testFunc();
                }
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
        
        stats.totalTime = totalTimer.elapsedMs();
        stats.avgTime = stats.totalTime / stats.iterations;
        stats.throughput = (stats.iterations * 1000.0) / stats.totalTime;
        
        return stats;
    }
    
    /**
     * @brief 测量函数执行时间
     * @param func 要测量的函数
     * @param funcName 函数名称（用于日志）
     * @return 执行时间（毫秒）
     */
    template<typename Func>
    static double measureExecutionTime(Func&& func, const std::string& funcName = "") {
        Timer timer;
        func();
        double elapsed = timer.elapsedMs();
        
        if (!funcName.empty()) {
            std::cout << funcName << " took " << elapsed << " ms" << std::endl;
        }
        
        return elapsed;
    }
};

} // namespace test
} // namespace cllm
