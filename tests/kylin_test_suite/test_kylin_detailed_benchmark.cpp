/**
 * @file test_kylin_detailed_benchmark.cpp
 * @brief Kylin 引擎精细化性能基准测试
 *
 * 测试场景:
 * 1. Prefill 阶段测试（不同 prompt 长度）
 * 2. Decode 阶段测试（单 token 增量生成）
 * 3. 端到端生成测试（Prefill + 连续 Decode）
 * 4. 长上下文测试
 * 5. 多次迭代统计（P50/P95/P99）
 */

#include <gtest/gtest.h>
#include "cllm/kylin/gguf/transformer.h"
#include "cllm/common/logger.h"

#include <filesystem>
#include <vector>
#include <chrono>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <random>
#include <iomanip>
#include <iostream>
#include <cstdlib>

using namespace cllm;
using namespace cllm::kylin;
namespace fs = std::filesystem;

static double getEnvDouble(const char* name, double fallback) {
    const char* val = std::getenv(name);
    if (!val || !*val) return fallback;
    try {
        return std::stod(val);
    } catch (...) {
        return fallback;
    }
}

static double decodeTargetForPrefill(size_t prefillLen) {
    if (prefillLen >= 64) return getEnvDouble("CLLM_DECODE_TPS_TARGET_VLONG", 15.0);
    if (prefillLen >= 32) return getEnvDouble("CLLM_DECODE_TPS_TARGET_LONG", 20.0);
    return getEnvDouble("CLLM_DECODE_TPS_TARGET_SHORT", 20.0);
}

static double decodeTargetForRegression() {
    return getEnvDouble("CLLM_DECODE_TPS_TARGET_REGRESSION", 20.0);
}

static double decodeTargetForBenchmark() {
    return getEnvDouble("CLLM_DECODE_TPS_TARGET_BENCH", 25.0);
}

static double decodeTargetForContext(size_t ctxLen) {
    if (ctxLen >= 64) return getEnvDouble("CLLM_DECODE_TPS_TARGET_CTX64", 14.0);
    return getEnvDouble("CLLM_DECODE_TPS_TARGET_CTX", 20.0);
}

/**
 * @brief 性能统计结构
 */
struct PerfStats {
    double mean = 0.0;
    double stddev = 0.0;
    double p50 = 0.0;
    double p95 = 0.0;
    double p99 = 0.0;
    double min = 0.0;
    double max = 0.0;
    size_t count = 0;
    
    static PerfStats compute(const std::vector<double>& values) {
        PerfStats stats;
        if (values.empty()) return stats;
        
        stats.count = values.size();
        
        // Mean
        stats.mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
        
        // Stddev
        double sqSum = 0.0;
        for (double v : values) {
            sqSum += (v - stats.mean) * (v - stats.mean);
        }
        stats.stddev = std::sqrt(sqSum / values.size());
        
        // Min/Max
        stats.min = *std::min_element(values.begin(), values.end());
        stats.max = *std::max_element(values.begin(), values.end());
        
        // Percentiles
        std::vector<double> sorted = values;
        std::sort(sorted.begin(), sorted.end());
        
        auto percentile = [&sorted](double p) {
            size_t idx = static_cast<size_t>(std::ceil(p * sorted.size() / 100.0) - 1);
            idx = std::min(idx, sorted.size() - 1);
            return sorted[idx];
        };
        
        stats.p50 = percentile(50.0);
        stats.p95 = percentile(95.0);
        stats.p99 = percentile(99.0);
        
        return stats;
    }
    
    void print(const std::string& name, const std::string& unit) const {
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "[" << name << "] "
                  << "Mean=" << mean << " " << unit
                  << ", Std=" << stddev
                  << ", P50=" << p50
                  << ", P95=" << p95
                  << ", P99=" << p99
                  << ", Min=" << min
                  << ", Max=" << max
                  << " (n=" << count << ")" << std::endl;
    }
};

/**
 * @brief 简单 argmax 采样
 */
int argmaxSample(const std::vector<float>& logits, size_t vocabSize) {
    if (logits.empty() || vocabSize == 0) return 0;
    
    // 获取最后一个位置的 logits
    size_t numPositions = logits.size() / vocabSize;
    const float* lastLogits = logits.data() + (numPositions - 1) * vocabSize;
    
    int maxIdx = 0;
    float maxVal = lastLogits[0];
    for (size_t i = 1; i < vocabSize; ++i) {
        if (lastLogits[i] > maxVal) {
            maxVal = lastLogits[i];
            maxIdx = static_cast<int>(i);
        }
    }
    return maxIdx;
}

/**
 * @brief Kylin 详细基准测试类
 */
class KylinDetailedBenchmarkTest : public ::testing::Test {
protected:
    std::unique_ptr<GGMLTransformerModel> model_;
    std::string modelPath_;
    bool initialized_ = false;
    size_t vocabSize_ = 0;
    
    void SetUp() override {
        // 设置日志级别为 Info（减少调试输出）
        Logger::instance().setLevel(spdlog::level::info);
        
        // 查找模型文件
        std::vector<std::string> possiblePaths = {
            "model/Qwen/qwen3-0.6b-q4_k_m.gguf",
            "../model/Qwen/qwen3-0.6b-q4_k_m.gguf",
            "../../model/Qwen/qwen3-0.6b-q4_k_m.gguf",
            "/Users/dannypan/PycharmProjects/xllm/cpp/cLLM/model/Qwen/qwen3-0.6b-q4_k_m.gguf"
        };
        
        const char* envPath = std::getenv("CLLM_MODEL_PATH");
        if (envPath && fs::exists(envPath)) {
            possiblePaths.insert(possiblePaths.begin(), envPath);
        }
        
        for (const auto& path : possiblePaths) {
            if (fs::exists(path)) {
                modelPath_ = fs::absolute(path).string();
                break;
            }
        }
        
        if (modelPath_.empty()) {
            GTEST_SKIP() << "Model file not found";
            return;
        }
        
        std::cout << "Model path: " << modelPath_ << std::endl;
        
        // 加载模型
        model_ = std::make_unique<GGMLTransformerModel>(BackendType::CPU);
        if (!model_->loadFromGGUF(modelPath_)) {
            GTEST_SKIP() << "Failed to load model";
            return;
        }
        
        vocabSize_ = model_->getConfig().vocabSize;
        initialized_ = true;
        
        std::cout << "Model loaded successfully" << std::endl;
        std::cout << "Vocab size: " << vocabSize_ << std::endl;
        std::cout << "Max seq len: " << model_->getMaxSeqLen() << std::endl;
    }
    
    void TearDown() override {
        model_.reset();
    }
    
    /**
     * @brief 生成随机 token 序列
     */
    std::vector<int32_t> generateRandomTokens(size_t length) {
        std::vector<int32_t> tokens(length);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int32_t> dis(100, static_cast<int32_t>(vocabSize_ - 1));
        
        for (size_t i = 0; i < length; ++i) {
            tokens[i] = dis(gen);
        }
        return tokens;
    }
    
    /**
     * @brief 测量单次 Prefill 时间（毫秒）
     */
    double measurePrefillTime(const std::vector<int32_t>& inputIds) {
        model_->clearKVCache();
        
        auto start = std::chrono::high_resolution_clock::now();
        model_->forward(inputIds);
        auto end = std::chrono::high_resolution_clock::now();
        
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
    
    /**
     * @brief 测量单次 Decode 时间（毫秒）
     */
    double measureDecodeTime(int32_t tokenId, size_t position) {
        auto start = std::chrono::high_resolution_clock::now();
        model_->forwardOneToken(tokenId, position);
        auto end = std::chrono::high_resolution_clock::now();
        
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
};

/**
 * @brief Prefill 阶段基准测试（不同 prompt 长度）
 */
TEST_F(KylinDetailedBenchmarkTest, PrefillBenchmark) {
    if (!initialized_) {
        GTEST_SKIP() << "Model not initialized";
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Prefill Benchmark (Different Prompt Lengths)" << std::endl;
    std::cout << "========================================" << std::endl;
    
    std::vector<size_t> promptLengths = {1, 4, 8, 16, 32, 64};
    const int numIterations = 5; // 每个长度测试次数
    
    for (size_t promptLen : promptLengths) {
        std::vector<double> times;
        times.reserve(numIterations);
        
        for (int i = 0; i < numIterations; ++i) {
            auto tokens = generateRandomTokens(promptLen);
            double time_ms = measurePrefillTime(tokens);
            times.push_back(time_ms);
        }
        
        PerfStats stats = PerfStats::compute(times);
        double tokensPerSec = (promptLen * 1000.0) / stats.mean;
        
        std::cout << "Prompt length=" << std::setw(4) << promptLen
                  << ": " << std::fixed << std::setprecision(2)
                  << "mean=" << stats.mean << " ms"
                  << ", P50=" << stats.p50 << " ms"
                  << ", P95=" << stats.p95 << " ms"
                  << ", throughput=" << tokensPerSec << " tok/s" << std::endl;
        
        EXPECT_GT(tokensPerSec, 10.0) << "Prefill throughput too low for length " << promptLen;
    }
}

/**
 * @brief Decode 阶段基准测试（单 token 增量生成）
 */
TEST_F(KylinDetailedBenchmarkTest, DecodeBenchmark) {
    if (!initialized_) {
        GTEST_SKIP() << "Model not initialized";
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Decode Benchmark (Single Token Generation)" << std::endl;
    std::cout << "========================================" << std::endl;
    
    const size_t prefillLen = 16; // 先 prefill 一些 tokens
    const int numDecodeTokens = 32; // 生成的 token 数量
    
    // 初始化：Prefill
    auto prefillTokens = generateRandomTokens(prefillLen);
    model_->clearKVCache();
    model_->forward(prefillTokens);
    
    std::cout << "After prefill: KV cache length = " << model_->getKVCacheLength() << std::endl;
    
    // Decode 阶段
    std::vector<double> decodeTimes;
    decodeTimes.reserve(numDecodeTokens);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int32_t> dis(100, static_cast<int32_t>(vocabSize_ - 1));
    
    for (int i = 0; i < numDecodeTokens; ++i) {
        int32_t nextToken = dis(gen);
        size_t position = model_->getKVCacheLength();
        
        double time_ms = measureDecodeTime(nextToken, position);
        decodeTimes.push_back(time_ms);
    }
    
    PerfStats stats = PerfStats::compute(decodeTimes);
    double tokensPerSec = 1000.0 / stats.mean; // 单 token 生成速度
    
    stats.print("Decode Latency", "ms");
    std::cout << "Decode throughput: " << std::fixed << std::setprecision(2)
              << tokensPerSec << " tok/s" << std::endl;
    
    const double target = decodeTargetForBenchmark();
    EXPECT_GE(tokensPerSec, target) << "Decode throughput should be >= " << target << " tok/s";
}

/**
 * @brief 端到端生成测试（Prefill + 连续 Decode）
 */
TEST_F(KylinDetailedBenchmarkTest, EndToEndGenerationBenchmark) {
    if (!initialized_) {
        GTEST_SKIP() << "Model not initialized";
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "End-to-End Generation Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // 测试配置
    struct TestConfig {
        size_t prefillLen;
        size_t generateLen;
        std::string name;
    };
    
    std::vector<TestConfig> configs = {
        {8, 16, "Short prompt (8) + 16 tokens"},
        {16, 32, "Medium prompt (16) + 32 tokens"},
        {32, 64, "Long prompt (32) + 64 tokens"},
        {64, 32, "Very long prompt (64) + 32 tokens"},
    };
    
    for (const auto& config : configs) {
        std::cout << "\n--- " << config.name << " ---" << std::endl;
        
        model_->clearKVCache();
        
        // 生成随机 prompt tokens
        auto prefillTokens = generateRandomTokens(config.prefillLen);
        
        // Prefill 阶段
        auto prefillStart = std::chrono::high_resolution_clock::now();
        std::vector<float> logits = model_->forward(prefillTokens);
        auto prefillEnd = std::chrono::high_resolution_clock::now();
        
        double prefillTime = std::chrono::duration<double, std::milli>(prefillEnd - prefillStart).count();
        double prefillTokPerSec = (config.prefillLen * 1000.0) / prefillTime;
        
        std::cout << "Prefill: " << std::fixed << std::setprecision(2)
                  << prefillTime << " ms (" << prefillTokPerSec << " tok/s)" << std::endl;
        
        // Decode 阶段
        std::vector<double> decodeTimes;
        decodeTimes.reserve(config.generateLen);
        
        std::vector<int32_t> generatedTokens;
        generatedTokens.reserve(config.generateLen);
        
        for (size_t i = 0; i < config.generateLen; ++i) {
            // 使用 argmax 采样
            int nextToken = argmaxSample(logits, vocabSize_);
            generatedTokens.push_back(nextToken);
            
            // Decode
            size_t position = model_->getKVCacheLength();
            auto decodeStart = std::chrono::high_resolution_clock::now();
            logits = model_->forwardOneToken(nextToken, position);
            auto decodeEnd = std::chrono::high_resolution_clock::now();
            
            double decodeTime = std::chrono::duration<double, std::milli>(decodeEnd - decodeStart).count();
            decodeTimes.push_back(decodeTime);
        }
        
        // 统计 Decode 性能
        PerfStats decodeStats = PerfStats::compute(decodeTimes);
        double decodeTokPerSec = 1000.0 / decodeStats.mean;
        
        double totalDecodeTime = std::accumulate(decodeTimes.begin(), decodeTimes.end(), 0.0);
        double totalTime = prefillTime + totalDecodeTime;
        double overallTokPerSec = ((config.prefillLen + config.generateLen) * 1000.0) / totalTime;
        
        std::cout << "Decode: mean=" << decodeStats.mean << " ms/tok"
                  << ", P50=" << decodeStats.p50 << " ms"
                  << ", P95=" << decodeStats.p95 << " ms"
                  << " (" << decodeTokPerSec << " tok/s)" << std::endl;
        std::cout << "Total: " << totalTime << " ms, Overall: " << overallTokPerSec << " tok/s" << std::endl;
        
        // 验证 Decode 吞吐量（长上下文允许下降）
        const double target = decodeTargetForPrefill(config.prefillLen);
        EXPECT_GE(decodeTokPerSec, target) << "Decode throughput should be >= "
                                           << target << " tok/s for " << config.name;
    }
}

/**
 * @brief 长上下文压力测试
 */
TEST_F(KylinDetailedBenchmarkTest, LongContextBenchmark) {
    if (!initialized_) {
        GTEST_SKIP() << "Model not initialized";
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Long Context Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;
    
    const size_t maxSeqLen = std::min(model_->getMaxSeqLen(), static_cast<size_t>(512));
    
    // 测试不同的上下文长度（限制在64以内避免计算内存溢出）
    std::vector<size_t> contextLengths = {16, 32, 64};
    
    for (size_t ctxLen : contextLengths) {
        if (ctxLen > maxSeqLen) continue;
        
        std::cout << "\n--- Context length: " << ctxLen << " ---" << std::endl;
        
        model_->clearKVCache();
        
        // Prefill
        auto prefillTokens = generateRandomTokens(ctxLen);
        
        auto start = std::chrono::high_resolution_clock::now();
        model_->forward(prefillTokens);
        auto end = std::chrono::high_resolution_clock::now();
        
        double prefillTime = std::chrono::duration<double, std::milli>(end - start).count();
        double prefillTokPerSec = (ctxLen * 1000.0) / prefillTime;
        
        std::cout << "Prefill: " << std::fixed << std::setprecision(2)
                  << prefillTime << " ms (" << prefillTokPerSec << " tok/s)" << std::endl;
        
        // 测量 Decode 延迟（在长上下文后）
        const int numDecodeTokens = 10;
        std::vector<double> decodeTimes;
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int32_t> dis(100, static_cast<int32_t>(vocabSize_ - 1));
        
        for (int i = 0; i < numDecodeTokens; ++i) {
            int32_t nextToken = dis(gen);
            size_t position = model_->getKVCacheLength();
            
            auto decodeStart = std::chrono::high_resolution_clock::now();
            model_->forwardOneToken(nextToken, position);
            auto decodeEnd = std::chrono::high_resolution_clock::now();
            
            double decodeTime = std::chrono::duration<double, std::milli>(decodeEnd - decodeStart).count();
            decodeTimes.push_back(decodeTime);
        }
        
        PerfStats decodeStats = PerfStats::compute(decodeTimes);
        double decodeTokPerSec = 1000.0 / decodeStats.mean;
        
        std::cout << "Decode after " << ctxLen << " tokens: mean=" << decodeStats.mean << " ms"
                  << " (" << decodeTokPerSec << " tok/s)" << std::endl;
        
        // 长上下文 Decode 可能略慢，但应该还在合理范围
        const double ctxTarget = decodeTargetForContext(ctxLen);
        EXPECT_GE(decodeTokPerSec, ctxTarget) << "Decode throughput should be >= "
                                              << ctxTarget << " tok/s after context " << ctxLen;
    }
}

/**
 * @brief 性能回归测试（固定场景，用于CI）
 */
TEST_F(KylinDetailedBenchmarkTest, PerformanceRegressionTest) {
    if (!initialized_) {
        GTEST_SKIP() << "Model not initialized";
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Performance Regression Test" << std::endl;
    std::cout << "========================================" << std::endl;
    
    const size_t prefillLen = 16;
    const size_t generateLen = 32;
    const int numRuns = 3;
    
    std::vector<double> prefillThroughputs;
    std::vector<double> decodeThroughputs;
    
    for (int run = 0; run < numRuns; ++run) {
        model_->clearKVCache();
        
        // Prefill
        auto prefillTokens = generateRandomTokens(prefillLen);
        
        auto prefillStart = std::chrono::high_resolution_clock::now();
        std::vector<float> logits = model_->forward(prefillTokens);
        auto prefillEnd = std::chrono::high_resolution_clock::now();
        
        double prefillTime = std::chrono::duration<double, std::milli>(prefillEnd - prefillStart).count();
        prefillThroughputs.push_back((prefillLen * 1000.0) / prefillTime);
        
        // Decode
        std::vector<double> decodeTimes;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int32_t> dis(100, static_cast<int32_t>(vocabSize_ - 1));
        
        for (size_t i = 0; i < generateLen; ++i) {
            int32_t nextToken = dis(gen);
            size_t position = model_->getKVCacheLength();
            
            auto decodeStart = std::chrono::high_resolution_clock::now();
            logits = model_->forwardOneToken(nextToken, position);
            auto decodeEnd = std::chrono::high_resolution_clock::now();
            
            decodeTimes.push_back(std::chrono::duration<double, std::milli>(decodeEnd - decodeStart).count());
        }
        
        PerfStats decodeStats = PerfStats::compute(decodeTimes);
        decodeThroughputs.push_back(1000.0 / decodeStats.mean);
    }
    
    PerfStats prefillStats = PerfStats::compute(prefillThroughputs);
    PerfStats decodeStats = PerfStats::compute(decodeThroughputs);
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Prefill throughput: " << prefillStats.mean << " +/- " << prefillStats.stddev << " tok/s" << std::endl;
    std::cout << "Decode throughput:  " << decodeStats.mean << " +/- " << decodeStats.stddev << " tok/s" << std::endl;
    
    // 性能基线
    EXPECT_GE(prefillStats.mean, 100.0) << "Prefill throughput regression (expected >= 100 tok/s)";
    const double decodeTarget = decodeTargetForRegression();
    EXPECT_GE(decodeStats.mean, decodeTarget) << "Decode throughput regression (expected >= "
                                             << decodeTarget << " tok/s)";
    
    std::cout << "\n*** PERFORMANCE SUMMARY ***" << std::endl;
    std::cout << "Prefill: " << prefillStats.mean << " tok/s (target: >= 100)" << std::endl;
    std::cout << "Decode:  " << decodeStats.mean << " tok/s (target: >= " << decodeTarget << ")" << std::endl;
    std::cout << "***************************" << std::endl;
}

/**
 * @brief Warmup 后的稳定性测试
 */
TEST_F(KylinDetailedBenchmarkTest, WarmupStabilityTest) {
    if (!initialized_) {
        GTEST_SKIP() << "Model not initialized";
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Warmup Stability Test" << std::endl;
    std::cout << "========================================" << std::endl;
    
    const size_t promptLen = 16;
    const int warmupRuns = 3;
    const int measureRuns = 10;
    
    // Warmup 阶段
    std::cout << "Warming up..." << std::endl;
    for (int i = 0; i < warmupRuns; ++i) {
        auto tokens = generateRandomTokens(promptLen);
        measurePrefillTime(tokens);
    }
    
    // 测量阶段
    std::cout << "Measuring..." << std::endl;
    std::vector<double> times;
    for (int i = 0; i < measureRuns; ++i) {
        auto tokens = generateRandomTokens(promptLen);
        double time_ms = measurePrefillTime(tokens);
        times.push_back(time_ms);
        std::cout << "  Run " << (i + 1) << ": " << std::fixed << std::setprecision(2) << time_ms << " ms" << std::endl;
    }
    
    PerfStats stats = PerfStats::compute(times);
    double cv = (stats.mean > 0) ? (stats.stddev / stats.mean) : 0;
    
    std::cout << "\nAfter warmup:" << std::endl;
    stats.print("Prefill Latency", "ms");
    std::cout << "Coefficient of Variation: " << std::fixed << std::setprecision(2) << (cv * 100) << "%" << std::endl;
    
    const double cvThreshold = getEnvDouble("CLLM_WARMUP_CV_THRESHOLD", 0.6);
    EXPECT_LT(cv, cvThreshold) << "Performance should be stable (CV < " << (cvThreshold * 100) << "%)";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    std::cout << "========================================" << std::endl;
    std::cout << "Kylin Detailed Performance Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return RUN_ALL_TESTS();
}
