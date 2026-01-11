#include <gtest/gtest.h>

#include <cllm/inference/inference_engine.h>
#include <cllm/model/config.h>
#include <cllm/common/json.h>

#include <chrono>
#include <filesystem>
#include <string>

using namespace cllm;
using namespace cllm::inference;

namespace {

std::string getEnvOrEmpty(const char* name) {
    const char* v = std::getenv(name);
    return v ? std::string(v) : std::string();
}

std::string detectProjectRoot() {
    namespace fs = std::filesystem;
    fs::path cwd = fs::current_path();

    // 常见执行目录：
    // - <repo>/build
    // - <repo>/build/bin
    // - <repo>
    for (int up = 0; up <= 3; ++up) {
        fs::path candidate = cwd;
        for (int i = 0; i < up; ++i) {
            candidate = candidate.parent_path();
        }
        if (fs::exists(candidate / "model") && fs::exists(candidate / "config")) {
            return candidate.string();
        }
    }
    return cwd.string();
}

std::string getModelBinPath(const std::string &dtype) {
    // ✅ 优先使用环境变量（避免硬编码个人目录）
    // - CLLM_TEST_MODEL_BIN_DIR: 指向包含 qwen3_0.6b_cllm_*.bin 的目录
    const std::string envDir = getEnvOrEmpty("CLLM_TEST_MODEL_BIN_DIR");

    std::string baseDir;
    if (!envDir.empty()) {
        baseDir = envDir;
    } else {
        // 默认使用当前仓库下的路径
        baseDir = detectProjectRoot() + "/model/Qwen";
    }

    return baseDir + "/qwen3_0.6b_cllm_" + dtype + ".bin";
}

std::string getDefaultModelBinPath() {
    return getModelBinPath("fp32");
}

std::string getDefaultConfigJsonPath() {
    // ✅ 优先使用环境变量（HF 模型目录，包含 config.json/tokenizer.json）
    // - CLLM_TEST_HF_MODEL_DIR: 例如 <repo>/model/Qwen/Qwen3-0.6B
    const std::string hfDir = getEnvOrEmpty("CLLM_TEST_HF_MODEL_DIR");
    if (!hfDir.empty()) {
        return hfDir + "/config.json";
    }

    return detectProjectRoot() + "/model/Qwen/Qwen3-0.6B/config.json";
}

bool loadQwenConfig(ModelConfig &config) {
    // 硬编码 Qwen3-0.6B 配置，避免复杂的 JSON 解析
    config.vocabSize = 151936;
    config.hiddenSize = 1024;
    config.numLayers = 28;
    config.numAttentionHeads = 16;
    config.numKeyValueHeads = 8;  // Qwen3 使用 GQA
    config.intermediateSize = 3072;
    config.maxSequenceLength = 40960;

    config.modelType = "qwen";
    config.useKVCache = false;
    config.useQuantization = false;
    config.useMemoryCompression = false;
    config.quantizationType.clear();

    return true;
}

} // namespace

TEST(InferenceEngineQwenTest, ForwardBasic_FP32) {
    const std::string binPath = getModelBinPath("fp32");
    if (!std::filesystem::exists(binPath)) {
        GTEST_SKIP() << "Qwen fp32 bin file not found at " << binPath
                     << ", please ensure local model bins exist (see model/export_qwen_bin.py) "
                     << "or set CLLM_TEST_MODEL_BIN_DIR.";
    }

    ModelConfig config;
    if (!loadQwenConfig(config)) {
        GTEST_SKIP() << "Failed to load Qwen config.json, skipping test.";
    }

    InferenceEngine engine(config, binPath);
    ASSERT_TRUE(engine.initialize());

    std::vector<int> inputIds = {1, 2, 3, 4};
    Tensor logits = engine.forward(inputIds);

    const auto &shape = logits.shape();
    ASSERT_EQ(shape.size(), 2u);
    EXPECT_EQ(shape[0], inputIds.size());
    EXPECT_EQ(shape[1], config.vocabSize);
}

TEST(InferenceEngineQwenTest, ForwardBasic_FP16) {
    const std::string binPath = getModelBinPath("fp16");
    if (!std::filesystem::exists(binPath)) {
        GTEST_SKIP() << "Qwen fp16 bin file not found at " << binPath;
    }

    ModelConfig config;
    if (!loadQwenConfig(config)) {
        GTEST_SKIP() << "Failed to load Qwen config.json";
    }

    InferenceEngine engine(config, binPath);
    ASSERT_TRUE(engine.initialize());

    std::vector<int> inputIds = {1, 2, 3, 4};
    Tensor logits = engine.forward(inputIds);

    const auto &shape = logits.shape();
    ASSERT_EQ(shape.size(), 2u);
    EXPECT_EQ(shape[0], inputIds.size());
    EXPECT_EQ(shape[1], config.vocabSize);
}

TEST(InferenceEngineQwenTest, ForwardBasic_INT8) {
    const std::string binPath = getModelBinPath("int8");
    if (!std::filesystem::exists(binPath)) {
        GTEST_SKIP() << "Qwen int8 bin file not found at " << binPath;
    }

    ModelConfig config;
    if (!loadQwenConfig(config)) {
        GTEST_SKIP() << "Failed to load Qwen config.json";
    }

    InferenceEngine engine(config, binPath);
    ASSERT_TRUE(engine.initialize());

    std::vector<int> inputIds = {1, 2, 3, 4};
    Tensor logits = engine.forward(inputIds);

    const auto &shape = logits.shape();
    ASSERT_EQ(shape.size(), 2u);
    EXPECT_EQ(shape[0], inputIds.size());
    EXPECT_EQ(shape[1], config.vocabSize);
}

TEST(InferenceEngineQwenTest, PerformanceComparison) {
    ModelConfig config;
    if (!loadQwenConfig(config)) {
        GTEST_SKIP() << "Failed to load Qwen config.json";
    }

    const std::vector<std::string> dtypes = {"fp32", "fp16", "int8"};
    const std::vector<int> inputIds = {1, 2, 3, 4, 5, 6, 7, 8};
    const int warmupRuns = 2;
    const int benchRuns = 10;

    std::cout << "\n=== Performance Comparison ==="  << std::endl;
    std::cout << "Input tokens: " << inputIds.size() << std::endl;
    std::cout << "Warmup runs: " << warmupRuns << ", Bench runs: " << benchRuns << std::endl;

    for (const auto &dtype : dtypes) {
        const std::string binPath = getModelBinPath(dtype);
        if (!std::filesystem::exists(binPath)) {
            std::cout << "[" << dtype << "] SKIP (file not found)" << std::endl;
            continue;
        }

        InferenceEngine engine(config, binPath);
        if (!engine.initialize()) {
            std::cout << "[" << dtype << "] SKIP (init failed)" << std::endl;
            continue;
        }

        // Warmup
        for (int i = 0; i < warmupRuns; ++i) {
            engine.forward(inputIds);
        }

        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < benchRuns; ++i) {
            Tensor logits = engine.forward(inputIds);
        }
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        double avgMs = static_cast<double>(duration) / benchRuns;
        double tokensPerSec = (inputIds.size() * benchRuns * 1000.0) / duration;

        std::cout << "[" << dtype << "] Avg: " << avgMs << " ms/iter, "
                  << tokensPerSec << " tokens/sec" << std::endl;
    }
}
