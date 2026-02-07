/**
 * @file test_model_loading.cpp
 * @brief Stage 1: 模型加载测试
 *
 * 测试内容：
 * - 配置文件解析
 * - 权重加载验证
 * - 内存分配检查
 * - 量化格式验证
 */

#include "kylin_test_framework.h"
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

namespace kylin_test {

namespace fs = std::filesystem;

// 测试配置
struct ModelLoadingConfig {
    std::string modelPath = "/Users/dannypan/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B";
    std::string configPath;
    std::string weightsPath;

    ModelLoadingConfig() {
        configPath = modelPath + "/config.json";
        weightsPath = modelPath + "/model.safetensors";
    }
};

// Test 1: 配置文件存在性检查
class ConfigFileExistenceTest : public TestCase {
public:
    ConfigFileExistenceTest() : TestCase(
        "config_file_existence",
        "验证配置文件是否存在"
    ) {}

    void execute() override {
        ModelLoadingConfig config;

        log(LogLevel::INFO, "Checking config file: " + config.configPath);
        assertTrue(fs::exists(config.configPath),
                   "Config file does not exist: " + config.configPath);

        log(LogLevel::INFO, "Checking weights file: " + config.weightsPath);
        assertTrue(fs::exists(config.weightsPath),
                   "Weights file does not exist: " + config.weightsPath);

        log(LogLevel::INFO, "All required files exist");
    }
};

// Test 2: 配置文件解析
class ConfigParsingTest : public TestCase {
public:
    ConfigParsingTest() : TestCase(
        "config_parsing",
        "解析模型配置文件并验证关键参数"
    ) {}

    void execute() override {
        ModelLoadingConfig config;

        log(LogLevel::INFO, "Opening config file...");
        std::ifstream f(config.configPath);
        assertTrue(f.is_open(), "Failed to open config file");

        log(LogLevel::INFO, "Parsing JSON...");
        nlohmann::json jsonConfig;
        try {
            f >> jsonConfig;
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("JSON parse error: ") + e.what());
        }

        // 验证关键字段
        log(LogLevel::INFO, "Checking required fields...");
        assertTrue(jsonConfig.contains("model_type"), "Missing model_type");
        assertTrue(jsonConfig.contains("hidden_size"), "Missing hidden_size");
        assertTrue(jsonConfig.contains("num_hidden_layers"), "Missing num_hidden_layers");
        assertTrue(jsonConfig.contains("num_attention_heads"), "Missing num_attention_heads");
        assertTrue(jsonConfig.contains("vocab_size"), "Missing vocab_size");

        // 记录配置信息
        std::string modelType = jsonConfig["model_type"];
        int hiddenSize = jsonConfig["hidden_size"];
        int numLayers = jsonConfig["num_hidden_layers"];
        int numHeads = jsonConfig["num_attention_heads"];
        int vocabSize = jsonConfig["vocab_size"];

        log(LogLevel::INFO, "Model type: " + modelType);
        log(LogLevel::INFO, "Hidden size: " + std::to_string(hiddenSize));
        log(LogLevel::INFO, "Num layers: " + std::to_string(numLayers));
        log(LogLevel::INFO, "Num heads: " + std::to_string(numHeads));
        log(LogLevel::INFO, "Vocab size: " + std::to_string(vocabSize));

        // 验证数值范围
        assertTrue(hiddenSize > 0 && hiddenSize <= 8192,
                   "Invalid hidden_size: " + std::to_string(hiddenSize));
        assertTrue(numLayers > 0 && numLayers <= 128,
                   "Invalid num_hidden_layers: " + std::to_string(numLayers));
        assertTrue(vocabSize > 0 && vocabSize <= 200000,
                   "Invalid vocab_size: " + std::to_string(vocabSize));

        log(LogLevel::INFO, "Config parsing successful");
    }
};

// Test 3: 权重文件验证
class WeightsValidationTest : public TestCase {
public:
    WeightsValidationTest() : TestCase(
        "weights_validation",
        "验证权重文件格式和大小"
    ) {}

    void execute() override {
        ModelLoadingConfig config;

        log(LogLevel::INFO, "Checking weights file size...");
        auto fileSize = fs::file_size(config.weightsPath);
        log(LogLevel::INFO, "File size: " + std::to_string(fileSize / (1024 * 1024)) + " MB");

        assertTrue(fileSize > 0, "Weights file is empty");
        assertTrue(fileSize > 100 * 1024 * 1024, "Weights file too small (< 100MB)");

        log(LogLevel::INFO, "Checking file format (safetensors header)...");
        std::ifstream f(config.weightsPath, std::ios::binary);
        assertTrue(f.is_open(), "Failed to open weights file");

        // 读取 safetensors 头部
        uint64_t headerLen;
        f.read(reinterpret_cast<char*>(&headerLen), sizeof(headerLen));

        assertTrue(f.gcount() == sizeof(headerLen), "Failed to read header length");
        assertTrue(headerLen > 0 && headerLen < 100 * 1024 * 1024,
                   "Invalid header length: " + std::to_string(headerLen));

        log(LogLevel::INFO, "Header length: " + std::to_string(headerLen));

        // 读取并解析头部 JSON
        std::string headerJson(headerLen, '\0');
        f.read(headerJson.data(), headerLen);

        try {
            auto header = nlohmann::json::parse(headerJson);
            log(LogLevel::INFO, "Number of tensors: " + std::to_string(header.size()));

            // 验证关键张量存在
            assertTrue(header.contains("model.embed_tokens.weight"),
                       "Missing embed_tokens.weight");
            assertTrue(header.contains("model.layers.0.self_attn.q_proj.weight"),
                       "Missing first layer q_proj");

            log(LogLevel::INFO, "Key tensors verified");
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Failed to parse header: ") + e.what());
        }

        log(LogLevel::INFO, "Weights validation successful");
    }
};

// Test 4: 量化配置测试
class QuantizationConfigTest : public TestCase {
public:
    QuantizationConfigTest() : TestCase(
        "quantization_config",
        "验证量化配置（如果使用）"
    ) {}

    void execute() override {
        ModelLoadingConfig config;

        // 检查是否存在量化配置
        std::string quantConfigPath = config.modelPath + "/quantize_config.json";

        if (!fs::exists(quantConfigPath)) {
            log(LogLevel::INFO, "No quantization config found (using full precision)");
            return;
        }

        log(LogLevel::INFO, "Found quantization config");
        std::ifstream f(quantConfigPath);
        assertTrue(f.is_open(), "Failed to open quantize_config.json");

        nlohmann::json quantConfig;
        try {
            f >> quantConfig;
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Failed to parse quant config: ") + e.what());
        }

        // 验证量化参数
        if (quantConfig.contains("bits")) {
            int bits = quantConfig["bits"];
            log(LogLevel::INFO, "Quantization bits: " + std::to_string(bits));
            assertTrue(bits == 4 || bits == 8, "Unsupported quantization bits");
        }

        if (quantConfig.contains("group_size")) {
            int groupSize = quantConfig["group_size"];
            log(LogLevel::INFO, "Group size: " + std::to_string(groupSize));
        }

        log(LogLevel::INFO, "Quantization config valid");
    }
};

// Test 5: 内存使用估计
class MemoryUsageEstimationTest : public TestCase {
public:
    MemoryUsageEstimationTest() : TestCase(
        "memory_usage_estimation",
        "估计模型加载后的内存使用"
    ) {}

    void execute() override {
        ModelLoadingConfig config;

        // 读取配置
        std::ifstream f(config.configPath);
        nlohmann::json jsonConfig;
        f >> jsonConfig;

        int hiddenSize = jsonConfig["hidden_size"];
        int numLayers = jsonConfig["num_hidden_layers"];
        int vocabSize = jsonConfig["vocab_size"];
        int intermediateSize = jsonConfig.contains("intermediate_size")
                               ? jsonConfig["intermediate_size"].get<int>()
                               : hiddenSize * 4;

        // 估计参数数量
        // Embedding: vocab_size * hidden_size
        size_t embedParams = static_cast<size_t>(vocabSize) * hiddenSize;

        // Each layer: 4 linear layers (Q, K, V, O) + 2 FFN layers
        // Attention: 4 * hidden_size * hidden_size
        // FFN: hidden_size * intermediate_size * 2
        size_t layerParams = 4ULL * hiddenSize * hiddenSize
                           + 2ULL * hiddenSize * intermediateSize;

        size_t totalParams = embedParams + numLayers * layerParams;

        // 估计内存使用 (FP32)
        size_t memoryBytes = totalParams * 4;

        log(LogLevel::INFO, "Estimated parameter count: " +
            std::to_string(totalParams / 1000000) + "M");
        log(LogLevel::INFO, "Estimated memory (FP32): " +
            std::to_string(memoryBytes / (1024 * 1024)) + " MB");

        // 检查实际文件大小
        size_t fileSize = fs::file_size(config.weightsPath);
        log(LogLevel::INFO, "Actual weights file size: " +
            std::to_string(fileSize / (1024 * 1024)) + " MB");

        // 验证文件大小合理
        // 如果使用了量化，文件大小应该远小于 FP32 估计
        if (fileSize < memoryBytes / 4) {
            log(LogLevel::INFO, "Quantization detected (file size < 25% of FP32)");
        }

        assertTrue(fileSize > 0, "Invalid file size");
        log(LogLevel::INFO, "Memory estimation complete");
    }
};

// 创建 Stage 1 测试套件
std::shared_ptr<TestSuite> createModelConfigTestSuite() {
    auto suite = std::make_shared<TestSuite>("Stage 1: Model Config Validation");

    suite->addTest(std::make_shared<ConfigFileExistenceTest>());
    suite->addTest(std::make_shared<ConfigParsingTest>());
    suite->addTest(std::make_shared<WeightsValidationTest>());
    suite->addTest(std::make_shared<QuantizationConfigTest>());
    suite->addTest(std::make_shared<MemoryUsageEstimationTest>());

    return suite;
}

} // namespace kylin_test
