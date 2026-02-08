#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <cllm/kylin/hf/transformer.h>
#include <cllm/tokenizer/hf_tokenizer.h>
#include <cllm/common/logger.h>
#include <spdlog/spdlog.h>

using namespace cllm;

std::string modelPath = "/Users/dannypan/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B";

void loadConfigFromFile(const std::string& modelPath, cllm::kylin::HFModelConfig& config) {
    std::ifstream configFile(modelPath + "/config.json");
    if (!configFile.is_open()) {
        throw std::runtime_error("无法打开 config.json");
    }

    try {
        nlohmann::json configJson = nlohmann::json::parse(configFile);
        config.architecture = configJson.value("architecture", "");
        config.modelType = configJson.value("model_type", "");
        config.torchDtype = configJson.value("torch_dtype", "bfloat16");
        config.hiddenSize = configJson.value("hidden_size", 0);
        config.numHiddenLayers = configJson.value("num_hidden_layers", 0);
        config.numAttentionHeads = configJson.value("num_attention_heads", 0);
        config.numKeyValueHeads = configJson.value("num_key_value_heads", 0);
        config.intermediateSize = configJson.value("intermediate_size", 0);
        config.vocabSize = configJson.value("vocab_size", 0);
        config.headDim = configJson.value("head_dim", 0);
        config.maxPositionEmbeddings = configJson.value("max_position_embeddings", 0);
        config.rmsNormEps = configJson.value("rms_norm_eps", 1e-6f);
        config.ropeTheta = configJson.value("rope_theta", 10000.0f);
        config.tieWordEmbeddings = configJson.value("tie_word_embeddings", false);
        config.attentionBias = configJson.value("attention_bias", false);
        config.hiddenAct = configJson.value("hidden_act", "silu");
        config.bosTokenId = configJson.value("bos_token_id", 0);
        config.eosTokenId = configJson.value("eos_token_id", 0);
        config.padTokenId = configJson.value("pad_token_id", -1);
    } catch (const std::exception& e) {
        throw std::runtime_error("配置解析失败: " + std::string(e.what()));
    }
}

void printConfig(const cllm::kylin::HFModelConfig& config) {
    std::cout << "✅ 模型配置加载成功" << std::endl;
    std::cout << "   - 模型类型: " << config.modelType << std::endl;
    std::cout << "   - Vocab Size: " << config.vocabSize << std::endl;
    std::cout << "   - Hidden Size: " << config.hiddenSize << std::endl;
    std::cout << "   - Layers: " << config.numHiddenLayers << std::endl;
    std::cout << "   - Heads: " << config.numAttentionHeads << std::endl;
}

// 辅助函数：应用 temperature 和 repetition penalty
void applySamplingParams(std::vector<float>& logits, float temperature, float repetitionPenalty, 
                         const std::vector<int>& generatedTokens, int lastToken) {
    // 应用 temperature
    if (temperature != 1.0f && temperature > 0.0f) {
        for (auto& logit : logits) {
            logit /= temperature;
        }
    }
    
    // 应用 repetition penalty
    if (repetitionPenalty > 1.0f) {
        // 对已经生成的 token 进行惩罚
        for (int tokenId : generatedTokens) {
            if (tokenId >= 0 && tokenId < (int)logits.size()) {
                if (logits[tokenId] > 0) {
                    logits[tokenId] /= repetitionPenalty;
                } else {
                    logits[tokenId] *= repetitionPenalty;
                }
            }
        }
        // 也对最后一个 token 进行惩罚
        if (lastToken >= 0 && lastToken < (int)logits.size()) {
            if (logits[lastToken] > 0) {
                logits[lastToken] /= repetitionPenalty;
            } else {
                logits[lastToken] *= repetitionPenalty;
            }
        }
    }
}

void runGreedyGeneration(HFTokenizer& tokenizer, cllm::kylin::HFTransformerModel& transformer, 
                         const std::string& prompt, int maxTokens = 30,
                         float temperature = 1.0f, float repetitionPenalty = 1.0f) {
    std::cout << std::endl;
    std::cout << "═══════════════════════════════════════════════════════════════════════════════════════════" << std::endl;
    std::cout << "测试提示词: \"" << prompt << "\"" << std::endl;
    std::cout << "采样参数: temperature=" << temperature << ", repetition_penalty=" << repetitionPenalty << std::endl;
    std::cout << "═══════════════════════════════════════════════════════════════════════════════════════════" << std::endl;

    // 重置 KV Cache，确保每次测试都是独立的状态
    transformer.releaseKVCache(0);

    std::vector<int> inputIds = tokenizer.encode(prompt, false);
    std::cout << "📝 输入 Tokens: " << inputIds.size() << std::endl;
    std::cout << "   Tokens: [";
    for (size_t i = 0; i < std::min(inputIds.size(), (size_t)10); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << inputIds[i];
    }
    if (inputIds.size() > 10) std::cout << " ...";
    std::cout << "]" << std::endl;

    std::vector<int> generatedTokens;

    std::cout << std::endl;
    std::cout << "🚀 开始生成..." << std::endl;

    int eosId = tokenizer.getEosId();
    std::cout << "   EOS Token ID: " << eosId << std::endl;

    // 首先处理整个提示词，填充 KV Cache
    std::cout << "   处理提示词..." << std::endl;
    for (size_t i = 0; i < inputIds.size(); ++i) {
        std::vector<float> logits = transformer.forward({inputIds[i]});
        if (i == inputIds.size() - 1) {
            // 最后一个 token 的 logits 用于生成下一个 token
            std::cout << "   提示词处理完成，开始生成..." << std::endl;
        }
    }

    // 获取最后一个 token 用于生成
    int currentToken = inputIds.back();

    // 开始计时纯生成时间
    auto generationStartTime = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < maxTokens; ++step) {
        std::vector<float> logits = transformer.forward({currentToken});
        
        // 应用 sampling 参数
        applySamplingParams(logits, temperature, repetitionPenalty, generatedTokens, currentToken);

        int nextToken = 0;
        float maxVal = logits[0];
        for (size_t i = 1; i < logits.size(); ++i) {
            if (logits[i] > maxVal) {
                maxVal = logits[i];
                nextToken = i;
            }
        }

        generatedTokens.push_back(nextToken);

        std::string tokenName = tokenizer.idToToken(nextToken);
        std::cout << "   Step " << std::setw(2) << (step + 1) << ": Token " << std::setw(6) << nextToken
                  << " (\"" << tokenName << "\")"
                  << " logit=" << std::fixed << std::setprecision(2) << maxVal;

        if (nextToken == eosId) {
            std::cout << " [EOS]";
        } else if (nextToken == 151668) {
            std::cout << " [<|im_end|>]";
        }

        std::cout << std::endl;

        if (nextToken == eosId || nextToken == 151668) {
            std::cout << "   → 遇到停止 token，停止生成" << std::endl;
            break;
        }

        currentToken = nextToken;
    }

    auto generationEndTime = std::chrono::high_resolution_clock::now();
    auto generationDuration = std::chrono::duration_cast<std::chrono::milliseconds>(generationEndTime - generationStartTime);
    float generationSeconds = generationDuration.count() / 1000.0f;
    float tokensPerSecond = generatedTokens.size() / generationSeconds;

    std::cout << std::endl;
    std::cout << "📊 生成结果统计:" << std::endl;
    std::cout << "   - 生成的 Token 数量: " << generatedTokens.size() << std::endl;
    std::cout << "   - 纯生成时间: " << std::fixed << std::setprecision(3) << generationSeconds << "s" << std::endl;
    std::cout << "   - 生成吞吐量: " << std::fixed << std::setprecision(2) << tokensPerSecond << " tokens/s" << std::endl;
    std::cout << "   - Tokens: [";
    for (size_t i = 0; i < generatedTokens.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << generatedTokens[i];
    }
    std::cout << "]" << std::endl;

    std::string decodedSkipSpecial = tokenizer.decode(generatedTokens, true);
    std::cout << std::endl;
    std::cout << "📝 解码结果 (skipSpecial=true):" << std::endl;
    std::cout << "   \"" << decodedSkipSpecial << "\"" << std::endl;

    std::string decodedWithSpecial = tokenizer.decode(generatedTokens, false);
    std::cout << std::endl;
    std::cout << "📝 解码结果 (skipSpecial=false):" << std::endl;
    std::cout << "   \"" << decodedWithSpecial << "\"" << std::endl;

    std::cout << std::endl;
    std::cout << "──────────────────────────────────────────────────────────────────────────────────────────────" << std::endl;
}

int main(int argc, char** argv) {
    std::string modelPath = "/Users/dannypan/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B";
    std::string deviceType = "cpu";
    std::string inputText = "hello";
    std::string quantTypeStr = "fp32";
    int maxTokens = 30;
    float temperature = 1.0f;
    float repetitionPenalty = 1.0f;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            modelPath = argv[++i];
        } else if (arg == "--device" && i + 1 < argc) {
            deviceType = argv[++i];
        } else if (arg == "--input" && i + 1 < argc) {
            inputText = argv[++i];
        } else if (arg == "--max_tokens" && i + 1 < argc) {
            maxTokens = std::stoi(argv[++i]);
        } else if (arg == "--temperature" && i + 1 < argc) {
            temperature = std::stof(argv[++i]);
        } else if (arg == "--repetition_penalty" && i + 1 < argc) {
            repetitionPenalty = std::stof(argv[++i]);
        } else if (arg == "--quant" && i + 1 < argc) {
            quantTypeStr = argv[++i];
        }
    }

    cllm::Logger::instance().setLevel(spdlog::level::info);

    std::cout << "╔════════════════════════════════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║          cLLM 模型输出分析 - 直接推理测试                                                ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════════════════════════════════════════╝" << std::endl;
    std::cout << std::endl;
    std::cout << "模型路径: " << modelPath << std::endl;
    std::cout << "设备类型: " << deviceType << std::endl;

    std::cout << std::endl << "【Step 1】加载模型配置..." << std::endl;
    cllm::kylin::HFModelConfig config;
    loadConfigFromFile(modelPath, config);
    printConfig(config);

    std::cout << std::endl << "【Step 2】加载 Tokenizer..." << std::endl;
    HFTokenizer tokenizer(ModelType::QWEN);
    if (!tokenizer.load(modelPath)) {
        std::cerr << "❌ Tokenizer 加载失败" << std::endl;
        return 1;
    }
    std::cout << "✅ Tokenizer 加载成功" << std::endl;
    std::cout << "   - Vocab Size: " << tokenizer.getVocabSize() << std::endl;
    std::cout << "   - BOS ID: " << tokenizer.getBosId() << std::endl;
    std::cout << "   - EOS ID: " << tokenizer.getEosId() << std::endl;

    std::cout << std::endl << "【Step 3】加载 Transformer 模型..." << std::endl;
    cllm::kylin::DeviceType device = (deviceType == "gpu") ? cllm::kylin::DeviceType::Metal : cllm::kylin::DeviceType::CPU;
    
    // 解析量化类型
    cllm::kylin::QuantType quantType = cllm::kylin::QuantType::FP32;
    if (quantTypeStr == "int8") {
        quantType = cllm::kylin::QuantType::INT8;
    } else if (quantTypeStr == "fp16") {
        quantType = cllm::kylin::QuantType::FP16;
    }
    std::cout << "   - 量化类型: " << quantTypeStr << std::endl;
    
    cllm::kylin::HFTransformerModel transformer(modelPath, device, quantType);
    if (!transformer.isLoaded()) {
        std::cerr << "❌ Transformer 模型加载失败" << std::endl;
        return 1;
    }
    std::cout << "✅ Transformer 模型加载成功" << std::endl;

    runGreedyGeneration(tokenizer, transformer, inputText, maxTokens, temperature, repetitionPenalty);

    std::cout << std::endl;
    std::cout << "╔════════════════════════════════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║                         测试完成                                                         ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════════════════════════════════════════╝" << std::endl;

    return 0;
}
