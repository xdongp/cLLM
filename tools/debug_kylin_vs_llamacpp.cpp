/**
 * @file debug_kylin_vs_llamacpp.cpp
 * @brief 详细比较 Kylin 和 llama.cpp 的中间计算结果
 */

#include "cllm/kylin/gguf/transformer.h"
#include "cllm/kylin/gguf/loader.h"
#include "cllm/kylin/gguf/context.h"
#include <llama.h>
#include <ggml.h>
#include <ggml-cpu.h>
#include <iostream>
#include <iomanip>
#include <cmath>

void printTensorStats(const char* name, const float* data, size_t size) {
    if (!data || size == 0) return;
    
    float min = data[0], max = data[0], sum = 0;
    int nanCount = 0, infCount = 0;
    
    for (size_t i = 0; i < size; ++i) {
        if (std::isnan(data[i])) { ++nanCount; continue; }
        if (std::isinf(data[i])) { ++infCount; continue; }
        min = std::min(min, data[i]);
        max = std::max(max, data[i]);
        sum += data[i];
    }
    
    std::cout << "  " << name << ": size=" << size 
              << ", min=" << std::fixed << std::setprecision(6) << min 
              << ", max=" << max 
              << ", mean=" << (sum / size)
              << ", nan=" << nanCount << ", inf=" << infCount << std::endl;
    
    // 打印前 10 个值
    std::cout << "    First 10: ";
    for (size_t i = 0; i < std::min(size, size_t(10)); ++i) {
        std::cout << std::fixed << std::setprecision(4) << data[i] << " ";
    }
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <gguf_model_path>" << std::endl;
        return 1;
    }
    
    std::string modelPath = argv[1];
    int testToken = 9707;  // "Hello"
    
    std::cout << "========================================" << std::endl;
    std::cout << "Kylin vs llama.cpp Comparison" << std::endl;
    std::cout << "Testing token: " << testToken << " (Hello)" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // ========== Part 1: Kylin ==========
    std::cout << "\n[Part 1: Kylin Backend]" << std::endl;
    
    cllm::kylin::GGMLTransformerModel kylinModel;
    if (!kylinModel.loadFromGGUF(modelPath)) {
        std::cerr << "Failed to load Kylin model" << std::endl;
        return 1;
    }
    
    auto kylinConfig = kylinModel.getConfig();
    std::cout << "  Vocab size: " << kylinConfig.vocabSize << std::endl;
    std::cout << "  Hidden size: " << kylinConfig.embeddingLength << std::endl;
    
    kylinModel.clearKVCache();
    auto kylinLogits = kylinModel.forward({testToken});
    
    std::cout << "\n  Kylin Logits:" << std::endl;
    printTensorStats("logits", kylinLogits.data(), std::min(kylinLogits.size(), size_t(10000)));
    
    // 找 top-5
    std::vector<std::pair<float, int>> kylinScored;
    for (size_t i = 0; i < kylinConfig.vocabSize && i < kylinLogits.size(); ++i) {
        kylinScored.push_back({kylinLogits[i], static_cast<int>(i)});
    }
    std::partial_sort(kylinScored.begin(), kylinScored.begin() + 5, kylinScored.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    
    std::cout << "\n  Kylin Top-5:" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << "    " << (i+1) << ". Token " << std::setw(6) << kylinScored[i].second 
                  << " | logit=" << std::fixed << std::setprecision(4) << kylinScored[i].first << std::endl;
    }
    
    // ========== Part 2: llama.cpp ==========
    std::cout << "\n[Part 2: llama.cpp Backend]" << std::endl;
    
    llama_backend_init();
    
    llama_model_params modelParams = llama_model_default_params();
    modelParams.n_gpu_layers = 0;
    
    llama_model* llamaModel = llama_model_load_from_file(modelPath.c_str(), modelParams);
    if (!llamaModel) {
        std::cerr << "Failed to load llama.cpp model" << std::endl;
        llama_backend_free();
        return 1;
    }
    
    int vocabSize = llama_vocab_n_tokens(llama_model_get_vocab(llamaModel));
    std::cout << "  Vocab size: " << vocabSize << std::endl;
    
    llama_context_params ctxParams = llama_context_default_params();
    ctxParams.n_ctx = 2048;
    ctxParams.n_batch = 512;
    
    llama_context* ctx = llama_init_from_model(llamaModel, ctxParams);
    if (!ctx) {
        std::cerr << "Failed to create llama.cpp context" << std::endl;
        llama_model_free(llamaModel);
        llama_backend_free();
        return 1;
    }
    
    // 创建 batch
    llama_batch batch = llama_batch_init(512, 0, 1);
    batch.n_tokens = 1;
    batch.token[0] = testToken;
    batch.pos[0] = 0;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0] = new llama_seq_id[1];
    batch.seq_id[0][0] = 0;
    batch.logits[0] = 1;
    
    int result = llama_decode(ctx, batch);
    if (result != 0) {
        std::cerr << "llama_decode failed: " << result << std::endl;
    }
    
    float* llamaLogits = llama_get_logits(ctx);
    
    std::cout << "\n  llama.cpp Logits:" << std::endl;
    printTensorStats("logits", llamaLogits, std::min(size_t(vocabSize), size_t(10000)));
    
    // 找 top-5
    std::vector<std::pair<float, int>> llamaScored;
    for (int i = 0; i < vocabSize; ++i) {
        llamaScored.push_back({llamaLogits[i], i});
    }
    std::partial_sort(llamaScored.begin(), llamaScored.begin() + 5, llamaScored.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    
    std::cout << "\n  llama.cpp Top-5:" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << "    " << (i+1) << ". Token " << std::setw(6) << llamaScored[i].second 
                  << " | logit=" << std::fixed << std::setprecision(4) << llamaScored[i].first << std::endl;
    }
    
    // ========== Part 3: Comparison ==========
    std::cout << "\n[Part 3: Comparison]" << std::endl;
    
    // 比较 logits
    float maxDiff = 0;
    int maxDiffIdx = 0;
    size_t compareSize = std::min(static_cast<size_t>(vocabSize), kylinLogits.size());
    
    for (size_t i = 0; i < compareSize; ++i) {
        float diff = std::abs(kylinLogits[i] - llamaLogits[i]);
        if (diff > maxDiff) {
            maxDiff = diff;
            maxDiffIdx = static_cast<int>(i);
        }
    }
    
    std::cout << "  Max logit diff: " << maxDiff << " at token " << maxDiffIdx << std::endl;
    std::cout << "    Kylin: " << kylinLogits[maxDiffIdx] << std::endl;
    std::cout << "    llama.cpp: " << llamaLogits[maxDiffIdx] << std::endl;
    
    // 比较 top token
    bool topMatch = (kylinScored[0].second == llamaScored[0].second);
    std::cout << "\n  Top token match: " << (topMatch ? "YES" : "NO") << std::endl;
    std::cout << "    Kylin top: " << kylinScored[0].second << " (logit=" << kylinScored[0].first << ")" << std::endl;
    std::cout << "    llama.cpp top: " << llamaScored[0].second << " (logit=" << llamaScored[0].first << ")" << std::endl;
    
    // 清理
    delete[] batch.seq_id[0];
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(llamaModel);
    llama_backend_free();
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Comparison Complete" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
