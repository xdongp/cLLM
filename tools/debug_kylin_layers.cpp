/**
 * @file debug_kylin_layers.cpp
 * @brief 逐层调试 Kylin 模型输出
 * 
 * 比较 Kylin 各层输出与 llama.cpp 的差异，定位问题所在
 */

#include "llama.h"
#include "ggml.h"
#include "cllm/kylin/gguf/transformer.h"
#include "cllm/kylin/core/tensor_stats.h"
#include "cllm/common/logger.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace cllm;
using namespace cllm::kylin;

// 打印 top-k logits
void printTopK(const float* logits, size_t vocabSize, size_t k = 10) {
    std::vector<std::pair<float, int>> logitPairs;
    for (size_t i = 0; i < vocabSize; ++i) {
        logitPairs.emplace_back(logits[i], static_cast<int>(i));
    }
    std::partial_sort(logitPairs.begin(), logitPairs.begin() + k, logitPairs.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    
    std::cout << "  Top-" << k << " tokens:" << std::endl;
    for (size_t i = 0; i < k; ++i) {
        std::cout << "    " << std::setw(3) << i+1 << ". Token " << std::setw(6) << logitPairs[i].second 
                  << ": logit=" << std::fixed << std::setprecision(4) << logitPairs[i].first << std::endl;
    }
}

// 计算统计信息
void computeStats(const float* data, size_t n, float& minVal, float& maxVal, float& mean, float& stddev) {
    minVal = std::numeric_limits<float>::max();
    maxVal = std::numeric_limits<float>::lowest();
    double sum = 0.0;
    
    for (size_t i = 0; i < n; ++i) {
        minVal = std::min(minVal, data[i]);
        maxVal = std::max(maxVal, data[i]);
        sum += data[i];
    }
    mean = static_cast<float>(sum / n);
    
    double varSum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double diff = data[i] - mean;
        varSum += diff * diff;
    }
    stddev = static_cast<float>(std::sqrt(varSum / n));
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <gguf_model_path>" << std::endl;
        return 1;
    }
    
    std::string modelPath = argv[1];
    
    std::cout << "========================================" << std::endl;
    std::cout << "Kylin Layer-by-Layer Debug Tool" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Test token: "Hello" -> 9707
    const int32_t testToken = 9707;
    std::cout << "\nTest input: token " << testToken << " (\"Hello\")" << std::endl;
    
    // ========== Part 1: llama.cpp Reference ==========
    std::cout << "\n[Part 1] llama.cpp Reference Output" << std::endl;
    std::cout << "------------------------------------" << std::endl;
    
    ggml_backend_load_all();
    
    llama_model_params modelParams = llama_model_default_params();
    modelParams.n_gpu_layers = 0;  // CPU only for comparison
    
    llama_model* llamaModel = llama_model_load_from_file(modelPath.c_str(), modelParams);
    if (!llamaModel) {
        std::cerr << "Failed to load llama.cpp model" << std::endl;
        return 1;
    }
    
    llama_context_params ctxParams = llama_context_default_params();
    ctxParams.n_ctx = 512;
    ctxParams.n_batch = 512;
    ctxParams.n_seq_max = 1;
    
    llama_context* llamaCtx = llama_init_from_model(llamaModel, ctxParams);
    if (!llamaCtx) {
        std::cerr << "Failed to create llama.cpp context" << std::endl;
        llama_model_free(llamaModel);
        return 1;
    }
    
    // Run inference
    llama_batch batch = llama_batch_init(1, 0, 1);
    batch.token[0] = testToken;
    batch.pos[0] = 0;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = true;
    batch.n_tokens = 1;
    
    if (llama_decode(llamaCtx, batch) != 0) {
        std::cerr << "llama_decode failed" << std::endl;
        llama_batch_free(batch);
        llama_free(llamaCtx);
        llama_model_free(llamaModel);
        return 1;
    }
    
    const float* llamaLogits = llama_get_logits(llamaCtx);
    const struct llama_vocab* vocab = llama_model_get_vocab(llamaModel);
    int llamaVocabSize = llama_vocab_n_tokens(vocab);
    
    float llamaMin, llamaMax, llamaMean, llamaStd;
    computeStats(llamaLogits, llamaVocabSize, llamaMin, llamaMax, llamaMean, llamaStd);
    
    std::cout << "llama.cpp logits stats:" << std::endl;
    std::cout << "  vocab_size: " << llamaVocabSize << std::endl;
    std::cout << "  min: " << llamaMin << ", max: " << llamaMax << std::endl;
    std::cout << "  mean: " << llamaMean << ", stddev: " << llamaStd << std::endl;
    printTopK(llamaLogits, llamaVocabSize, 5);
    
    // Save llama.cpp logits for comparison
    std::vector<float> llamaLogitsCopy(llamaLogits, llamaLogits + llamaVocabSize);
    
    llama_batch_free(batch);
    llama_free(llamaCtx);
    llama_model_free(llamaModel);
    
    // ========== Part 2: Kylin Model ==========
    std::cout << "\n[Part 2] Kylin Model Output" << std::endl;
    std::cout << "------------------------------------" << std::endl;
    
    GGMLTransformerModel kylinModel;
    if (!kylinModel.loadFromGGUF(modelPath)) {
        std::cerr << "Failed to load Kylin model" << std::endl;
        return 1;
    }
    
    const auto& config = kylinModel.getConfig();
    std::cout << "Kylin config:" << std::endl;
    std::cout << "  architecture: " << config.architecture << std::endl;
    std::cout << "  vocab_size: " << config.vocabSize << std::endl;
    std::cout << "  embedding_length: " << config.embeddingLength << std::endl;
    std::cout << "  block_count: " << config.blockCount << std::endl;
    std::cout << "  head_count: " << config.headCount << std::endl;
    std::cout << "  head_count_kv: " << config.headCountKV << std::endl;
    std::cout << "  head_dim: " << config.headDim() << std::endl;
    std::cout << "  rms_norm_eps: " << std::scientific << config.rmsNormEps << std::endl;
    std::cout << "  rope_freq_base: " << std::fixed << config.ropeFreqBase << std::endl;
    
    // Run Kylin inference
    std::vector<int32_t> inputIds = {testToken};
    auto kylinLogits = kylinModel.forward(inputIds);
    
    size_t kylinVocabSize = config.vocabSize;
    
    // 打印 embedding 的实际值（前 10 个元素）
    std::cout << "\nKylin embedding (first 10 elements of hidden state after embedding lookup):" << std::endl;
    if (kylinLogits.size() >= 10) {
        // 注意：forward 返回的是 logits，不是 embedding
        // 我们需要检查模型的调试接口
    }
    
    float kylinMin, kylinMax, kylinMean, kylinStd;
    computeStats(kylinLogits.data(), kylinVocabSize, kylinMin, kylinMax, kylinMean, kylinStd);
    
    std::cout << "\nKylin logits stats:" << std::endl;
    std::cout << "  vocab_size: " << kylinVocabSize << std::endl;
    std::cout << "  min: " << kylinMin << ", max: " << kylinMax << std::endl;
    std::cout << "  mean: " << kylinMean << ", stddev: " << kylinStd << std::endl;
    printTopK(kylinLogits.data(), kylinVocabSize, 5);
    
    // ========== Part 3: Comparison ==========
    std::cout << "\n[Part 3] Comparison" << std::endl;
    std::cout << "------------------------------------" << std::endl;
    
    // Compare logits
    float maxDiff = 0.0f;
    float sumDiff = 0.0f;
    size_t maxDiffIdx = 0;
    
    size_t minVocab = std::min(static_cast<size_t>(llamaVocabSize), kylinVocabSize);
    for (size_t i = 0; i < minVocab; ++i) {
        float diff = std::abs(llamaLogitsCopy[i] - kylinLogits[i]);
        sumDiff += diff;
        if (diff > maxDiff) {
            maxDiff = diff;
            maxDiffIdx = i;
        }
    }
    
    std::cout << "Logits comparison:" << std::endl;
    std::cout << "  Max diff: " << maxDiff << " at token " << maxDiffIdx << std::endl;
    std::cout << "  Mean diff: " << sumDiff / minVocab << std::endl;
    std::cout << "  llama.cpp[" << maxDiffIdx << "]: " << llamaLogitsCopy[maxDiffIdx] << std::endl;
    std::cout << "  Kylin[" << maxDiffIdx << "]: " << kylinLogits[maxDiffIdx] << std::endl;
    
    // Check Layer 0 debug stats
    std::cout << "\n[Part 4] Kylin Layer 0 Debug Stats" << std::endl;
    std::cout << "------------------------------------" << std::endl;
    
    auto layer0Stats = kylinModel.getLayer0DebugStats();
    for (const auto& [name, stats] : layer0Stats) {
        if (stats.isValid()) {
            std::cout << "  " << name << ": min=" << std::setprecision(6) << stats.minVal 
                      << ", max=" << stats.maxVal << ", mean=" << stats.mean 
                      << ", std=" << stats.stddev << std::endl;
        }
    }
    
    // Final Norm and Logits stats
    auto finalNormStats = kylinModel.getFinalNormStats();
    auto logitsStats = kylinModel.getLogitsStats();
    
    std::cout << "\n  Final norm: min=" << finalNormStats.minVal << ", max=" << finalNormStats.maxVal 
              << ", mean=" << finalNormStats.mean << ", std=" << finalNormStats.stddev << std::endl;
    std::cout << "  Logits tensor: min=" << logitsStats.minVal << ", max=" << logitsStats.maxVal 
              << ", mean=" << logitsStats.mean << ", std=" << logitsStats.stddev << std::endl;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Debug Complete" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
