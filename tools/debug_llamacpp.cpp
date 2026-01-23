/**
 * @file debug_llamacpp.cpp
 * @brief 直接使用 llama.cpp API 进行推理调试
 */

#include <llama.h>
#include <ggml.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cmath>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <gguf_model_path>" << std::endl;
        return 1;
    }
    
    std::string modelPath = argv[1];
    
    std::cout << "========================================" << std::endl;
    std::cout << "LLama.cpp Direct Debug Tool" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // 初始化 llama.cpp 后端
    llama_backend_init();
    
    // 加载模型
    llama_model_params modelParams = llama_model_default_params();
    modelParams.n_gpu_layers = 0;  // CPU only
    
    std::cout << "\nLoading model..." << std::endl;
    llama_model* model = llama_model_load_from_file(modelPath.c_str(), modelParams);
    if (!model) {
        std::cerr << "Failed to load model" << std::endl;
        llama_backend_free();
        return 1;
    }
    
    int vocabSize = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::cout << "  Vocab size: " << vocabSize << std::endl;
    
    // 创建上下文
    llama_context_params ctxParams = llama_context_default_params();
    ctxParams.n_ctx = 2048;
    ctxParams.n_batch = 512;
    
    llama_context* ctx = llama_init_from_model(model, ctxParams);
    if (!ctx) {
        std::cerr << "Failed to create context" << std::endl;
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    
    // 测试 token 9707 (Hello)
    std::vector<llama_token> inputTokens = {9707};
    std::cout << "\n=== Test: Forward token 9707 (Hello) ===" << std::endl;
    
    // 创建 batch
    llama_batch batch = llama_batch_init(512, 0, 1);
    batch.n_tokens = static_cast<int32_t>(inputTokens.size());
    
    for (size_t i = 0; i < inputTokens.size(); ++i) {
        batch.token[i] = inputTokens[i];
        batch.pos[i] = static_cast<llama_pos>(i);
        batch.n_seq_id[i] = 1;
        batch.seq_id[i] = new llama_seq_id[1];
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == inputTokens.size() - 1);  // 只计算最后一个 token 的 logits
    }
    
    // 推理
    std::cout << "  Running llama_decode..." << std::endl;
    int result = llama_decode(ctx, batch);
    if (result != 0) {
        std::cerr << "  llama_decode failed with code: " << result << std::endl;
    } else {
        std::cout << "  llama_decode succeeded" << std::endl;
    }
    
    // 获取 logits
    float* logitsPtr = llama_get_logits(ctx);
    if (!logitsPtr) {
        std::cerr << "  Failed to get logits" << std::endl;
    } else {
        // 分析 logits
        std::vector<std::pair<float, int>> scored;
        for (int i = 0; i < vocabSize; ++i) {
            scored.push_back({logitsPtr[i], i});
        }
        std::partial_sort(scored.begin(), scored.begin() + 10, scored.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
        
        std::cout << "\n  Top-10 tokens:" << std::endl;
        for (int i = 0; i < 10; ++i) {
            std::cout << "    " << std::setw(2) << (i + 1) << ". Token " << std::setw(6) << scored[i].second 
                      << " | logit=" << std::fixed << std::setprecision(4) << scored[i].first << std::endl;
        }
        
        // 统计
        float maxLogit = -std::numeric_limits<float>::infinity();
        float minLogit = std::numeric_limits<float>::infinity();
        float sumLogit = 0.0f;
        
        for (int i = 0; i < vocabSize; ++i) {
            if (!std::isnan(logitsPtr[i]) && !std::isinf(logitsPtr[i])) {
                maxLogit = std::max(maxLogit, logitsPtr[i]);
                minLogit = std::min(minLogit, logitsPtr[i]);
                sumLogit += logitsPtr[i];
            }
        }
        
        std::cout << "\n  Logits statistics:" << std::endl;
        std::cout << "    Max: " << maxLogit << std::endl;
        std::cout << "    Min: " << minLogit << std::endl;
        std::cout << "    Mean: " << (sumLogit / vocabSize) << std::endl;
    }
    
    // 清理
    for (int i = 0; i < batch.n_tokens; ++i) {
        if (batch.seq_id[i]) {
            delete[] batch.seq_id[i];
        }
    }
    llama_batch_free(batch);
    
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Debug Complete" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
