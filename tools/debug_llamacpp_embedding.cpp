/**
 * @file debug_llamacpp_embedding.cpp
 * @brief 使用 llama.cpp 获取 embedding 输出
 */

#include <llama.h>
#include <ggml.h>
#include <iostream>
#include <iomanip>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <gguf_model_path>" << std::endl;
        return 1;
    }
    
    std::string modelPath = argv[1];
    int testToken = 9707;  // "Hello"
    
    std::cout << "========================================" << std::endl;
    std::cout << "llama.cpp Embedding Debug" << std::endl;
    std::cout << "Testing token: " << testToken << std::endl;
    std::cout << "========================================" << std::endl;
    
    llama_backend_init();
    
    llama_model_params modelParams = llama_model_default_params();
    modelParams.n_gpu_layers = 0;
    
    llama_model* model = llama_model_load_from_file(modelPath.c_str(), modelParams);
    if (!model) {
        std::cerr << "Failed to load model" << std::endl;
        llama_backend_free();
        return 1;
    }
    
    std::cout << "\nModel loaded" << std::endl;
    
    // 获取 token embedding
    // llama.cpp 提供了 llama_get_embeddings 但这是用于句子 embedding 的
    // 我们需要直接访问模型的 embedding 层
    
    // 创建上下文
    llama_context_params ctxParams = llama_context_default_params();
    ctxParams.n_ctx = 2048;
    ctxParams.n_batch = 512;
    ctxParams.embeddings = true;  // 启用 embeddings
    
    llama_context* ctx = llama_init_from_model(model, ctxParams);
    if (!ctx) {
        std::cerr << "Failed to create context" << std::endl;
        llama_model_free(model);
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
    
    // 推理
    std::cout << "Running llama_decode..." << std::endl;
    int result = llama_decode(ctx, batch);
    if (result != 0) {
        std::cerr << "llama_decode failed: " << result << std::endl;
    }
    
    // 获取 logits
    float* logits = llama_get_logits(ctx);
    if (logits) {
        std::cout << "\nLogits first 20: ";
        for (int i = 0; i < 20; ++i) {
            std::cout << std::fixed << std::setprecision(4) << logits[i] << " ";
        }
        std::cout << std::endl;
        
        // 统计
        int vocabSize = llama_vocab_n_tokens(llama_model_get_vocab(model));
        float min = logits[0], max = logits[0], sum = 0;
        for (int i = 0; i < vocabSize; ++i) {
            min = std::min(min, logits[i]);
            max = std::max(max, logits[i]);
            sum += logits[i];
        }
        std::cout << "Logits stats: min=" << min << ", max=" << max << ", mean=" << sum/vocabSize << std::endl;
    }
    
    // 清理
    delete[] batch.seq_id[0];
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Debug Complete" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
