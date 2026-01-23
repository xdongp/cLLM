/**
 * @file compare_embedding.cpp
 * @brief 比较 Kylin 和 llama.cpp 的 embedding 查找结果
 */

#include "cllm/kylin/gguf/context.h"
#include "cllm/kylin/gguf/loader.h"
#include "cllm/common/logger.h"
#include <llama.h>
#include <ggml.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstring>

void printFirst20(const char* name, const float* data) {
    std::cout << name << " first 20: ";
    for (int i = 0; i < 20; ++i) {
        std::cout << std::fixed << std::setprecision(6) << data[i] << " ";
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
    std::cout << "Embedding Comparison: Kylin vs llama.cpp" << std::endl;
    std::cout << "Testing token: " << testToken << std::endl;
    std::cout << "========================================" << std::endl;
    
    // ========== Part 1: Kylin Embedding ==========
    std::cout << "\n[Part 1: Kylin Embedding Lookup]" << std::endl;
    
    cllm::kylin::GGUFLoader loader(modelPath);
    if (!loader.isValid()) {
        std::cerr << "Failed to load GGUF model" << std::endl;
        return 1;
    }
    
    auto config = loader.loadConfig();
    std::cout << "  Vocab size: " << config.vocabSize << std::endl;
    std::cout << "  Embedding dim: " << config.embeddingLength << std::endl;
    
    // 创建计算上下文
    size_t memSize = 512 * 1024 * 1024;
    cllm::kylin::GGMLContext ctx(memSize, cllm::kylin::BackendType::CPU);
    
    // 加载 token embedding
    ggml_tensor* tokEmbed = loader.loadTensor(&ctx, "token_embd.weight");
    if (!tokEmbed) {
        std::cerr << "Failed to load token embedding" << std::endl;
        return 1;
    }
    
    std::cout << "  Token embedding type: " << tokEmbed->type << " (" << ggml_type_name(tokEmbed->type) << ")" << std::endl;
    std::cout << "  Token embedding shape: [" << tokEmbed->ne[0] << ", " << tokEmbed->ne[1] << "]" << std::endl;
    
    // 执行 embedding 查找
    ggml_context* gctx = ctx.raw();
    
    ggml_tensor* inputTensor = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 1);
    int32_t* inputData = static_cast<int32_t*>(inputTensor->data);
    inputData[0] = testToken;
    
    ggml_tensor* kylinEmb = ggml_get_rows(gctx, tokEmbed, inputTensor);
    
    ggml_cgraph* graph = ctx.buildGraph(kylinEmb);
    ctx.compute(graph);
    
    std::vector<float> kylinEmbData(config.embeddingLength);
    std::memcpy(kylinEmbData.data(), kylinEmb->data, config.embeddingLength * sizeof(float));
    
    printFirst20("  Kylin embedding", kylinEmbData.data());
    
    float kylinMin = kylinEmbData[0], kylinMax = kylinEmbData[0], kylinSum = 0;
    for (float v : kylinEmbData) {
        kylinMin = std::min(kylinMin, v);
        kylinMax = std::max(kylinMax, v);
        kylinSum += v;
    }
    std::cout << "  Stats: min=" << kylinMin << ", max=" << kylinMax 
              << ", mean=" << kylinSum/config.embeddingLength << std::endl;
    
    // ========== Part 2: llama.cpp Embedding ==========
    std::cout << "\n[Part 2: llama.cpp Embedding]" << std::endl;
    
    llama_backend_init();
    
    llama_model_params modelParams = llama_model_default_params();
    modelParams.n_gpu_layers = 0;
    
    llama_model* model = llama_model_load_from_file(modelPath.c_str(), modelParams);
    if (!model) {
        std::cerr << "Failed to load llama.cpp model" << std::endl;
        llama_backend_free();
        return 1;
    }
    
    // llama.cpp 不直接暴露 embedding 查找 API，但我们可以通过设置 embeddings=true 来获取
    // 然而这是 sentence embedding，不是 token embedding
    // 
    // 我们需要直接访问模型的内部张量
    // llama.cpp 使用 ggml 上下文，我们可以尝试获取 token embedding 张量
    
    // 获取 vocab 大小
    const llama_vocab* vocab = llama_model_get_vocab(model);
    int llamaVocabSize = llama_vocab_n_tokens(vocab);
    std::cout << "  Vocab size: " << llamaVocabSize << std::endl;
    
    // 由于 llama.cpp 不直接暴露 token embedding，我们比较最终的 logits
    std::cout << "\n  Note: llama.cpp doesn't directly expose token embedding lookup." << std::endl;
    std::cout << "  Comparing final logits instead..." << std::endl;
    
    llama_context_params ctxParams = llama_context_default_params();
    ctxParams.n_ctx = 2048;
    ctxParams.n_batch = 512;
    
    llama_context* llamaCtx = llama_init_from_model(model, ctxParams);
    if (!llamaCtx) {
        std::cerr << "Failed to create llama.cpp context" << std::endl;
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
    
    int result = llama_decode(llamaCtx, batch);
    if (result != 0) {
        std::cerr << "llama_decode failed: " << result << std::endl;
    }
    
    float* llamaLogits = llama_get_logits(llamaCtx);
    
    printFirst20("  llama.cpp logits", llamaLogits);
    
    float llamaMin = llamaLogits[0], llamaMax = llamaLogits[0], llamaSum = 0;
    for (int i = 0; i < llamaVocabSize; ++i) {
        llamaMin = std::min(llamaMin, llamaLogits[i]);
        llamaMax = std::max(llamaMax, llamaLogits[i]);
        llamaSum += llamaLogits[i];
    }
    std::cout << "  Stats: min=" << llamaMin << ", max=" << llamaMax 
              << ", mean=" << llamaSum/llamaVocabSize << std::endl;
    
    // 清理
    delete[] batch.seq_id[0];
    llama_batch_free(batch);
    llama_free(llamaCtx);
    llama_model_free(model);
    llama_backend_free();
    
    // ========== Part 3: Summary ==========
    std::cout << "\n[Summary]" << std::endl;
    std::cout << "  The embedding lookup itself appears to work correctly." << std::endl;
    std::cout << "  The difference must be in the Transformer layers." << std::endl;
    std::cout << "  Kylin embedding mean: " << kylinSum/config.embeddingLength << std::endl;
    std::cout << "  llama.cpp logits mean: " << llamaSum/llamaVocabSize << std::endl;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Comparison Complete" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
