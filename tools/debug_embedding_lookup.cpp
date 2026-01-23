/**
 * @file debug_embedding_lookup.cpp
 * @brief 调试 embedding 查找操作
 */

#include "cllm/kylin/gguf/context.h"
#include "cllm/kylin/gguf/loader.h"
#include "cllm/common/logger.h"
#include <ggml.h>
#include <ggml-cpu.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstring>

void printTensorInfo(const char* name, const ggml_tensor* t) {
    if (!t) {
        std::cout << name << ": null" << std::endl;
        return;
    }
    
    std::cout << name << ": type=" << t->type 
              << ", shape=[" << t->ne[0] << ", " << t->ne[1] << ", " << t->ne[2] << ", " << t->ne[3] << "]"
              << ", nb=[" << t->nb[0] << ", " << t->nb[1] << ", " << t->nb[2] << ", " << t->nb[3] << "]"
              << std::endl;
}

void printFirstN(const char* name, const float* data, size_t n) {
    std::cout << name << " first " << n << ": ";
    for (size_t i = 0; i < n; ++i) {
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
    std::cout << "Embedding Lookup Debug" << std::endl;
    std::cout << "Testing token: " << testToken << std::endl;
    std::cout << "========================================" << std::endl;
    
    // 加载 GGUF 模型
    cllm::kylin::GGUFLoader loader(modelPath);
    if (!loader.isValid()) {
        std::cerr << "Failed to load GGUF model" << std::endl;
        return 1;
    }
    
    auto config = loader.loadConfig();
    std::cout << "\nModel config:" << std::endl;
    std::cout << "  vocab_size: " << config.vocabSize << std::endl;
    std::cout << "  embedding_length: " << config.embeddingLength << std::endl;
    
    // 创建计算上下文
    size_t memSize = 512 * 1024 * 1024;  // 512 MB
    cllm::kylin::GGMLContext ctx(memSize, cllm::kylin::BackendType::CPU);
    
    // 加载 token embedding 张量
    std::map<std::string, ggml_tensor*> tensors;
    
    std::string embedName = "token_embd.weight";
    ggml_tensor* tokEmbed = loader.loadTensor(&ctx, embedName);
    if (!tokEmbed) {
        embedName = "model.embed_tokens.weight";
        tokEmbed = loader.loadTensor(&ctx, embedName);
    }
    
    if (!tokEmbed) {
        std::cerr << "Failed to load token embedding" << std::endl;
        return 1;
    }
    
    std::cout << "\n=== Token Embedding Tensor ===" << std::endl;
    printTensorInfo("token_embd", tokEmbed);
    
    // 检查量化类型
    std::cout << "  Quantization type: " << ggml_type_name(tokEmbed->type) << std::endl;
    
    // 创建输入张量
    std::cout << "\n=== Embedding Lookup Test ===" << std::endl;
    
    ggml_context* gctx = ctx.raw();
    
    ggml_tensor* inputTensor = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, 1);
    int32_t* inputData = static_cast<int32_t*>(inputTensor->data);
    inputData[0] = testToken;
    
    printTensorInfo("input", inputTensor);
    std::cout << "  Input token: " << inputData[0] << std::endl;
    
    // 执行 embedding 查找
    ggml_tensor* output = ggml_get_rows(gctx, tokEmbed, inputTensor);
    printTensorInfo("output", output);
    
    // 构建计算图
    ggml_cgraph* graph = ctx.buildGraph(output);
    std::cout << "\nGraph built: " << ggml_graph_n_nodes(graph) << " nodes" << std::endl;
    
    // 计算
    ctx.compute(graph);
    std::cout << "Computation done" << std::endl;
    
    // 检查输出
    if (output->type == GGML_TYPE_F32) {
        const float* outData = static_cast<const float*>(output->data);
        size_t outSize = output->ne[0];
        
        std::cout << "\nOutput embedding (should be for token " << testToken << "):" << std::endl;
        printFirstN("  Values", outData, 20);
        
        // 统计
        float min = outData[0], max = outData[0], sum = 0;
        for (size_t i = 0; i < outSize; ++i) {
            min = std::min(min, outData[i]);
            max = std::max(max, outData[i]);
            sum += outData[i];
        }
        std::cout << "  Stats: min=" << min << ", max=" << max << ", mean=" << sum/outSize << std::endl;
    } else {
        std::cout << "\nOutput is not F32! Type: " << output->type << std::endl;
    }
    
    // 检查 Q4_K 张量的内部结构
    std::cout << "\n=== Q4_K Tensor Info ===" << std::endl;
    
    if (tokEmbed->type == GGML_TYPE_Q4_K) {
        std::cout << "Token embedding is Q4_K" << std::endl;
        
        size_t embedDim = tokEmbed->ne[0];  // embedding dimension
        size_t vocabSize = tokEmbed->ne[1];  // vocab size
        
        std::cout << "  Embed dim: " << embedDim << ", vocab: " << vocabSize << std::endl;
        
        // 获取 row 的偏移
        size_t nb0 = tokEmbed->nb[0];  // bytes per element
        size_t nb1 = tokEmbed->nb[1];  // bytes per row
        
        std::cout << "  nb0 (bytes per element in Q4_K): " << nb0 << std::endl;
        std::cout << "  nb1 (bytes per row): " << nb1 << std::endl;
        std::cout << "  Offset for token " << testToken << ": " << testToken * nb1 << " bytes" << std::endl;
        std::cout << "  Total tensor size: " << ggml_nbytes(tokEmbed) << " bytes" << std::endl;
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Debug Complete" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
