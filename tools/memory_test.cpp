/**
 * @file memory_test.cpp
 * @brief 内存占用测试工具
 */

#include <iostream>
#include <chrono>
#include <thread>
#include <unistd.h>
#include <sys/resource.h>
#include <cllm/kylin/hf/transformer.h>
#include <cllm/tokenizer/hf_tokenizer.h>

using namespace cllm;

// 获取当前进程的内存使用（MB）
size_t getCurrentMemoryMB() {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        // ru_maxrss 在 macOS 上是字节，在 Linux 上是 KB
        #ifdef __APPLE__
            return usage.ru_maxrss / (1024 * 1024);
        #else
            return usage.ru_maxrss / 1024;
        #endif
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <fp32|fp16|int8>" << std::endl;
        return 1;
    }

    std::string modelPath = argv[1];
    std::string quantStr = argv[2];

    cllm::kylin::QuantType quantType = cllm::kylin::QuantType::FP32;
    if (quantStr == "int8") {
        quantType = cllm::kylin::QuantType::INT8;
    } else if (quantStr == "fp16") {
        quantType = cllm::kylin::QuantType::FP16;
    }

    std::cout << "╔════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║              内存占用测试                                  ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;
    std::cout << "模型路径: " << modelPath << std::endl;
    std::cout << "量化类型: " << quantStr << std::endl;
    std::cout << std::endl;

    size_t memBefore = getCurrentMemoryMB();
    std::cout << "1. 初始内存占用: " << memBefore << " MB" << std::endl;

    // 加载 Tokenizer
    std::cout << "2. 加载 Tokenizer..." << std::endl;
    HFTokenizer tokenizer(ModelType::QWEN);
    if (!tokenizer.load(modelPath)) {
        std::cerr << "❌ Tokenizer 加载失败" << std::endl;
        return 1;
    }
    size_t memAfterTokenizer = getCurrentMemoryMB();
    std::cout << "   Tokenizer 内存: " << (memAfterTokenizer - memBefore) << " MB" << std::endl;

    // 加载模型
    std::cout << "3. 加载 Transformer 模型 (" << quantStr << ")..." << std::endl;
    size_t memBeforeModel = getCurrentMemoryMB();
    
    cllm::kylin::HFTransformerModel transformer(modelPath, cllm::kylin::DeviceType::CPU, quantType);
    if (!transformer.isLoaded()) {
        std::cerr << "❌ Transformer 模型加载失败" << std::endl;
        return 1;
    }
    
    size_t memAfterModel = getCurrentMemoryMB();
    size_t modelMemory = memAfterModel - memBeforeModel;
    std::cout << "   模型权重内存: " << modelMemory << " MB" << std::endl;
    std::cout << "   总内存占用: " << memAfterModel << " MB" << std::endl;

    // 运行一次推理
    std::cout << "4. 运行推理测试..." << std::endl;
    std::string prompt = "The future of AI is";
    auto tokens = tokenizer.encode(prompt, true);
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 50; ++i) {
        auto logits = transformer.forward(tokens);
        // 简单贪心采样
        int nextToken = 0;
        float maxLogit = logits[0];
        for (size_t j = 1; j < logits.size(); ++j) {
            if (logits[j] > maxLogit) {
                maxLogit = logits[j];
                nextToken = j;
            }
        }
        tokens.push_back(nextToken);
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    size_t memAfterInference = getCurrentMemoryMB();
    std::cout << "   推理后内存: " << memAfterInference << " MB" << std::endl;

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    float throughput = 50.0f / (duration / 1000.0f);
    
    std::cout << std::endl;
    std::cout << "╔════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║                    测试结果                                ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;
    std::cout << "量化类型: " << quantStr << std::endl;
    std::cout << "模型权重内存: " << modelMemory << " MB" << std::endl;
    std::cout << "总内存占用: " << memAfterInference << " MB" << std::endl;
    std::cout << "推理速度: " << throughput << " tokens/s" << std::endl;
    std::cout << std::endl;

    return 0;
}
