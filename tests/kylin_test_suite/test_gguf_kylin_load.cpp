/**
 * @file test_gguf_kylin_load.cpp
 * @brief 测试 Kylin 后端加载 GGUF 模型
 */

#include "cllm/kylin/gguf/loader.h"
#include "cllm/kylin/core/tensor.h"
#include "cllm/common/logger.h"

#include <iostream>
#include <vector>

using namespace cllm::kylin;

int main(int argc, char** argv) {
    // 初始化日志
    cllm::Logger::instance().setLevel(spdlog::level::debug);
    
    std::string modelPath = "../model/Qwen/qwen3-0.6b-q4_k_m.gguf";
    if (argc > 1) {
        modelPath = argv[1];
    }
    
    std::cout << "========================================" << std::endl;
    std::cout << "Testing GGUF Loading for Kylin Backend" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Model: " << modelPath << std::endl << std::endl;
    
    try {
        // 1. 创建 GGUF Loader
        std::cout << "1. Creating GGUFLoader..." << std::endl;
        GGUFLoader loader(modelPath);
        
        if (!loader.isValid()) {
            std::cerr << "Failed to load GGUF file!" << std::endl;
            return 1;
        }
        
        // 2. 加载配置
        std::cout << "2. Loading config..." << std::endl;
        auto config = loader.loadConfig();
        
        std::cout << "Model config:" << std::endl;
        std::cout << "  Architecture: " << config.architecture << std::endl;
        std::cout << "  Layers: " << config.blockCount << std::endl;
        std::cout << "  Hidden: " << config.embeddingLength << std::endl;
        std::cout << "  Heads: " << config.headCount << " (KV: " << config.headCountKV << ")" << std::endl;
        std::cout << "  Vocab: " << config.vocabSize << std::endl;
        std::cout << "  FFN: " << config.feedForwardLength << std::endl;
        
        const size_t numLayers = config.blockCount;
        const size_t hidden = config.embeddingLength;
        const size_t vocab = config.vocabSize;
        const size_t inter = config.feedForwardLength;
        const size_t headDim = config.headDim();
        const size_t qDim = config.headCount * headDim;
        const size_t kvDim = config.headCountKV * headDim;
        
        std::cout << "  Head dim: " << headDim << std::endl;
        std::cout << "  Q dim: " << qDim << std::endl;
        std::cout << "  KV dim: " << kvDim << std::endl << std::endl;
        
        // 3. 分配张量
        std::cout << "3. Allocating tensors..." << std::endl;
        
        Tensor embedding({vocab, hidden});
        std::vector<Tensor> wq(numLayers), wk(numLayers), wv(numLayers), wo(numLayers);
        std::vector<Tensor> wGate(numLayers), wUp(numLayers), wDown(numLayers);
        std::vector<Tensor> norm1(numLayers), norm2(numLayers);
        Tensor finalNorm({hidden});
        Tensor lmHead({hidden, vocab});
        
        for (size_t i = 0; i < numLayers; ++i) {
            wq[i].resize({hidden, qDim});
            wk[i].resize({hidden, kvDim});
            wv[i].resize({hidden, kvDim});
            wo[i].resize({qDim, hidden});
            wGate[i].resize({hidden, inter});
            wUp[i].resize({hidden, inter});
            wDown[i].resize({inter, hidden});
            norm1[i].resize({hidden});
            norm2[i].resize({hidden});
        }
        
        std::cout << "Tensors allocated." << std::endl << std::endl;
        
        // 4. 加载权重
        std::cout << "4. Loading weights from GGUF..." << std::endl;
        bool success = loader.loadInto(
            embedding, wq, wk, wv, wo,
            wGate, wUp, wDown,
            norm1, norm2,
            finalNorm, lmHead
        );
        
        if (!success) {
            std::cerr << "❌ Failed to load weights!" << std::endl;
            return 1;
        }
        
        std::cout << "✅ Weights loaded successfully!" << std::endl << std::endl;
        
        // 5. 验证权重数据
        std::cout << "5. Verifying loaded weights..." << std::endl;
        
        auto checkTensor = [](const Tensor& t, const std::string& name) {
            size_t nanCount = 0, infCount = 0, zeroCount = 0;
            float minVal = std::numeric_limits<float>::max();
            float maxVal = std::numeric_limits<float>::lowest();
            double sum = 0.0;
            
            for (size_t i = 0; i < t.size(); ++i) {
                float val = t[i];
                if (std::isnan(val)) nanCount++;
                else if (std::isinf(val)) infCount++;
                else if (val == 0.0f) zeroCount++;
                else {
                    minVal = std::min(minVal, val);
                    maxVal = std::max(maxVal, val);
                    sum += val;
                }
            }
            
            std::cout << "  " << name << ": size=" << t.size()
                     << ", range=[" << minVal << ", " << maxVal << "]"
                     << ", mean=" << (sum / t.size())
                     << ", zeros=" << zeroCount
                     << ", NaN=" << nanCount
                     << ", Inf=" << infCount << std::endl;
        };
        
        checkTensor(embedding, "embedding");
        checkTensor(wq[0], "wq[0]");
        checkTensor(wk[0], "wk[0]");
        checkTensor(wv[0], "wv[0]");
        checkTensor(wGate[0], "wGate[0]");
        checkTensor(finalNorm, "finalNorm");
        checkTensor(lmHead, "lmHead");
        
        std::cout << std::endl << "✅ Test completed successfully!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Exception: " << e.what() << std::endl;
        return 1;
    }
}
