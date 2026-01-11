/**
 * @file test_modelexecutor_init.cpp
 * @brief 简单测试 ModelExecutor 初始化问题
 */

#include <cllm/model/executor.h>
#include <cllm/common/logger.h>
#include <iostream>
#include <exception>

using namespace cllm;

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Testing ModelExecutor Initialization" << std::endl;
    std::cout << "========================================" << std::endl;
    
    try {
        std::cout << "\n1. Creating ModelExecutor with empty path (Kylin backend)..." << std::endl;
        
        ModelExecutor executor(
            "",     // 空路径
            "",     // 不使用量化
            true,   // 启用 SIMD
            false   // 使用 Kylin 后端
        );
        
        std::cout << "   ✓ ModelExecutor created successfully" << std::endl;
        
        std::cout << "\n2. Loading model..." << std::endl;
        executor.loadModel();
        
        std::cout << "   ✓ Model loaded successfully" << std::endl;
        
        std::cout << "\n3. Checking model status..." << std::endl;
        std::cout << "   - Is loaded: " << (executor.isLoaded() ? "Yes" : "No") << std::endl;
        
        const auto& config = executor.getConfig();
        std::cout << "   - Vocab size: " << config.vocabSize << std::endl;
        std::cout << "   - Hidden size: " << config.hiddenSize << std::endl;
        std::cout << "   - Num layers: " << config.numLayers << std::endl;
        
        std::cout << "\n✅ All tests passed!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Exception caught: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "\n❌ Unknown exception caught" << std::endl;
        return 2;
    }
}
