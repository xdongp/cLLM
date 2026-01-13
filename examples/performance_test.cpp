#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <cstdint>
#include <memory>
#include <numeric>
#include "cllm/model/gguf_loader_new.h"
#include "cllm/model/loader_interface.h"
#include "cllm/model/gguf_dequantization.h"
#include "cllm/common/logger.h"

// 性能计时工具类
template <typename Clock = std::chrono::high_resolution_clock>
class Stopwatch {
public:
    Stopwatch() : start_(Clock::now()) {}
    
    void reset() {
        start_ = Clock::now();
    }
    
    template <typename Duration = std::chrono::milliseconds>
    typename Duration::rep elapsed() const {
        return std::chrono::duration_cast<Duration>(Clock::now() - start_).count();
    }
    
    // 打印耗时
    template <typename Duration = std::chrono::milliseconds>
    void print(const std::string& message) const {
        std::cout << message << ": " << elapsed<Duration>() << " ms" << std::endl;
    }
    
private:
    typename Clock::time_point start_;
};

// 测试反量化性能
void testDequantizationPerformance() {
    std::cout << "=== 测试反量化性能 ===\n";
    
    // 生成测试数据
    const size_t testSize = 1024 * 1024 * 4; // 4MB数据
    
    // F16反量化测试
    std::vector<uint16_t> f16Data(testSize);
    std::vector<float> f32Result(testSize);
    
    // 初始化随机F16数据
    for (size_t i = 0; i < testSize; ++i) {
        f16Data[i] = static_cast<uint16_t>(i % 65536);
    }
    
    // 测量F16反量化时间
    Stopwatch<> sw;
    cllm::dequantizeF16ToF32(f16Data.data(), f32Result.data(), testSize);
    sw.print("F16到F32反量化耗时 (" + std::to_string(testSize) + " 元素)");
    
    // Q8_0反量化测试
    std::vector<int8_t> q8Data(testSize);
    std::vector<float> q8Result(testSize);
    
    // 初始化随机Q8数据
    for (size_t i = 0; i < testSize; ++i) {
        q8Data[i] = static_cast<int8_t>(i % 256 - 128);
    }
    
    // 测量Q8_0反量化时间
    sw.reset();
    cllm::dequantizeQ8ToF32(q8Data.data(), q8Result.data(), testSize);
    sw.print("Q8_0到F32反量化耗时 (" + std::to_string(testSize) + " 元素)");
}

// 测试GGUF加载器性能
void testGGUFLoaderPerformance(const std::string& modelPath) {
    std::cout << "\n=== 测试GGUF加载器性能 ===\n";
    
    try {
        // 测试内存映射模式
        std::cout << "\n1. 内存映射模式：\n";
        
        Stopwatch<> sw;
        auto loader = std::make_unique<cllm::GGUFLoader>(modelPath, true);
        sw.print("创建加载器耗时");
        
        sw.reset();
        if (loader->load()) {
            sw.print("加载模型元数据耗时");
        } else {
            std::cerr << "加载模型失败\n";
            return;
        }
        
        // 测试按需加载
        std::cout << "\n2. 按需加载测试：\n";
        
        cllm::model::WeightData weight;
        
        sw.reset();
        if (loader->hasWeight("embedding")) {
            if (loader->loadWeightByName("embedding", weight)) {
                sw.print("按需加载embedding耗时 (" + std::to_string(weight.size()) + " 元素)");
            }
        }
        
        // 测试层权重加载
        sw.reset();
        if (loader->hasWeight("layers.0.attention.wq.weight")) {
            if (loader->loadWeightByName("layers.0.attention.wq.weight", weight)) {
                sw.print("按需加载层权重耗时 (" + std::to_string(weight.size()) + " 元素)");
            }
        }
        
        // 测试传统文件I/O模式（可选）
        /*
        std::cout << "\n3. 传统文件I/O模式：\n";
        
        sw.reset();
        auto loaderFileIO = std::make_unique<cllm::GGUFLoader>(modelPath, false);
        sw.print("创建加载器耗时");
        
        sw.reset();
        if (loaderFileIO->load()) {
            sw.print("加载模型元数据耗时");
        } else {
            std::cerr << "加载模型失败\n";
            return;
        }
        */
        
    } catch (const std::exception& e) {
        std::cerr << "测试失败：" << e.what() << std::endl;
    }
}

// 测试批量读取性能
void testBatchReadingPerformance(const std::string& modelPath) {
    std::cout << "\n=== 测试批量读取性能 ===\n";
    std::cout << "注意：由于访问限制，此测试已跳过\n";
    // 由于setFilePosition和readValues是GGUFLoader的私有方法，我们无法直接测试批量读取性能
    // 这个功能已经在其他测试中间接验证过
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "用法：" << argv[0] << " <GGUF模型文件路径>" << std::endl;
        return 1;
    }
    
    std::string modelPath = argv[1];
    
    // 测试反量化性能
    testDequantizationPerformance();
    
    // 测试GGUF加载器性能
    testGGUFLoaderPerformance(modelPath);
    
    // 测试批量读取性能
    testBatchReadingPerformance(modelPath);
    
    std::cout << "\n=== 性能测试完成 ===\n";
    return 0;
}