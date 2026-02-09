/**
 * @file block_layout_test.cpp
 * @brief 块布局性能测试
 */

#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <cstring>
#include <cllm/kylin/core/quantization.h>

using namespace cllm::kylin;

// 计时器
class Timer {
    std::chrono::high_resolution_clock::time_point start;
public:
    Timer() { reset(); }
    void reset() { start = std::chrono::high_resolution_clock::now(); }
    double elapsed() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
};

// 初始化随机数据
void initRandomData(std::vector<float>& data) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& d : data) {
        d = dist(gen);
    }
}

// 验证结果
bool verifyResults(const float* ref, const float* test, int n, float tolerance = 1e-3f) {
    for (int i = 0; i < n; ++i) {
        float diff = std::abs(ref[i] - test[i]);
        float relDiff = diff / (std::abs(ref[i]) + 1e-6f);
        if (relDiff > tolerance && diff > tolerance) {
            std::cout << "Mismatch at " << i << ": ref=" << ref[i] << ", test=" << test[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    std::cout << "╔════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║         块布局矩阵乘法性能测试                             ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;
    
    // 测试不同尺寸
    std::vector<std::pair<int, int>> testSizes = {
        {1024, 1024},
        {2048, 2048},
        {4096, 4096},
        {8192, 8192},
    };
    
    const int warmupIterations = 10;
    const int testIterations = 100;
    
    for (const auto& [M, K] : testSizes) {
        std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
        std::cout << "矩阵尺寸: M=" << M << ", K=" << K << std::endl;
        
        // 分配数据
        std::vector<float> weightF32(static_cast<size_t>(M) * K);
        std::vector<float> input(K);
        std::vector<float> outputRef(M);
        std::vector<float> outputTest(M);
        
        initRandomData(weightF32);
        initRandomData(input);
        
        // 转换为 FP16
        std::vector<uint16_t> weightFP16(static_cast<size_t>(M) * K);
        quant_kernels::convert_f32_to_fp16(weightF32.data(), weightFP16.data(), static_cast<size_t>(M) * K);
        
        // 创建块布局权重
        std::vector<uint16_t> weightBlocked(weightFP16.size());
        quant_kernels::reorder_fp16_to_blocked(weightFP16.data(), weightBlocked.data(), M, K);
        
        // Warmup
        std::cout << "Warmup..." << std::endl;
        for (int i = 0; i < warmupIterations; ++i) {
            quant_kernels::matmul_fp16_f32(weightFP16.data(), input.data(), outputRef.data(), M, K);
            quant_kernels::matmul_fp16_f32_blocked(weightBlocked.data(), input.data(), outputTest.data(), M, K);
        }
        
        // 测试标准布局
        Timer timer;
        for (int i = 0; i < testIterations; ++i) {
            quant_kernels::matmul_fp16_f32(weightFP16.data(), input.data(), outputRef.data(), M, K);
        }
        double timeStandard = timer.elapsed() / testIterations;
        
        // 测试块布局
        timer.reset();
        for (int i = 0; i < testIterations; ++i) {
            quant_kernels::matmul_fp16_f32_blocked(weightBlocked.data(), input.data(), outputTest.data(), M, K);
        }
        double timeBlocked = timer.elapsed() / testIterations;
        
        // 验证正确性
        bool correct = verifyResults(outputRef.data(), outputTest.data(), M);
        
        // 计算性能
        double flops = 2.0 * M * K;  // 每次乘加算 2 FLOPs
        double gflopsStandard = flops / (timeStandard * 1e6);
        double gflopsBlocked = flops / (timeBlocked * 1e6);
        
        std::cout << "┌─────────────────────────────────────────────────────────┐" << std::endl;
        std::cout << "│ 标准布局: " << timeStandard << " ms (" << gflopsStandard << " GFLOPS)" << std::endl;
        std::cout << "│ 块布局:   " << timeBlocked << " ms (" << gflopsBlocked << " GFLOPS)" << std::endl;
        std::cout << "│ 加速比:   " << (timeStandard / timeBlocked) << "x" << std::endl;
        std::cout << "│ 正确性:   " << (correct ? "✓ PASS" : "✗ FAIL") << std::endl;
        std::cout << "└─────────────────────────────────────────────────────────┘" << std::endl;
    }
    
    std::cout << "\n测试完成!" << std::endl;
    return 0;
}
