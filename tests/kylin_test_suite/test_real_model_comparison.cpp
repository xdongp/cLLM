/**
 * @file test_real_model_comparison.cpp
 * @brief 真实模型 CPU vs GPU 对比测试
 *
 * 此测试使用实际的 HFTransformerModel 进行 CPU 和 GPU 推理对比
 * 精确定位 GPU 输出乱码的问题所在层
 */

#pragma once

#include "kylin_test_framework.h"
#include "test_common_types.h"
#include <fstream>
#include <iomanip>
#include <cmath>
#include <limits>

// 注意：HFTransformerModel 和 Tokenizer 的完整定义在 kylin_test_main.cpp 中包含

namespace kylin_test {

// ============================================================================
// 辅助函数：计算张量统计信息
// ============================================================================
inline TensorStats computeStats(const std::vector<float>& data) {
    TensorStats stats;
    if (data.empty()) return stats;
    
    stats.min = data[0];
    stats.max = data[0];
    double sum = 0.0;
    double sumSq = 0.0;
    
    for (float val : data) {
        stats.min = std::min(stats.min, val);
        stats.max = std::max(stats.max, val);
        sum += val;
        sumSq += val * val;
    }
    
    stats.mean = sum / data.size();
    double variance = (sumSq / data.size()) - (stats.mean * stats.mean);
    stats.std = std::sqrt(std::max(0.0, variance));
    
    return stats;
}

// ============================================================================
// 辅助函数：计算两个张量的差异
// ============================================================================
inline TensorComparison compareTensors(
    const std::vector<float>& cpu,
    const std::vector<float>& gpu,
    const std::string& name
) {
    TensorComparison comp;
    comp.name = name;
    comp.cpuStats = computeStats(cpu);
    comp.gpuStats = computeStats(gpu);
    
    if (cpu.size() != gpu.size() || cpu.empty()) {
        comp.maxDiff = std::numeric_limits<float>::infinity();
        comp.meanDiff = std::numeric_limits<float>::infinity();
        comp.rmse = std::numeric_limits<float>::infinity();
        return comp;
    }
    
    double sumDiff = 0.0;
    double sumDiffSq = 0.0;
    comp.maxDiff = 0.0;
    
    for (size_t i = 0; i < cpu.size(); ++i) {
        float diff = std::abs(cpu[i] - gpu[i]);
        comp.maxDiff = std::max(comp.maxDiff, diff);
        sumDiff += diff;
        sumDiffSq += diff * diff;
    }
    
    comp.meanDiff = sumDiff / cpu.size();
    comp.rmse = std::sqrt(sumDiffSq / cpu.size());
    
    return comp;
}

// ============================================================================
// 辅助函数：打印对比结果
// ============================================================================
inline void printComparison(std::function<void(LogLevel, const std::string&)> logFunc, const TensorComparison& comp) {
    logFunc(LogLevel::INFO, "  对比: " + comp.name);
    logFunc(LogLevel::INFO, "    CPU 统计: min=" + std::to_string(comp.cpuStats.min) + 
                        ", max=" + std::to_string(comp.cpuStats.max) + 
                        ", mean=" + std::to_string(comp.cpuStats.mean) + 
                        ", std=" + std::to_string(comp.cpuStats.std));
    logFunc(LogLevel::INFO, "    GPU 统计: min=" + std::to_string(comp.gpuStats.min) + 
                        ", max=" + std::to_string(comp.gpuStats.max) + 
                        ", mean=" + std::to_string(comp.gpuStats.mean) + 
                        ", std=" + std::to_string(comp.gpuStats.std));
    logFunc(LogLevel::INFO, "    差异: max=" + std::to_string(comp.maxDiff) + 
                        ", mean=" + std::to_string(comp.meanDiff) + 
                        ", RMSE=" + std::to_string(comp.rmse));
    
    if (comp.maxDiff > 1.0f) {
        logFunc(LogLevel::WARN, "    ⚠ 差异较大！");
    } else if (comp.maxDiff > 0.01f) {
        logFunc(LogLevel::WARN, "    ⚠ 有差异");
    } else {
        logFunc(LogLevel::PASS, "    ✓ 基本一致");
    }
}

// ============================================================================
// 真实模型 CPU vs GPU 逐层对比测试
// ============================================================================

class RealModelLayerComparisonTest : public TestCase {
public:
    RealModelLayerComparisonTest() : TestCase(
        "real_model_layer_comparison",
        "真实模型 CPU vs GPU 逐层对比"
    ) {}

    void execute() override {
        log(LogLevel::INFO, "开始真实模型 CPU vs GPU 逐层对比测试...");
        
        std::string modelPath = getModelPath();
        log(LogLevel::INFO, "模型路径: " + modelPath);
        
        // 测试输入：使用简单的 token IDs
        // 101, 102 是 BERT 的 [CLS], [SEP]，这里用作文本生成的起始 token
        std::vector<int32_t> inputIds = {101};  // 单 token 输入
        
        try {
            // 1. 加载 CPU 模型 (使用 FP16 精度，与 Metal 一致)
            log(LogLevel::INFO, "\n[1/4] 加载 CPU 模型...");
            log(LogLevel::INFO, "  模型路径: " + modelPath);
            log(LogLevel::INFO, "  设备: CPU, 精度: FP16 (与 Metal 一致)");
            
            cllm::kylin::HFTransformerModel cpuModel(modelPath, cllm::kylin::DeviceType::CPU, 
                                                      cllm::kylin::QuantType::FP16);
            if (!cpuModel.isLoaded()) {
                log(LogLevel::FAIL, "CPU 模型加载失败，请检查模型路径是否正确");
                log(LogLevel::INFO, "预期路径: " + modelPath);
                log(LogLevel::INFO, "请确保路径包含 config.json 和 model.safetensors");
                return;
            }
            log(LogLevel::PASS, "CPU 模型加载成功");
            log(LogLevel::INFO, "  配置: hiddenSize=" + std::to_string(cpuModel.hiddenSize()) + 
                              ", vocabSize=" + std::to_string(cpuModel.vocabSize()));
            
            // 2. 加载 GPU 模型
            log(LogLevel::INFO, "\n[2/4] 加载 GPU 模型...");
            log(LogLevel::INFO, "  模型路径: " + modelPath);
            log(LogLevel::INFO, "  设备: Metal GPU (FP16)");
            
            cllm::kylin::HFTransformerModel gpuModel(modelPath, cllm::kylin::DeviceType::Metal);
            if (!gpuModel.isLoaded()) {
                log(LogLevel::FAIL, "GPU 模型加载失败");
                return;
            }
            if (!gpuModel.isUsingGPU()) {
                log(LogLevel::FAIL, "GPU 后端未启用，可能 Metal 不支持");
                return;
            }
            log(LogLevel::PASS, "GPU 模型加载成功");
            
            // 3. CPU 推理并导出中间结果
            log(LogLevel::INFO, "\n[3/4] CPU 推理并导出中间结果...");
            log(LogLevel::INFO, "  输入 token ID: " + std::to_string(inputIds[0]));
            
            std::vector<cllm::kylin::HFTransformerModel::LayerDebugOutput> cpuLayerOutputs;
            std::vector<float> cpuEmbedding, cpuFinalNorm;
            
            auto cpuLogits = cpuModel.forwardWithDebugCPU(
                inputIds, cpuLayerOutputs, cpuEmbedding, cpuFinalNorm
            );
            
            if (cpuLogits.empty()) {
                log(LogLevel::FAIL, "CPU 推理失败");
                return;
            }
            log(LogLevel::PASS, "CPU 推理完成，共 " + std::to_string(cpuLayerOutputs.size()) + " 层");
            
            // 4. GPU 推理并导出中间结果
            log(LogLevel::INFO, "\n[4/4] GPU 推理并导出中间结果...");
            log(LogLevel::INFO, "  输入 token ID: " + std::to_string(inputIds[0]));
            
            std::vector<cllm::kylin::GGMLGPUBackend::LayerOutput> gpuLayerOutputs;
            std::vector<float> gpuEmbedding, gpuFinalNorm;
            
            auto gpuLogits = gpuModel.forwardWithDebugGPU(
                inputIds, gpuLayerOutputs, gpuEmbedding, gpuFinalNorm
            );
            
            if (gpuLogits.empty()) {
                log(LogLevel::FAIL, "GPU 推理失败");
                return;
            }
            log(LogLevel::PASS, "GPU 推理完成，共 " + std::to_string(gpuLayerOutputs.size()) + " 层");
            
            // 5. 对比结果
            log(LogLevel::INFO, "\n========== 对比结果 ==========");
            
            // 对比 Layer 0 的中间结果
            if (!cpuLayerOutputs.empty() && !gpuLayerOutputs.empty()) {
                log(LogLevel::INFO, "\n--- Layer 0 详细对比 ---");
                
                auto inputNormComp = compareTensors(cpuLayerOutputs[0].inputNormOutput, 
                                                     gpuLayerOutputs[0].afterInputNorm, 
                                                     "Layer 0 InputNorm");
                printComparison([this](LogLevel level, const std::string& msg) { log(level, msg); }, inputNormComp);
                
                auto qkvComp = compareTensors(cpuLayerOutputs[0].qkvOutput, 
                                              gpuLayerOutputs[0].afterQKV, 
                                              "Layer 0 QKV");
                printComparison([this](LogLevel level, const std::string& msg) { log(level, msg); }, qkvComp);
                
                auto attnComp = compareTensors(cpuLayerOutputs[0].attentionOutput, 
                                               gpuLayerOutputs[0].afterAttention, 
                                               "Layer 0 Attention");
                printComparison([this](LogLevel level, const std::string& msg) { log(level, msg); }, attnComp);
                
                auto postNormComp = compareTensors(cpuLayerOutputs[0].postNormOutput, 
                                                   gpuLayerOutputs[0].afterPostNorm, 
                                                   "Layer 0 PostNorm");
                printComparison([this](LogLevel level, const std::string& msg) { log(level, msg); }, postNormComp);
                
                auto ffnComp = compareTensors(cpuLayerOutputs[0].ffnOutput, 
                                              gpuLayerOutputs[0].afterFFN, 
                                              "Layer 0 FFN");
                printComparison([this](LogLevel level, const std::string& msg) { log(level, msg); }, ffnComp);
            }
            
            // Logits 对比
            log(LogLevel::INFO, "\n--- Logits ---");
            auto logitsComp = compareTensors(cpuLogits, gpuLogits, "Logits");
            printComparison([this](LogLevel level, const std::string& msg) { log(level, msg); }, logitsComp);
            
            // 总结
            log(LogLevel::INFO, "\n========== 对比总结 ==========");
            if (logitsComp.maxDiff > 0.01f) {
                log(LogLevel::WARN, "发现差异！CPU vs GPU logits max diff = " + 
                    std::to_string(logitsComp.maxDiff));
            } else {
                log(LogLevel::PASS, "✓ CPU 和 GPU logits 基本一致");
            }
            
            log(LogLevel::PASS, "逐层对比测试完成");
            
        } catch (const std::exception& e) {
            log(LogLevel::ERROR, "测试异常: " + std::string(e.what()));
            throw;
        }
    }

private:
    std::string getModelPath() {
        // 优先使用环境变量
        const char* envPath = std::getenv("CLLM_MODEL_PATH");
        if (envPath) return envPath;
        
        // 默认使用绝对路径
        return "/Users/dannypan/PycharmProjects/cLLM/model/Qwen/Qwen3-0.6B";
    }
};

// ============================================================================
// 测试套件注册
// ============================================================================

inline void registerRealModelTests(TestSuite& suite) {
    suite.addTest(std::make_shared<RealModelLayerComparisonTest>());
}

} // namespace kylin_test
