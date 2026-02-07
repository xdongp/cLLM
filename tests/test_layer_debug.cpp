#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <chrono>

#include "cllm/kylin/hf/hf_transformer_model.h"
#include "cllm/tokenizer/hf_tokenizer.h"
#include "cllm/common/logger.h"

using namespace cllm;
using namespace cllm::kylin;

int main(int argc, char** argv) {
    // 设置日志级别为 INFO 以查看调试输出
    cllm::Logger::instance().setLevel(spdlog::level::info);
    
    std::string model_path = "model/Qwen/Qwen3-0.6B";
    std::string test_input = "你好";
    std::string device_type = "cpu";  // "cpu" 或 "gpu"
    
    // 解析命令行参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--input" && i + 1 < argc) {
            test_input = argv[++i];
        } else if (arg == "--device" && i + 1 < argc) {
            device_type = argv[++i];
        }
    }
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "逐层输出对比测试" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "模型路径: " << model_path << std::endl;
    std::cout << "测试输入: '" << test_input << "'" << std::endl;
    std::cout << "设备类型: " << (device_type == "gpu" ? "GPU" : "CPU") << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    try {
        // 加载 Tokenizer
        std::cout << "\n加载 Tokenizer..." << std::endl;
        auto tokenizer = std::make_unique<HFTokenizer>();
        if (!tokenizer->load(model_path)) {
            std::cerr << "Failed to load tokenizer" << std::endl;
            return 1;
        }
        
        // Tokenize 输入
        std::vector<int32_t> input_ids_temp = tokenizer->encode(test_input, true);
        std::vector<int32_t> input_ids(input_ids_temp.begin(), input_ids_temp.end());
        std::cout << "Tokenized input: [";
        for (size_t i = 0; i < input_ids.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << input_ids[i];
        }
        std::cout << "]" << std::endl;
        
        // 创建模型
        std::cout << "\n加载模型..." << std::endl;
        DeviceType device = (device_type == "gpu") ? DeviceType::Metal : DeviceType::CPU;
        QuantType quant = QuantType::FP32;
        
        auto model = std::make_unique<HFTransformerModel>(model_path, device, quant);
        if (!model->isLoaded()) {
            std::cerr << "Failed to load model" << std::endl;
            return 1;
        }
        
        std::cout << "模型加载成功" << std::endl;
        std::cout << "隐藏层维度: " << model->hiddenSize() << std::endl;
        std::cout << "词表大小: " << model->vocabSize() << std::endl;
        
        // 执行前向传播
        std::cout << "\n" << std::string(60, '-') << std::endl;
        std::cout << "开始前向传播..." << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<float> logits = model->forward(input_ids);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "\n前向传播完成，耗时: " << duration.count() << " ms" << std::endl;
        
        // 分析 logits
        std::cout << "\n" << std::string(60, '-') << std::endl;
        std::cout << "Logits 统计" << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        
        float min_val = logits[0], max_val = logits[0];
        double sum = 0;
        size_t nan_count = 0, inf_count = 0;
        
        for (size_t i = 0; i < logits.size(); ++i) {
            float v = logits[i];
            if (std::isnan(v)) { nan_count++; continue; }
            if (std::isinf(v)) { inf_count++; continue; }
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
            sum += v;
        }
        
        std::cout << "词表大小: " << logits.size() << std::endl;
        std::cout << "Min: " << min_val << std::endl;
        std::cout << "Max: " << max_val << std::endl;
        std::cout << "Mean: " << (sum / logits.size()) << std::endl;
        std::cout << "NaN count: " << nan_count << std::endl;
        std::cout << "Inf count: " << inf_count << std::endl;
        
        // 找到 top 5 tokens
        std::cout << "\nTop 5 tokens:" << std::endl;
        std::vector<std::pair<float, int>> top_tokens;
        for (size_t i = 0; i < logits.size(); ++i) {
            top_tokens.push_back({logits[i], static_cast<int>(i)});
        }
        std::partial_sort(top_tokens.begin(), top_tokens.begin() + 5, top_tokens.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });
        
        for (int i = 0; i < 5; ++i) {
            int token_id = top_tokens[i].second;
            float logit = top_tokens[i].first;
            std::string token_text = tokenizer->decode({token_id}, true);
            std::cout << "  [" << i+1 << "] Token ID: " << token_id 
                      << ", Logit: " << logit 
                      << ", Text: '" << token_text << "'" << std::endl;
        }
        
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "测试完成" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
