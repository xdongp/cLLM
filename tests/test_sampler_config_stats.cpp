/**
 * @file test_sampler_config_stats.cpp
 * @brief SamplerConfig和SamplerStats测试
 * @author cLLM Team
 * @date 2026-01-09
 */

#include "cllm/sampler.h"
#include "cllm/sampler/config.h"
#include "cllm/sampler/stats.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace cllm;

void testSamplerConfig() {
    std::cout << "Testing SamplerConfig..." << std::endl;
    
    // 测试默认构造
    SamplerConfig config1;
    assert(std::abs(config1.getTemperature() - 1.0f) < 1e-6f);
    assert(config1.getTopK() == -1);
    assert(std::abs(config1.getTopP() - (-1.0f)) < 1e-6f);
    
    // 测试设置参数
    config1.setTemperature(0.7f);
    config1.setTopK(50);
    config1.setTopP(0.9f);
    
    assert(std::abs(config1.getTemperature() - 0.7f) < 1e-6f);
    assert(config1.getTopK() == 50);
    assert(std::abs(config1.getTopP() - 0.9f) < 1e-6f);
    
    // 测试预设配置
    SamplerConfig config2;
    config2.loadPreset("greedy");
    assert(config2.getTemperature() < 0.1f);
    
    config2.loadPreset("creative");
    assert(config2.getTemperature() > 1.0f);
    assert(config2.getTopP() > 0.9f);
    
    config2.loadPreset("balanced");
    assert(config2.getTopK() > 0);
    
    // 测试重置
    config2.reset();
    assert(std::abs(config2.getTemperature() - 1.0f) < 1e-6f);
    
    std::cout << "✓ SamplerConfig tests passed" << std::endl;
}

void testSamplerStats() {
    std::cout << "Testing SamplerStats..." << std::endl;
    
    // 测试默认构造
    SamplerStats stats;
    assert(stats.getTotalSamples() == 0);
    assert(stats.getGreedySamples() == 0);
    
    // 测试统计增加
    stats.incrementGreedySamples();
    assert(stats.getGreedySamples() == 1);
    assert(stats.getTotalSamples() == 1);
    
    stats.incrementTopKSamples();
    stats.incrementTopPSamples();
    assert(stats.getTotalSamples() == 3);
    
    // 测试百分比计算
    assert(std::abs(stats.getGreedyPercentage() - 33.33f) < 1.0f);
    
    // 测试拷贝构造
    SamplerStats stats2(stats);
    assert(stats2.getTotalSamples() == 3);
    assert(stats2.getGreedySamples() == 1);
    
    // 测试赋值运算符
    SamplerStats stats3;
    stats3 = stats;
    assert(stats3.getTotalSamples() == 3);
    
    // 测试重置
    stats.reset();
    assert(stats.getTotalSamples() == 0);
    
    // 测试toString
    std::string statsStr = stats2.toString();
    assert(statsStr.find("Total Samples: 3") != std::string::npos);
    
    std::cout << "✓ SamplerStats tests passed" << std::endl;
}

void testSamplerIntegration() {
    std::cout << "Testing Sampler integration..." << std::endl;
    
    // 测试带配置的构造
    SamplerConfig config;
    config.setTemperature(0.7f);
    config.setGreedyThreshold(0.1f);
    
    Sampler sampler(config);
    
    // 测试配置获取
    SamplerConfig retrievedConfig = sampler.getConfig();
    assert(std::abs(retrievedConfig.getTemperature() - 0.7f) < 1e-6f);
    
    // 测试采样并记录统计
    FloatArray logits(100);
    for (size_t i = 0; i < 100; ++i) {
        logits[i] = static_cast<float>(i);
    }
    
    // 执行几次采样
    sampler.sample(logits, 0.05f);  // 贪心采样
    sampler.sample(logits, 0.7f, 10, -1.0f);  // Top-K采样
    sampler.sample(logits, 0.7f, -1, 0.9f);  // Top-P采样
    sampler.sample(logits, 1.0f);  // 温度采样
    
    // 检查统计
    SamplerStats stats = sampler.getStats();
    assert(stats.getTotalSamples() == 4);
    assert(stats.getGreedySamples() > 0);
    assert(stats.getTopKSamples() > 0);
    assert(stats.getTopPSamples() > 0);
    assert(stats.getTemperatureSamples() > 0);
    
    std::cout << "Stats:" << std::endl;
    std::cout << stats.toString() << std::endl;
    
    // 测试重置统计
    sampler.resetStats();
    stats = sampler.getStats();
    assert(stats.getTotalSamples() == 0);
    
    std::cout << "✓ Sampler integration tests passed" << std::endl;
}

int main() {
    std::cout << "Running SamplerConfig and SamplerStats tests..." << std::endl;
    std::cout << "=================================================" << std::endl;
    
    try {
        testSamplerConfig();
        testSamplerStats();
        testSamplerIntegration();
        
        std::cout << "=================================================" << std::endl;
        std::cout << "✓ All tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "✗ Test failed: " << e.what() << std::endl;
        return 1;
    }
}
