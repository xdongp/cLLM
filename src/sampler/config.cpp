/**
 * @file config.cpp
 * @brief 采样器配置类实现
 * @author cLLM Team
 * @date 2026-01-09
 */

#include "cllm/sampler/config.h"
#include <stdexcept>

namespace cllm {

SamplerConfig::SamplerConfig()
    : temperature_(1.0f)
    , topK_(-1)
    , topP_(-1.0f)
    , greedyThreshold_(0.1f) {
}

SamplerConfig::~SamplerConfig() {
}

void SamplerConfig::setTemperature(float temperature) {
    if (temperature < 0.0f) {
        throw std::invalid_argument("Temperature must be non-negative");
    }
    temperature_ = temperature;
}

float SamplerConfig::getTemperature() const {
    return temperature_;
}

void SamplerConfig::setTopK(int topK) {
    topK_ = topK;
}

int SamplerConfig::getTopK() const {
    return topK_;
}

void SamplerConfig::setTopP(float topP) {
    if (topP < -1.0f || topP > 1.0f) {
        throw std::invalid_argument("TopP must be in range [-1.0, 1.0]");
    }
    topP_ = topP;
}

float SamplerConfig::getTopP() const {
    return topP_;
}

void SamplerConfig::setGreedyThreshold(float threshold) {
    if (threshold < 0.0f || threshold > 1.0f) {
        throw std::invalid_argument("Greedy threshold must be in range [0.0, 1.0]");
    }
    greedyThreshold_ = threshold;
}

float SamplerConfig::getGreedyThreshold() const {
    return greedyThreshold_;
}

void SamplerConfig::loadPreset(const std::string& presetName) {
    if (presetName == "greedy") {
        // 贪心采样：温度接近0
        temperature_ = 0.01f;
        topK_ = -1;
        topP_ = -1.0f;
    } else if (presetName == "creative") {
        // 创造性采样：高温度、Top-P采样
        temperature_ = 1.2f;
        topK_ = -1;
        topP_ = 0.95f;
    } else if (presetName == "balanced") {
        // 平衡采样：中等温度、Top-K采样
        temperature_ = 0.7f;
        topK_ = 50;
        topP_ = -1.0f;
    } else if (presetName == "precise") {
        // 精确采样：低温度、Top-K采样
        temperature_ = 0.3f;
        topK_ = 10;
        topP_ = -1.0f;
    } else {
        throw std::invalid_argument("Unknown preset: " + presetName);
    }
}

void SamplerConfig::reset() {
    temperature_ = 1.0f;
    topK_ = -1;
    topP_ = -1.0f;
    greedyThreshold_ = 0.1f;
}

}
