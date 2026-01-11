#include "cllm/tokenizer/config.h"

namespace cllm {

TokenizerConfig::TokenizerConfig()
    : maxTokens_(100),
      temperature_(0.7f),
      topP_(0.9f),
      topK_(50),
      repeatPenalty_(1.0f) {
}

TokenizerConfig::~TokenizerConfig() {
}

void TokenizerConfig::setMaxTokens(int maxTokens) {
    maxTokens_ = maxTokens;
}

void TokenizerConfig::setTemperature(float temperature) {
    temperature_ = temperature;
}

void TokenizerConfig::setTopP(float topP) {
    topP_ = topP;
}

void TokenizerConfig::setTopK(int topK) {
    topK_ = topK;
}

void TokenizerConfig::setStopTokens(const std::vector<int>& stopTokens) {
    stopTokens_ = stopTokens;
}

void TokenizerConfig::setRepeatPenalty(float repeatPenalty) {
    repeatPenalty_ = repeatPenalty;
}

int TokenizerConfig::getMaxTokens() const {
    return maxTokens_;
}

float TokenizerConfig::getTemperature() const {
    return temperature_;
}

float TokenizerConfig::getTopP() const {
    return topP_;
}

int TokenizerConfig::getTopK() const {
    return topK_;
}

std::vector<int> TokenizerConfig::getStopTokens() const {
    return stopTokens_;
}

float TokenizerConfig::getRepeatPenalty() const {
    return repeatPenalty_;
}

}
