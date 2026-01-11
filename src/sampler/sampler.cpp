#include "cllm/sampler.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <limits>
#include <queue>
#include <functional>

namespace cllm {

Sampler::Sampler()
    : gen_(std::random_device{}()), config_(), stats_(), sampleCount_(0) {
    // 使用配置中的默认值初始化config_
    config_.setTemperature(Config::instance().samplerTemperature());
    config_.setTopK(Config::instance().samplerTopK());
    config_.setTopP(Config::instance().samplerTopP());
    config_.setGreedyThreshold(Config::instance().samplerGreedyThreshold());
}

Sampler::Sampler(const SamplerConfig& config)
    : gen_(std::random_device{}()), config_(config), stats_(), sampleCount_(0) {
    // 如果配置参数为空或默认，则从全局配置加载
    if (config.getTemperature() == 1.0f && config.getTopK() == -1 && config.getTopP() == -1.0f) {
        config_.setTemperature(Config::instance().samplerTemperature());
        config_.setTopK(Config::instance().samplerTopK());
        config_.setTopP(Config::instance().samplerTopP());
        config_.setGreedyThreshold(Config::instance().samplerGreedyThreshold());
    }
}

Sampler::~Sampler() = default;

int Sampler::sample(const FloatArray& logits, float temperature, int topK, float topP) {
    // 参数验证
    if (logits.empty()) {
        throw std::invalid_argument("logits cannot be empty");
    }
    if (temperature < 0.0f) {
        throw std::invalid_argument("temperature must be non-negative");
    }
    if (topP < -1.0f || topP > 1.0f) {
        throw std::invalid_argument("topP must be in range [-1.0, 1.0]");
    }
    
    sampleCount_++;
    int result = sampleSingle(logits.data(), logits.size(), temperature, topK, topP);
    
    // 更新统计信息
    if (temperature <= config_.getGreedyThreshold()) {
        stats_.incrementGreedySamples();
    } else if (topK > 0) {
        stats_.incrementTopKSamples();
    } else if (topP > 0.0f && topP < 1.0f) {
        stats_.incrementTopPSamples();
    } else {
        stats_.incrementTemperatureSamples();
    }
    
    return result;
}

std::vector<int> Sampler::sampleBatch(const FloatArray& logits, int batchSize, float temperature, int topK, float topP) {
    std::vector<int> result(batchSize);
    size_t vocabSize = logits.size() / batchSize;
    
    for (int i = 0; i < batchSize; ++i) {
        const float* batchLogits = logits.data() + i * vocabSize;
        result[i] = sampleSingle(batchLogits, vocabSize, temperature, topK, topP);
        sampleCount_++;
    }
    
    return result;
}

void Sampler::setConfig(const SamplerConfig& config) {
    config_ = config;
}

SamplerConfig Sampler::getConfig() const {
    return config_;
}

SamplerStats Sampler::getStats() const {
    return stats_;
}

size_t Sampler::getSampleCount() const {
    return sampleCount_;
}

void Sampler::resetStats() {
    sampleCount_ = 0;
    stats_.reset();
}

int Sampler::sampleGreedy(const FloatArray& logits) {
    if (logits.empty()) {
        return 0;
    }
    
    int maxIndex = 0;
    float maxValue = logits[0];
    
    for (size_t i = 1; i < logits.size(); ++i) {
        if (logits[i] > maxValue) {
            maxValue = logits[i];
            maxIndex = static_cast<int>(i);
        }
    }
    
    return maxIndex;
}

int Sampler::sampleTemperature(const FloatArray& logits, float temperature) {
    return sampleSingle(logits.data(), logits.size(), temperature, -1, -1.0f);
}

int Sampler::sampleTopK(const FloatArray& logits, int k, float temperature) {
    return sampleSingle(logits.data(), logits.size(), temperature, k, -1.0f);
}

int Sampler::sampleTopP(const FloatArray& logits, float p, float temperature) {
    return sampleSingle(logits.data(), logits.size(), temperature, -1, p);
}

int Sampler::sampleSingle(const float* logits, size_t vocabSize, float temperature, int topK, float topP) {
    if (vocabSize == 0) {
        return 0;
    }
    
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    // Handle greedy sampling case
    if (temperature <= 0.0f || (topK <= 0 && topP <= 0.0f && std::abs(temperature - 1.0f) < 1e-6f)) {
        const float* maxIter = std::max_element(logits, logits + vocabSize);
        return static_cast<int>(maxIter - logits);
    }
    
    // Apply temperature scaling
    std::vector<float> scaledLogits(vocabSize);
    float maxLogit = *std::max_element(logits, logits + vocabSize);
    
    for (size_t i = 0; i < vocabSize; ++i) {
        scaledLogits[i] = std::exp((logits[i] - maxLogit) / temperature);
    }
    
    // Apply Top-K filtering
    if (topK > 0 && static_cast<size_t>(topK) < vocabSize) {
        // Create a min-heap to find top K logits
        std::priority_queue<std::pair<float, size_t>, 
                           std::vector<std::pair<float, size_t>>, 
                           std::greater<std::pair<float, size_t>>> minHeap;
        
        for (size_t i = 0; i < vocabSize; ++i) {
            if (minHeap.size() < static_cast<size_t>(topK)) {
                minHeap.emplace(scaledLogits[i], i);
            } else if (scaledLogits[i] > minHeap.top().first) {
                minHeap.pop();
                minHeap.emplace(scaledLogits[i], i);
            }
        }
        
        // Set all logits not in top K to 0
        std::vector<bool> isTopK(vocabSize, false);
        while (!minHeap.empty()) {
            isTopK[minHeap.top().second] = true;
            minHeap.pop();
        }
        
        for (size_t i = 0; i < vocabSize; ++i) {
            if (!isTopK[i]) {
                scaledLogits[i] = 0.0f;
            }
        }
    }
    
    // Calculate sum for normalization
    float sumExp = std::accumulate(scaledLogits.begin(), scaledLogits.end(), 0.0f);
    if (sumExp < 1e-6f) {
        // If all probabilities are zero, return greedy
        const float* maxIter = std::max_element(logits, logits + vocabSize);
        return static_cast<int>(maxIter - logits);
    }
    
    // Apply Top-P filtering if needed
    if (topP > 0.0f && topP < 1.0f) {
        // Create a list of indices sorted by probability
        std::vector<size_t> indices(vocabSize);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), 
                 [&](size_t a, size_t b) { return scaledLogits[a] > scaledLogits[b]; });
        
        // Normalize and find cumulative probabilities
        float cumulative = 0.0f;
        size_t cutoffIndex = vocabSize;
        
        for (size_t i = 0; i < vocabSize; ++i) {
            float prob = scaledLogits[indices[i]] / sumExp;
            cumulative += prob;
            
            if (cumulative >= topP) {
                cutoffIndex = i + 1;
                break;
            }
        }
        
        // Set probabilities beyond cutoff to 0
        for (size_t i = cutoffIndex; i < vocabSize; ++i) {
            scaledLogits[indices[i]] = 0.0f;
        }
        
        // Recalculate sum after filtering
        sumExp = std::accumulate(scaledLogits.begin(), scaledLogits.end(), 0.0f);
    }
    
    // Sample from the distribution
    float r = dist(gen_) * sumExp;
    float cumulative = 0.0f;
    
    for (size_t i = 0; i < vocabSize; ++i) {
        cumulative += scaledLogits[i];
        if (r <= cumulative) {
            return static_cast<int>(i);
        }
    }
    
    return static_cast<int>(vocabSize - 1);
}

}
