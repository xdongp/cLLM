/**
 * @file stats.cpp
 * @brief 采样器统计信息类实现
 * @author cLLM Team
 * @date 2026-01-09
 */

#include "cllm/sampler/stats.h"
#include <sstream>
#include <iomanip>

namespace cllm {

SamplerStats::SamplerStats() {
}

SamplerStats::SamplerStats(const SamplerStats& other) 
    : totalSamples_(other.totalSamples_)
    , greedySamples_(other.greedySamples_)
    , topKSamples_(other.topKSamples_)
    , topPSamples_(other.topPSamples_)
    , temperatureSamples_(other.temperatureSamples_) {
}

SamplerStats& SamplerStats::operator=(const SamplerStats& other) {
    if (this != &other) {
        totalSamples_ = other.totalSamples_;
        greedySamples_ = other.greedySamples_;
        topKSamples_ = other.topKSamples_;
        topPSamples_ = other.topPSamples_;
        temperatureSamples_ = other.temperatureSamples_;
    }
    return *this;
}

SamplerStats::~SamplerStats() {
}

void SamplerStats::incrementTotalSamples() {
    ++totalSamples_;
}

void SamplerStats::incrementGreedySamples() {
    ++greedySamples_;
    ++totalSamples_;
}

void SamplerStats::incrementTopKSamples() {
    ++topKSamples_;
    ++totalSamples_;
}

void SamplerStats::incrementTopPSamples() {
    ++topPSamples_;
    ++totalSamples_;
}

void SamplerStats::incrementTemperatureSamples() {
    ++temperatureSamples_;
    ++totalSamples_;
}

long long SamplerStats::getTotalSamples() const {
    return totalSamples_;
}

long long SamplerStats::getGreedySamples() const {
    return greedySamples_;
}

long long SamplerStats::getTopKSamples() const {
    return topKSamples_;
}

long long SamplerStats::getTopPSamples() const {
    return topPSamples_;
}

long long SamplerStats::getTemperatureSamples() const {
    return temperatureSamples_;
}

float SamplerStats::getGreedyPercentage() const {
    return calculatePercentage(greedySamples_);
}

float SamplerStats::getTopKPercentage() const {
    return calculatePercentage(topKSamples_);
}

float SamplerStats::getTopPPercentage() const {
    return calculatePercentage(topPSamples_);
}

float SamplerStats::getTemperaturePercentage() const {
    return calculatePercentage(temperatureSamples_);
}

void SamplerStats::reset() {
    totalSamples_ = 0;
    greedySamples_ = 0;
    topKSamples_ = 0;
    topPSamples_ = 0;
    temperatureSamples_ = 0;
}

std::string SamplerStats::toString() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    oss << "SamplerStats {" << std::endl;
    oss << "  Total Samples: " << totalSamples_ << std::endl;
    oss << "  Greedy: " << greedySamples_ 
        << " (" << calculatePercentage(greedySamples_) << "%)" << std::endl;
    oss << "  Top-K: " << topKSamples_ 
        << " (" << calculatePercentage(topKSamples_) << "%)" << std::endl;
    oss << "  Top-P: " << topPSamples_ 
        << " (" << calculatePercentage(topPSamples_) << "%)" << std::endl;
    oss << "  Temperature: " << temperatureSamples_ 
        << " (" << calculatePercentage(temperatureSamples_) << "%)" << std::endl;
    oss << "}";
    
    return oss.str();
}

float SamplerStats::calculatePercentage(long long count) const {
    if (totalSamples_ == 0) {
        return 0.0f;
    }
    return (static_cast<float>(count) / static_cast<float>(totalSamples_)) * 100.0f;
}

}
