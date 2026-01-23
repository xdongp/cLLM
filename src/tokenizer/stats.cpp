#include "cllm/tokenizer/stats.h"
#include <algorithm>

namespace cllm {

TokenizerStats::TokenizerStats()
    : encodeCount_(0),
      decodeCount_(0),
      generateCount_(0),
      streamGenerateCount_(0),
      totalEncodeTime_(0.0f),
      totalDecodeTime_(0.0f),
      totalGenerateTime_(0.0f),
      totalStreamGenerateTime_(0.0f),
      totalGeneratedTokens_(0) {
}

TokenizerStats::~TokenizerStats() {
}

TokenizerStats::TokenizerStats(const TokenizerStats& other)
    : encodeCount_(0),
      decodeCount_(0),
      generateCount_(0),
      streamGenerateCount_(0),
      totalEncodeTime_(0.0f),
      totalDecodeTime_(0.0f),
      totalGenerateTime_(0.0f),
      totalStreamGenerateTime_(0.0f),
      totalGeneratedTokens_(0) {
    std::lock_guard<std::mutex> lockOther(other.mutex_);
    encodeCount_ = other.encodeCount_;
    decodeCount_ = other.decodeCount_;
    generateCount_ = other.generateCount_;
    streamGenerateCount_ = other.streamGenerateCount_;
    totalEncodeTime_ = other.totalEncodeTime_;
    totalDecodeTime_ = other.totalDecodeTime_;
    totalGenerateTime_ = other.totalGenerateTime_;
    totalStreamGenerateTime_ = other.totalStreamGenerateTime_;
    totalGeneratedTokens_ = other.totalGeneratedTokens_;
}

TokenizerStats& TokenizerStats::operator=(const TokenizerStats& other) {
    if (this != &other) {
        std::lock(mutex_, other.mutex_);
        std::lock_guard<std::mutex> lockThis(mutex_, std::adopt_lock);
        std::lock_guard<std::mutex> lockOther(other.mutex_, std::adopt_lock);
        
        encodeCount_ = other.encodeCount_;
        decodeCount_ = other.decodeCount_;
        generateCount_ = other.generateCount_;
        streamGenerateCount_ = other.streamGenerateCount_;
        totalEncodeTime_ = other.totalEncodeTime_;
        totalDecodeTime_ = other.totalDecodeTime_;
        totalGenerateTime_ = other.totalGenerateTime_;
        totalStreamGenerateTime_ = other.totalStreamGenerateTime_;
        totalGeneratedTokens_ = other.totalGeneratedTokens_;
    }
    return *this;
}

void TokenizerStats::incrementEncodeCount() {
    std::lock_guard<std::mutex> lock(mutex_);
    encodeCount_++;
}

void TokenizerStats::incrementDecodeCount() {
    std::lock_guard<std::mutex> lock(mutex_);
    decodeCount_++;
}

void TokenizerStats::incrementGenerateCount() {
    std::lock_guard<std::mutex> lock(mutex_);
    generateCount_++;
}

void TokenizerStats::incrementStreamGenerateCount() {
    std::lock_guard<std::mutex> lock(mutex_);
    streamGenerateCount_++;
}

void TokenizerStats::addEncodeTime(float time) {
    std::lock_guard<std::mutex> lock(mutex_);
    totalEncodeTime_ += time;
}

void TokenizerStats::addDecodeTime(float time) {
    std::lock_guard<std::mutex> lock(mutex_);
    totalDecodeTime_ += time;
}

void TokenizerStats::addGenerateTime(float time) {
    std::lock_guard<std::mutex> lock(mutex_);
    totalGenerateTime_ += time;
}

void TokenizerStats::addStreamGenerateTime(float time) {
    std::lock_guard<std::mutex> lock(mutex_);
    totalStreamGenerateTime_ += time;
}

void TokenizerStats::addGeneratedTokens(int count) {
    std::lock_guard<std::mutex> lock(mutex_);
    totalGeneratedTokens_ += count;
}

long long TokenizerStats::getEncodeCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return encodeCount_;
}

long long TokenizerStats::getDecodeCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return decodeCount_;
}

long long TokenizerStats::getGenerateCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return generateCount_;
}

long long TokenizerStats::getStreamGenerateCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return streamGenerateCount_;
}

float TokenizerStats::getAverageEncodeTime() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (encodeCount_ == 0) {
        return 0.0f;
    }
    return totalEncodeTime_ / encodeCount_;
}

float TokenizerStats::getAverageDecodeTime() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (decodeCount_ == 0) {
        return 0.0f;
    }
    return totalDecodeTime_ / decodeCount_;
}

float TokenizerStats::getAverageGenerateTime() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (generateCount_ == 0) {
        return 0.0f;
    }
    return totalGenerateTime_ / generateCount_;
}

float TokenizerStats::getAverageStreamGenerateTime() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (streamGenerateCount_ == 0) {
        return 0.0f;
    }
    return totalStreamGenerateTime_ / streamGenerateCount_;
}

long long TokenizerStats::getTotalGeneratedTokens() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return totalGeneratedTokens_;
}

float TokenizerStats::getAverageTokensPerSecond() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (totalGenerateTime_ == 0.0f) {
        return 0.0f;
    }
    return totalGeneratedTokens_ / totalGenerateTime_;
}

void TokenizerStats::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    encodeCount_ = 0;
    decodeCount_ = 0;
    generateCount_ = 0;
    streamGenerateCount_ = 0;
    totalEncodeTime_ = 0.0f;
    totalDecodeTime_ = 0.0f;
    totalGenerateTime_ = 0.0f;
    totalStreamGenerateTime_ = 0.0f;
    totalGeneratedTokens_ = 0;
}

}
