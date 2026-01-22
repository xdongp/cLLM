#include "cllm/batch/manager.h"
#include "cllm/model/executor.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <numeric>
#include <stdexcept>

namespace cllm {

BatchManager::BatchManager(size_t maxContextLength, size_t maxBatchSize)
    : maxContextLength_((maxContextLength != 0) ? maxContextLength : Config::instance().serverMaxContextLength())
    , maxBatchSize_((maxBatchSize != 0) ? maxBatchSize : Config::instance().serverMaxBatchSize())
    , contextUsageThreshold_(Config::instance().schedulerContextUsageThreshold())
    , executor_(nullptr)
    , lastBatchProcessingTimeMs_(0)
    , adaptiveBatchSize_(8)
    , minAdaptiveBatchSize_(1)
    , maxAdaptiveBatchSize_(64) {
}

BatchManager::BatchManager(size_t maxContextLength, size_t maxBatchSize, ModelExecutor* executor)
    : maxContextLength_((maxContextLength != 0) ? maxContextLength : Config::instance().serverMaxContextLength())
    , maxBatchSize_((maxBatchSize != 0) ? maxBatchSize : Config::instance().serverMaxBatchSize())
    , contextUsageThreshold_(Config::instance().schedulerContextUsageThreshold())
    , executor_(executor)
    , lastBatchProcessingTimeMs_(0)
    , adaptiveBatchSize_(8)
    , minAdaptiveBatchSize_(1)
    , maxAdaptiveBatchSize_(64) {
}

BatchManager::~BatchManager() {
}

std::vector<RequestState> BatchManager::formBatch(
    const std::vector<RequestState>& pendingRequests,
    const std::vector<RequestState>& runningRequests,
    size_t availableSeqIds
) {
    std::vector<RequestState> batch;
    size_t currentBatchLength = 0;
    
    // 🔥 瓶颈分析: 详细日志记录批处理形成过程
    CLLM_DEBUG("[BatchManager::formBatch] === 批处理形成分析 ===");
    CLLM_DEBUG("[BatchManager::formBatch] pendingRequests.size()=%zu, runningRequests.size()=%zu, availableSeqIds=%zu",
              pendingRequests.size(), runningRequests.size(), availableSeqIds);
    
    size_t runningLength = calculateRunningRequestsLength(runningRequests);
    CLLM_DEBUG("[BatchManager::formBatch] runningLength=%zu, maxContextLength_=%zu, threshold=%.1f%%",
              runningLength, maxContextLength_, (runningLength * 100.0 / maxContextLength_));
    
    // 🔥 优化: 放宽运行中请求的长度限制，允许更多并发
    // 之前过于保守，导致无法形成新的批处理
    if (runningLength > maxContextLength_ * 0.9) {  // 从0.75放宽到0.9
        CLLM_DEBUG("[BatchManager] Running length (%zu) > 90%% of maxContextLength_ (%zu), skipping batch formation",
                  runningLength, maxContextLength_);
        return batch;
    }
    
    size_t avgLength = calculateAverageRequestLength(pendingRequests);
    size_t dynamicBatchSize = calculateOptimalBatchSize(pendingRequests, avgLength);
    
    // 🔥 禁用动态批处理：由于性能严重下降，暂时禁用adaptiveBatchSize
    // size_t adaptiveSize = adaptiveBatchSize(pendingRequests.size(), runningRequests.size());
    // dynamicBatchSize = std::min(dynamicBatchSize, adaptiveSize);
    
    CLLM_DEBUG("[BatchManager::formBatch] avgLength=%zu, calculated dynamicBatchSize=%zu, maxBatchSize_=%zu",
              avgLength, dynamicBatchSize, maxBatchSize_);
    
    // 🔥 优化: 考虑序列ID可用性，但更灵活
    // 关键：即使availableSeqIds较小，也允许形成较大的批处理（因为序列ID会在请求完成时立即释放）
    if (availableSeqIds > 0) {
        // 🔥 关键优化: 不要过度限制批处理大小
        // 如果availableSeqIds较小，可能是暂时的（有请求正在完成），允许稍微超过
        // 只有在availableSeqIds非常小（<4）时才限制
        if (availableSeqIds < 4) {
            CLLM_DEBUG("[BatchManager] availableSeqIds (%zu) < 4, limiting dynamicBatchSize to %zu",
                      availableSeqIds, availableSeqIds);
            dynamicBatchSize = std::min(dynamicBatchSize, availableSeqIds);
        } else {
            // availableSeqIds >= 4，允许dynamicBatchSize稍微超过（最多1.5倍）
            // 因为序列ID会在请求完成时立即释放
            size_t maxAllowed = static_cast<size_t>(availableSeqIds * 1.5);
            if (dynamicBatchSize > maxAllowed) {
                CLLM_DEBUG("[BatchManager] dynamicBatchSize (%zu) > maxAllowed (%zu), limiting to %zu",
                          dynamicBatchSize, maxAllowed, maxAllowed);
                dynamicBatchSize = maxAllowed;
            }
        }
    }
    
    // 🔥 优化: 更激进的批处理大小策略，优先达到dynamicBatchSize
    for (const auto& request : pendingRequests) {
        size_t requestLength = request.getTotalLength();
        size_t totalLength = runningLength + currentBatchLength + requestLength;
        
        // 🔥 关键优化: 非常激进的批处理形成策略（参考llama-bench的直接方式）
        // 1. 优先达到dynamicBatchSize（充分利用GPU）
        // 2. 允许上下文长度大幅超限（最多50%）
        // 3. 小批处理时非常宽松的限制
        bool withinContext = (totalLength <= maxContextLength_);
        bool withinBatchSize = (batch.size() < dynamicBatchSize);
        bool contextBoost = (totalLength <= maxContextLength_ * 1.5);  // 允许50%超限（非常激进）
        bool smallBatchBoost = (batch.size() < 16 && contextBoost);  // 小批处理时允许50%超限
        
        // 🔥 关键: 只要在批处理大小限制内，就允许加入（即使上下文长度超限）
        if (withinBatchSize && (withinContext || contextBoost)) {
            batch.push_back(request);
            currentBatchLength += requestLength;
            CLLM_DEBUG("[BatchManager] ✓ Added request to batch: batchSize=%zu, requestLength=%zu, totalLength=%zu/%zu, dynamicBatchSize=%zu",
                      batch.size(), requestLength, totalLength, maxContextLength_, dynamicBatchSize);
        } else {
            CLLM_DEBUG("[BatchManager] ✗ Stopped adding requests: batchSize=%zu, totalLength=%zu/%zu, dynamicBatchSize=%zu",
                      batch.size(), totalLength, maxContextLength_, dynamicBatchSize);
            CLLM_DEBUG("[BatchManager]   原因: withinBatchSize=%d, withinContext=%d, contextBoost=%d",
                      withinBatchSize, withinContext, contextBoost);
            break;
        }
    }
    
    CLLM_DEBUG("[BatchManager::formBatch] === 批处理形成完成: size=%zu, totalLength=%zu ===",
              batch.size(), currentBatchLength);
    
    updateStats(batch);
    return batch;
}

std::vector<RequestState> BatchManager::formMultipleBatches(
    const std::vector<RequestState>& pendingRequests,
    const std::vector<RequestState>& runningRequests
) {
    std::vector<RequestState> allBatches;
    std::vector<RequestState> remaining = pendingRequests;
    
    size_t runningLength = calculateRunningRequestsLength(runningRequests);
    
    while (!remaining.empty()) {
        std::vector<RequestState> batch;
        size_t currentBatchLength = 0;
        
        size_t avgLength = calculateAverageRequestLength(remaining);
        size_t dynamicBatchSize = calculateOptimalBatchSize(remaining, avgLength);
        
        std::vector<size_t> usedIndices;
        
        for (size_t i = 0; i < remaining.size(); ++i) {
            const auto& request = remaining[i];
            size_t requestLength = request.getTotalLength();
            size_t totalLength = runningLength + currentBatchLength + requestLength;
            
            if (totalLength <= maxContextLength_ && 
                batch.size() < dynamicBatchSize) {
                batch.push_back(request);
                currentBatchLength += requestLength;
                usedIndices.push_back(i);
            }
        }
        
        if (batch.empty()) {
            break;
        }
        
        allBatches.insert(allBatches.end(), batch.begin(), batch.end());
        runningLength += currentBatchLength;
        
        std::vector<RequestState> newRemaining;
        for (size_t i = 0; i < remaining.size(); ++i) {
            if (std::find(usedIndices.begin(), usedIndices.end(), i) == usedIndices.end()) {
                newRemaining.push_back(remaining[i]);
            }
        }
        remaining = newRemaining;
    }
    
    return allBatches;
}

BatchInput BatchManager::prepareBatchInput(const std::vector<RequestState>& batch) {
    BatchInput input;
    input.batchSize = batch.size();
    
    // 🔥 优化：预分配内存，减少重新分配开销
    size_t totalTokens = 0;
    for (const auto& request : batch) {
        totalTokens += request.getTotalLength();
    }
    input.inputIds.reserve(totalTokens);
    input.requestPositions.reserve(batch.size());
    input.sequenceIds.reserve(batch.size());
    
    size_t currentPos = 0;
    
    for (const auto& request : batch) {
        // 🔥 优化：对于单token生成场景，只传入新token（增量生成）
        // 这样可以避免重新构建整个inputIds（包括prompt和所有已生成的tokens）
        size_t promptSize = request.tokenizedPrompt.size();
        size_t generatedSize = request.generatedTokens.size();
        
        // 🔥 关键优化：如果generatedTokens不为空，说明这是增量生成，只传入最后一个token
        // llama.cpp支持增量推理，只需要传入新token即可
        if (generatedSize > 0) {
            // 增量生成：只传入最后一个token（新生成的token）
            input.inputIds.push_back(request.generatedTokens.back());
            input.requestPositions.push_back({currentPos, currentPos + 1});
        } else {
            // 首次生成：传入完整的prompt
            input.inputIds.insert(input.inputIds.end(), 
                                 request.tokenizedPrompt.begin(), 
                                 request.tokenizedPrompt.end());
            input.requestPositions.push_back({currentPos, currentPos + promptSize});
        }
        
        input.sequenceIds.push_back(request.requestId);
        currentPos = input.inputIds.size();
    }
    
    return input;
}

BatchInput BatchManager::prepareBatchInputIncremental(
    const std::vector<RequestState>& batch,
    const BatchInput& previousInput,
    const std::vector<size_t>& previousTokenCounts
) {
    // 🔥 优化2: 真正的增量输入准备 - 只追加新tokens，不重新复制已有tokens
    BatchInput input;
    input.batchSize = batch.size();
    
    // 检查是否可以增量更新
    bool canReuse = (previousInput.batchSize == batch.size()) && 
                    (previousTokenCounts.size() == batch.size());
    
    if (canReuse) {
        // 验证每个请求的token数量是否只增加了
        for (size_t i = 0; i < batch.size(); ++i) {
            size_t currentTokenCount = batch[i].getTotalLength();
            if (i >= previousTokenCounts.size() || 
                currentTokenCount < previousTokenCounts[i]) {
                canReuse = false;
                break;
            }
        }
    }
    
    if (canReuse && batch.size() == 1) {
        // 🔥 关键优化: 单请求场景，对于单token增量生成，直接构建只包含新token的BatchInput
        // 这样可以避免从previousInput拷贝整个vector（这是性能瓶颈）
        size_t i = 0;
        size_t currentTokenCount = batch[i].getTotalLength();
        size_t previousTokenCount = previousTokenCounts[i];
        
        if (currentTokenCount > previousTokenCount) {
            // 🔥 单token或多token增量生成：直接构建只包含新token的BatchInput（零拷贝previousInput）
            // 注意：llama.cpp支持增量推理，只需要传入新token即可
            size_t newTokensCount = currentTokenCount - previousTokenCount;
            
            input.inputIds.clear();
            input.inputIds.reserve(newTokensCount);
            
            // 只追加新生成的tokens（增量更新）
            size_t promptLength = batch[i].tokenizedPrompt.size();
            size_t generatedStartIdx = previousTokenCount - promptLength;
            
            input.inputIds.insert(input.inputIds.end(),
                                 batch[i].generatedTokens.begin() + generatedStartIdx,
                                 batch[i].generatedTokens.end());
            
            input.requestPositions = {{0, newTokensCount}};  // 只有新tokens
            input.sequenceIds = previousInput.sequenceIds;  // 重用sequenceIds（避免拷贝）
            input.batchSize = 1;
            
            #ifdef CLLM_DEBUG_MODE
            CLLM_DEBUG("[BatchManager] Incremental batch input prepared (single request, %zu new tokens, zero-copy previousInput)", newTokensCount);
            #endif
            return input;
        }
        // 如果没有新增tokens，继续下面的逻辑
    }
    
    if (canReuse) {
        // 🔥 关键优化: 对于多请求场景，只构建新tokens，不重用previousInput.inputIds
        // 这样可以避免拷贝整个previousInput.inputIds（可能包含大量tokens）
        // llama.cpp支持增量推理，只需要传入新tokens即可
        
        // 🔥 优化：计算总的新tokens数量，预分配内存
        size_t totalNewTokens = 0;
        for (size_t i = 0; i < batch.size(); ++i) {
            size_t currentTokenCount = batch[i].getTotalLength();
            size_t previousTokenCount = previousTokenCounts[i];
            if (currentTokenCount > previousTokenCount) {
                totalNewTokens += (currentTokenCount - previousTokenCount);
            }
        }
        
        input.inputIds.clear();
        input.inputIds.reserve(totalNewTokens);
        input.requestPositions.clear();
        input.requestPositions.reserve(batch.size());
        input.sequenceIds = previousInput.sequenceIds;  // 重用sequenceIds（避免拷贝）
        
        size_t currentPos = 0;
        
        for (size_t i = 0; i < batch.size(); ++i) {
            size_t currentTokenCount = batch[i].getTotalLength();
            size_t previousTokenCount = previousTokenCounts[i];
            
            // 🔥 优化: 如果只是新增了tokens，只追加新部分
            if (currentTokenCount > previousTokenCount) {
                // 有新增tokens，只追加新token
                size_t promptLength = batch[i].tokenizedPrompt.size();
                size_t generatedStartIdx = previousTokenCount - promptLength;
                
                // 只追加新生成的tokens（增量更新）
                input.inputIds.insert(input.inputIds.end(),
                                     batch[i].generatedTokens.begin() + generatedStartIdx,
                                     batch[i].generatedTokens.end());
                
                // 更新requestPositions（只包含新tokens的位置）
                size_t newTokensCount = currentTokenCount - previousTokenCount;
                input.requestPositions.push_back({currentPos, currentPos + newTokensCount});
                currentPos += newTokensCount;
            } else {
                // 如果没有新增tokens，requestPositions为空（表示该请求已完成或不需要处理）
                input.requestPositions.push_back({currentPos, currentPos});
            }
        }
        
        #ifdef CLLM_DEBUG_MODE
        CLLM_DEBUG("[BatchManager] Incremental batch input prepared (new tokens only, %zu new tokens, zero-copy previousInput)",
                  input.inputIds.size());
        #endif
    } else {
        // 无法重用，完整重新构建
        input = prepareBatchInput(batch);
        #ifdef CLLM_DEBUG_MODE
        CLLM_DEBUG("[BatchManager] Full batch input prepared (cannot reuse)");
        #endif
    }
    
    return input;
}

void BatchManager::processBatchOutput(
    std::vector<RequestState>& batch,
    const BatchOutput& output
) {
    for (size_t i = 0; i < batch.size(); ++i) {
        if (batch[i].isCompleted) {
            continue;
        }
        
        size_t vocabSize = executor_ ? executor_->getConfig().vocabSize : 32000;
        FloatArray requestLogits = output.getLogitsForRequest(i, vocabSize);
        
        float temperature = batch[i].temperature;
        int topK = batch[i].topK;
        float topP = batch[i].topP;
        
        int nextToken = sampler_.sample(requestLogits, temperature, topK, topP);
        
        batch[i].generatedTokens.push_back(nextToken);
        
        checkStoppingConditions(batch[i], nextToken);
    }
}

size_t BatchManager::calculateOptimalBatchSize(
    const std::vector<RequestState>& requests,
    size_t avgRequestLength
) {
    if (requests.empty()) {
        return 0;
    }
    
    // 🔥 优化2: 非常激进的批处理大小计算策略
    // 目标：充分利用GPU并行能力，尽可能形成大批处理
    // 关键：不要因为请求长度而过度限制批处理大小
    size_t dynamicBatchSize = maxBatchSize_;
    
    // 🔥 关键优化: 根据平均请求长度调整，但非常激进
    // 即使请求较长，也要允许非常大的批处理，充分利用GPU
    if (avgRequestLength > 500) {
        // ⚠️ 优化前: 最多2个请求（过于保守）
        // 🔥 优化后: 至少32个请求（非常激进），充分利用GPU并行能力
        dynamicBatchSize = std::max(size_t(32), maxBatchSize_);  // 至少32个
        CLLM_DEBUG("[BatchManager] avgLength > 500: dynamicBatchSize = %zu (min 32, maxBatchSize_=%zu)", 
                  dynamicBatchSize, maxBatchSize_);
    } else if (avgRequestLength > 200) {
        // 🔥 优化后: 至少48个请求
        dynamicBatchSize = std::max(size_t(48), static_cast<size_t>(maxBatchSize_ * 2));
        CLLM_DEBUG("[BatchManager] avgLength > 200: dynamicBatchSize = %zu (min 48)", dynamicBatchSize);
    } else {
        // 小请求，可以使用更大的批处理（maxBatchSize_的两倍或更多）
        dynamicBatchSize = std::min(static_cast<size_t>(maxBatchSize_ * 3), requests.size());
        CLLM_DEBUG("[BatchManager] avgLength <= 200: dynamicBatchSize = %zu", dynamicBatchSize);
    }
    
    // 确保不超过请求数量
    dynamicBatchSize = std::min(dynamicBatchSize, requests.size());
    
    CLLM_DEBUG("[BatchManager] calculateOptimalBatchSize: avgLength=%zu, dynamicBatchSize=%zu, maxBatchSize_=%zu, requests.size()=%zu",
              avgRequestLength, dynamicBatchSize, maxBatchSize_, requests.size());
    
    return dynamicBatchSize;
}

size_t BatchManager::adaptiveBatchSize(size_t queueSize, size_t runningCount) {
    if (lastBatchProcessingTimeMs_ > 500) {
        adaptiveBatchSize_ = std::max(minAdaptiveBatchSize_, adaptiveBatchSize_ / 2);
        CLLM_DEBUG("[BatchManager::adaptiveBatchSize] Last batch processing time too long (%zu ms), reducing batch size to %zu",
                  lastBatchProcessingTimeMs_, adaptiveBatchSize_);
    } else if (lastBatchProcessingTimeMs_ < 100 && queueSize > adaptiveBatchSize_ * 2) {
        adaptiveBatchSize_ = std::min(maxAdaptiveBatchSize_, adaptiveBatchSize_ * 2);
        CLLM_DEBUG("[BatchManager::adaptiveBatchSize] Last batch processing time short (%zu ms) and queue large (%zu), increasing batch size to %zu",
                  lastBatchProcessingTimeMs_, queueSize, adaptiveBatchSize_);
    }
    
    return adaptiveBatchSize_;
}

void BatchManager::updateBatchProcessingTime(size_t processingTimeMs) {
    lastBatchProcessingTimeMs_ = processingTimeMs;
    CLLM_DEBUG("[BatchManager::updateBatchProcessingTime] Updated batch processing time to %zu ms", processingTimeMs);
}

bool BatchManager::canAddToBatch(
    const RequestState& request,
    const std::vector<RequestState>& currentBatch,
    size_t currentBatchLength,
    size_t dynamicBatchSize
) {
    if (currentBatch.size() >= dynamicBatchSize) {
        return false;
    }
    
    size_t requestLength = request.getTotalLength();
    size_t totalLength = currentBatchLength + requestLength;
    
    return totalLength <= maxContextLength_;
}

BatchStats BatchManager::getStats() const {
    return stats_;
}

void BatchManager::resetStats() {
    stats_.reset();
}

size_t BatchManager::calculateRunningRequestsLength(
    const std::vector<RequestState>& runningRequests
) {
    size_t totalLength = 0;
    for (const auto& request : runningRequests) {
        totalLength += request.getTotalLength();
    }
    return totalLength;
}

size_t BatchManager::calculateAverageRequestLength(
    const std::vector<RequestState>& requests
) {
    if (requests.empty()) {
        return 0;
    }
    
    // 🔥 优化: 使用更激进的估算，避免过度限制批处理大小
    // 对于小批处理，使用较小的平均值；对于大批处理，使用较大的平均值
    size_t totalLength = 0;
    size_t minLength = SIZE_MAX;
    size_t maxLength = 0;
    
    for (const auto& request : requests) {
        size_t len = request.getTotalLength();
        totalLength += len;
        if (len < minLength) minLength = len;
        if (len > maxLength) maxLength = len;
    }
    
    size_t avgLength = totalLength / requests.size();
    
    // 🔥 关键优化: 对于小批处理，使用更小的平均值（避免过度限制）
    // 对于大批处理，使用更大的平均值（充分利用GPU）
    if (requests.size() <= 4) {
        // 小批处理：使用最小值和平均值的中间值
        avgLength = (minLength + avgLength) / 2;
    } else {
        // 大批处理：使用平均值和最大值的中间值（更激进）
        avgLength = (avgLength + maxLength) / 2;
    }
    
    return avgLength;
}

void BatchManager::updateStats(const std::vector<RequestState>& batch) {
    if (batch.empty()) {
        return;
    }
    
    size_t batchLength = 0;
    for (const auto& request : batch) {
        batchLength += request.getTotalLength();
    }
    
    stats_.update(batch.size(), batchLength);
}

void BatchManager::checkStoppingConditions(RequestState& request, int nextToken) {
    if (request.generatedTokens.size() >= static_cast<size_t>(request.maxTokens)) {
        request.isCompleted = true;
        return;
    }

    // 使用请求注入的 EOS token id（避免写死 2/0 导致不同模型行为错误）
    if (request.eosTokenId >= 0 && nextToken == request.eosTokenId) {
        request.isCompleted = true;
        return;
    }
}



}
