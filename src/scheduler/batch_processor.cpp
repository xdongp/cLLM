#include "cllm/scheduler/batch_processor.h"
#include <cstring>
#include "cllm/common/request_state.h"
#include "cllm/scheduler/scheduler.h"
#include "cllm/batch/manager.h"
#include "cllm/sampler.h"
#include "cllm/model/executor.h"
#include "cllm/common/logger.h"
#include "cllm/common/config.h"
#include <algorithm>
#include <chrono>

namespace cllm {

SchedulerBatchProcessor::SchedulerBatchProcessor(
    Scheduler* scheduler,
    ModelExecutor* executor,
    KVCache* cache,
    BatchManager* batchManager
) : scheduler_(scheduler), executor_(executor), cache_(cache), batchManager_(batchManager) {
}

SchedulerBatchProcessor::~SchedulerBatchProcessor() {
}

void SchedulerBatchProcessor::processBatch(std::vector<RequestState>& batch) {
    const int MAX_ITERATIONS = Config::instance().schedulerMaxIterations(); // 防止无限循环
    int iterationCount = 0;
    
    auto batchStartTime = std::chrono::steady_clock::now();
    
    // 🔥 优化：减少日志输出（在生产环境中关闭详细日志）
    #ifdef CLLM_DEBUG_MODE
    CLLM_DEBUG("processBatch: Starting batch processing with %zu requests", batch.size());
    for (size_t i = 0; i < batch.size(); ++i) {
        CLLM_DEBUG("processBatch: Request %zu - ID=%llu, maxTokens=%d, generatedTokens=%zu, isCompleted=%d, isFailed=%d",
                  i, batch[i].requestId, batch[i].maxTokens, batch[i].generatedTokens.size(),
                  batch[i].isCompleted ? 1 : 0, batch[i].isFailed ? 1 : 0);
    }
    #endif
    
    // 🔥 优化2: 初始化缓存（新批处理开始时清空）
    cachedBatchInput_.clear();
    cachedTokenCounts_.clear();
    cachedRequestIds_.clear();
    
    // 🔥 优化1: 动态批处理重组阈值（当活跃请求数 < 30% 时考虑重组）
    // 修复：更积极的重组策略，及时将慢速请求与新请求重组，减少响应时间长尾
    constexpr double BATCH_REGROUP_THRESHOLD = 0.3;
    constexpr size_t MIN_EFFICIENT_BATCH_SIZE = 6;  // 修复：增加最小批处理大小，避免过度频繁重组
    
    while (!isBatchComplete(batch)) {
        auto activeRequests = getActiveRequests(batch);
        
        if (activeRequests.empty()) {
            #ifdef CLLM_DEBUG_MODE
            CLLM_DEBUG("processBatch: No active requests, breaking loop");
            #endif
            break;
        }
        
        // 🔥 优化3: 动态批处理重组 - 如果活跃请求数 < 批处理大小的30%，提前结束
        // 修复：更积极的重组策略，当批处理效率下降时及时重组，避免慢速请求阻塞整个批处理
        if (activeRequests.size() < batch.size() * BATCH_REGROUP_THRESHOLD) {
            CLLM_DEBUG("processBatch: Active requests (%zu) < 30%% of batch size (%zu), batch efficiency degraded", 
                      activeRequests.size(), batch.size());
            
            // 🔥 关键修复: 当批处理效率过低时，提前结束当前批处理
            // 将剩余的活跃请求返回给Scheduler，让它可以与新到达的请求重组
            // 这样可以避免慢速请求一直占用批处理资源，导致响应时间长尾
            if (activeRequests.size() <= 3) {
                CLLM_INFO("processBatch: Batch efficiency too low (%zu/%zu), breaking to allow regrouping with new requests", 
                         activeRequests.size(), batch.size());
                
                // 🔥 关键修复：在提前结束前，确保所有请求的状态都被正确更新
                // 检查所有请求是否已经达到maxTokens限制，如果是则标记为完成
                for (auto& req : batch) {
                    if (!req.isCompleted && !req.isFailed && 
                        req.generatedTokens.size() >= static_cast<size_t>(req.maxTokens)) {
                        CLLM_DEBUG("processBatch: Request %llu reached max tokens limit (%zu >= %d) before batch end, marking as completed",
                                  req.requestId, req.generatedTokens.size(), req.maxTokens);
                        req.isCompleted = true;
                        
                        // Phase 7: 触发完成回调
                        if (scheduler_) {
                            scheduler_->triggerResponseCallback(req.requestId, req);
                        }
                    }
                }
                
                // 提前结束，剩余的活跃请求会在下次调度时与新请求重组
                break;
            }
        }
        
        // 超时保护
        if (++iterationCount >= MAX_ITERATIONS) {
            CLLM_WARN("Reached max iterations (%d), marking all active requests as failed", MAX_ITERATIONS);
            for (auto& req : batch) {
                if (!req.isCompleted && !req.isFailed) {
                    req.isFailed = true;
                }
            }
            break;
        }
        
        CLLM_DEBUG("processBatch: Iteration %d, active requests: %zu (batch size: %zu)", 
                  iterationCount, activeRequests.size(), batch.size());
        
        // 🔥 优化1: 传递已计算的活跃请求，避免在 processIteration 中重复计算
        processIteration(batch, activeRequests);
    }
    
    #ifdef CLLM_DEBUG_MODE
    CLLM_DEBUG("Batch processing completed after %d iterations", iterationCount);
    
    // 调试：记录最终状态
    for (size_t i = 0; i < batch.size(); ++i) {
        CLLM_DEBUG("processBatch: Final state - Request %zu - ID=%llu, generatedTokens=%zu, isCompleted=%d, isFailed=%d",
                  i, batch[i].requestId, batch[i].generatedTokens.size(),
                  batch[i].isCompleted ? 1 : 0, batch[i].isFailed ? 1 : 0);
    }
    #endif
    
    // 🔥 优化: 记录批处理时间并更新自适应批处理大小
    auto batchEndTime = std::chrono::steady_clock::now();
    auto processingTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        batchEndTime - batchStartTime
    ).count();
    
    if (batchManager_) {
        batchManager_->updateBatchProcessingTime(processingTimeMs);
        CLLM_DEBUG("processBatch: Batch processing time: %zu ms, batch size: %zu", 
                  processingTimeMs, batch.size());
    }
}

bool SchedulerBatchProcessor::isBatchComplete(const std::vector<RequestState>& batch) const {
    for (size_t i = 0; i < batch.size(); ++i) {
        const auto& req = batch[i];
        bool completed = req.isCompleted || req.isFailed || 
                        req.generatedTokens.size() >= static_cast<size_t>(req.maxTokens);
        CLLM_DEBUG("isBatchComplete - Request %zu (ID=%llu): isCompleted=%d, isFailed=%d, generatedTokens=%zu, maxTokens=%d, completed=%d", 
                  i, req.requestId, req.isCompleted, req.isFailed, req.generatedTokens.size(), req.maxTokens, completed);
        if (!completed) {
            CLLM_DEBUG("isBatchComplete - Request %zu (ID=%llu) is NOT complete, batch continues", i, req.requestId);
            return false;
        }
    }
    CLLM_DEBUG("isBatchComplete - All requests are complete");
    return true;
}

std::vector<RequestState> SchedulerBatchProcessor::getActiveRequests(
    const std::vector<RequestState>& batch
) const {
    std::vector<RequestState> active;
    
    for (const auto& req : batch) {
        bool isActive = !req.isCompleted && !req.isFailed && 
                       req.generatedTokens.size() < static_cast<size_t>(req.maxTokens);
        if (isActive) {
            active.push_back(req);
            CLLM_DEBUG("getActiveRequests - Request ID=%llu is active (generatedTokens=%zu, maxTokens=%d)",
                      req.requestId, req.generatedTokens.size(), req.maxTokens);
        } else {
            CLLM_DEBUG("getActiveRequests - Request ID=%llu is NOT active (isCompleted=%d, isFailed=%d, generatedTokens=%zu, maxTokens=%d)",
                      req.requestId, req.isCompleted ? 1 : 0, req.isFailed ? 1 : 0,
                      req.generatedTokens.size(), req.maxTokens);
        }
    }
    
    CLLM_DEBUG("getActiveRequests - Found %zu active requests out of %zu total", active.size(), batch.size());
    return active;
}

void SchedulerBatchProcessor::processIteration(
    std::vector<RequestState>& batch,
    const std::vector<RequestState>& activeRequests
) {
    CLLM_DEBUG("processIteration called with batch size: %zu, active requests: %zu", 
              batch.size(), activeRequests.size());
    
    // 🔥 优化1: 直接使用传入的活跃请求，避免重复计算
    if (activeRequests.empty()) {
        CLLM_DEBUG("No active requests, returning");
        return;
    }
    
    // 🔥 优化: 减少调试日志输出（在生产环境中关闭详细日志）
    // 只在DEBUG级别输出关键信息
    CLLM_DEBUG("processIteration: %zu active requests", activeRequests.size());
    
    // 🔥 优化2: 增量准备批处理输入（只更新新增的tokens）
    BatchInput input;
    
    // 🔥 关键优化: 对于单请求、单token场景，直接构建BatchInput，跳过BatchManager的复杂逻辑
    if (activeRequests.size() == 1) {
        const auto& req = activeRequests[0];
        size_t currentTokenCount = req.getTotalLength();
        
        // 检查是否是增量生成（有已生成的tokens）
        if (!req.generatedTokens.empty()) {
            // 🔥 单token增量生成：直接构建只包含新token的BatchInput（完全跳过BatchManager）
            // llama.cpp支持增量推理，只需要传入新token即可
            input.inputIds.clear();
            input.inputIds.push_back(req.generatedTokens.back());  // 只包含最后一个token（新token）
            input.requestPositions = {{0, 1}};
            input.sequenceIds = {req.requestId};
            input.batchSize = 1;
            
            // 更新缓存（用于后续迭代）
            cachedTokenCounts_.clear();
            cachedTokenCounts_.push_back(currentTokenCount);
            cachedRequestIds_.clear();
            cachedRequestIds_.push_back(req.requestId);
            cachedBatchInput_ = input;  // 缓存用于下次迭代
            
            CLLM_DEBUG("Using direct batch input preparation (single request, single token, bypass BatchManager)");
        } else {
            // 首次生成：使用BatchManager准备完整prompt
            input = batchManager_->prepareBatchInput(activeRequests);
            
            // 初始化缓存
            cachedBatchInput_ = input;
            cachedTokenCounts_.clear();
            cachedRequestIds_.clear();
            cachedTokenCounts_.push_back(currentTokenCount);
            cachedRequestIds_.push_back(req.requestId);
            
            CLLM_DEBUG("Using full batch input preparation (first iteration, single request)");
        }
    } else if (!cachedBatchInput_.empty() && cachedRequestIds_.size() == activeRequests.size()) {
        // 多请求场景：检查请求ID是否匹配（验证是否是同一个批处理）
        bool idsMatch = true;
        for (size_t i = 0; i < activeRequests.size() && i < cachedRequestIds_.size(); ++i) {
            if (activeRequests[i].requestId != cachedRequestIds_[i]) {
                idsMatch = false;
                break;
            }
        }
        
        if (idsMatch) {
            // 计算当前每个请求的token数量
            std::vector<size_t> currentTokenCounts;
            currentTokenCounts.reserve(activeRequests.size());
            for (const auto& req : activeRequests) {
                currentTokenCounts.push_back(req.getTotalLength());
            }
            
            // 使用增量准备
            input = batchManager_->prepareBatchInputIncremental(
                activeRequests, 
                cachedBatchInput_, 
                cachedTokenCounts_
            );
            
            // 更新缓存
            cachedTokenCounts_ = currentTokenCounts;
            cachedBatchInput_ = input;
            
            CLLM_DEBUG("Using incremental batch input preparation");
        } else {
            // 请求ID不匹配，完整重新构建
            input = batchManager_->prepareBatchInput(activeRequests);
            
            // 更新缓存
            cachedBatchInput_ = input;
            cachedTokenCounts_.clear();
            cachedRequestIds_.clear();
            for (const auto& req : activeRequests) {
                cachedTokenCounts_.push_back(req.getTotalLength());
                cachedRequestIds_.push_back(req.requestId);
            }
            
            CLLM_DEBUG("Using full batch input preparation (request IDs changed)");
        }
    } else {
        // 首次迭代，完整构建
        input = batchManager_->prepareBatchInput(activeRequests);
        
        // 初始化缓存
        cachedBatchInput_ = input;
        cachedTokenCounts_.clear();
        cachedRequestIds_.clear();
        for (const auto& req : activeRequests) {
            cachedTokenCounts_.push_back(req.getTotalLength());
            cachedRequestIds_.push_back(req.requestId);
        }
        
        CLLM_DEBUG("Using full batch input preparation (first iteration)");
    }
    
    CLLM_DEBUG("BatchInput prepared:");
    CLLM_DEBUG("  Batch size: %zu", input.batchSize);
    CLLM_DEBUG("  Input IDs size: %zu", input.inputIds.size());
    CLLM_DEBUG("  Request positions size: %zu", input.requestPositions.size());
    CLLM_DEBUG("  Sequence IDs size: %zu", input.sequenceIds.size());
    
    std::string inputIdsStr;
    for (size_t i = 0; i < std::min(input.inputIds.size(), (size_t)20); ++i) {
        inputIdsStr += " " + std::to_string(input.inputIds[i]);
    }
    if (input.inputIds.size() > 20) {
        inputIdsStr += " ...";
    }
    CLLM_DEBUG("  Input IDs: [%s]", inputIdsStr.c_str());
    
    // 🔥 关键修复：在调用executor->forward()之前，确保为每个requestId分配sequence ID
    // 这对于新请求是必需的，对于已存在的请求，getSequenceId会返回已分配的ID
    // 注意：LlamaCppBackend::forwardBatch()内部会自动分配sequence ID，但我们需要确保
    // 在首次调用前就分配好，以便正确跟踪位置
    // 实际上，forwardBatch()内部已经处理了sequence ID分配，所以这里不需要额外处理
    // 问题可能在于位置计算，让我们确保BatchInput的requestPositions正确设置
    
    #ifdef CLLM_DEBUG_MODE
    CLLM_DEBUG("Calling executor_->forward(input)...");
    #endif
    BatchOutput output = executor_->forward(input);
    
    #ifdef CLLM_DEBUG_MODE
    CLLM_DEBUG("Model forward pass completed");
    #endif
    
    CLLM_DEBUG("Calling updateRequestStates...");
    updateRequestStates(batch, output);
    
    CLLM_DEBUG("processIteration completed");
}

void SchedulerBatchProcessor::updateRequestStates(
    std::vector<RequestState>& batch,
    const BatchOutput& output
) {
    CLLM_DEBUG("updateRequestStates called with batch size: %zu", batch.size());
    
    Sampler sampler;
    
    // 创建batch索引到output索引的映射
    std::vector<size_t> batchToOutputIndex;
    size_t outputIndex = 0;
    for (size_t i = 0; i < batch.size(); ++i) {
        if (!batch[i].isCompleted && !batch[i].isFailed && 
            batch[i].generatedTokens.size() < batch[i].maxTokens) {
            batchToOutputIndex.push_back(i);
        }
    }
    
    CLLM_DEBUG("Active requests in output: %zu", batchToOutputIndex.size());
    
    for (size_t activeIdx = 0; activeIdx < batchToOutputIndex.size(); ++activeIdx) {
        size_t i = batchToOutputIndex[activeIdx];
        CLLM_DEBUG("Processing request %zu (output index %zu)", i, activeIdx);
        
        if (batch[i].isCompleted || batch[i].isFailed) {
            CLLM_DEBUG("Request %zu is completed or failed, skipping", i);
            continue;
        }
        
        if (batch[i].generatedTokens.size() >= batch[i].maxTokens) {
            CLLM_DEBUG("Request %zu reached max tokens limit BEFORE generation (%zu >= %d), marking as completed", 
                      i, batch[i].generatedTokens.size(), batch[i].maxTokens);
            batch[i].isCompleted = true;
            
            // Phase 7: 触发完成回调
            if (scheduler_) {
                scheduler_->triggerResponseCallback(batch[i].requestId, batch[i]);
            }
            continue;
        }
        
        CLLM_DEBUG("Request %zu - BEFORE generation: generatedTokens=%zu, maxTokens=%d", 
                  i, batch[i].generatedTokens.size(), batch[i].maxTokens);
        
        CLLM_DEBUG("Request %zu - Getting logits from output (using output index %zu)", i, activeIdx);
        
        // 从 ModelExecutor 获取模型的 vocab size（用于正确提取 logits）
        size_t modelVocabSize = executor_ ? executor_->getConfig().vocabSize : 32000;
        FloatArray fullLogits = output.getLogitsForRequest(activeIdx, modelVocabSize);
        
        CLLM_DEBUG("Request %zu - Full logits size: %zu (model vocab_size)", i, fullLogits.size());
        
        if (fullLogits.empty()) {
            CLLM_ERROR("Request %zu got empty logits from model!", i);
            batch[i].isFailed = true;
            
            // Phase 7: 触发失败回调
            if (scheduler_) {
                scheduler_->triggerResponseCallback(batch[i].requestId, batch[i]);
            }
            continue;
        }
        
        // 在采样前将 logits 裁剪到 tokenizer 的 vocab_size
        // 这是根本修复：确保采样只从 tokenizer 的有效范围内选择 token
        size_t tokenizerVocabSize = executor_ ? executor_->getConfig().tokenizerVocabSize : fullLogits.size();
        if (tokenizerVocabSize == 0) {
            // 如果未设置，默认使用模型的 vocab_size（向后兼容）
            tokenizerVocabSize = fullLogits.size();
        }
        FloatArray logits(std::min(fullLogits.size(), tokenizerVocabSize));
        // 🔥 优化：使用memcpy替代循环拷贝，提升性能
        std::memcpy(logits.data(), fullLogits.data(), logits.size() * sizeof(float));
        
        if (fullLogits.size() > tokenizerVocabSize) {
            CLLM_DEBUG("Request %zu - Clipped logits from %zu to %zu (tokenizer vocab_size)", 
                      i, fullLogits.size(), logits.size());
        }
        
        // 🔥 优化：只在DEBUG模式下统计logits信息，减少生产环境开销
        #ifdef CLLM_DEBUG_MODE
        // 统计 logits 信息
        float maxLogit = logits.empty() ? 0.0f : logits[0];
        float minLogit = logits.empty() ? 0.0f : logits[0];
        float sumLogits = 0.0f;
        size_t nonZeroCount = 0;
        size_t zeroCount = 0;
        
        for (size_t j = 0; j < logits.size(); ++j) {
            float val = logits[j];
            if (val > maxLogit) maxLogit = val;
            if (val < minLogit) minLogit = val;
            sumLogits += val;
            if (val != 0.0f) {
                nonZeroCount++;
            } else {
                zeroCount++;
            }
        }
        
        float avgLogit = logits.empty() ? 0.0f : sumLogits / logits.size();
        
        CLLM_DEBUG("Request %zu - Logits statistics: size=%zu, max=%.6f, min=%.6f, avg=%.6f, non_zero=%zu, zero=%zu",
                   i, logits.size(), maxLogit, minLogit, avgLogit, nonZeroCount, zeroCount);
        
        // 显示前 50 个 logits（如果 logits 数量大于 50）
        std::string logitsStr;
        size_t showCount = std::min(logits.size(), (size_t)50);
        for (size_t j = 0; j < showCount; ++j) {
            if (j > 0 && j % 10 == 0) {
                logitsStr += "\n  ";
            }
            logitsStr += " " + std::to_string(logits[j]);
        }
        if (logits.size() > showCount) {
            logitsStr += " ...";
        }
        CLLM_DEBUG("Request %zu - First %zu logits: [%s]", i, showCount, logitsStr.c_str());
        
        // 如果 logits 全为 0，显示警告
        if (nonZeroCount == 0) {
            CLLM_WARN("Request %zu - WARNING: All logits are zero! This will cause uniform sampling.", i);
        }
        
        // 显示最大值和最小值的位置
        if (nonZeroCount > 0) {
            size_t maxIdx = 0;
            size_t minIdx = 0;
            for (size_t j = 1; j < logits.size(); ++j) {
                if (logits[j] > logits[maxIdx]) maxIdx = j;
                if (logits[j] < logits[minIdx]) minIdx = j;
            }
            CLLM_DEBUG("Request %zu - Max logit at index %zu: %.6f, Min logit at index %zu: %.6f",
                       i, maxIdx, logits[maxIdx], minIdx, logits[minIdx]);
        }
        #endif
        
        // Get sampling parameters from request
        float temperature = batch[i].temperature;
        int topK = batch[i].topK;
        float topP = batch[i].topP;
        
        CLLM_DEBUG("Request %zu - Sampling with temp=%f, topK=%d, topP=%f", i, temperature, topK, topP);
        
        int nextToken = sampler.sample(logits, temperature, topK, topP);
        
        CLLM_DEBUG("Request %zu - Sampled token: %d", i, nextToken);
        
        if (nextToken == -1) {
            CLLM_ERROR("Sampler returned invalid token (-1) for request %zu", i);
            batch[i].isFailed = true;
            
            // Phase 7: 触发失败回调
            if (scheduler_) {
                scheduler_->triggerResponseCallback(batch[i].requestId, batch[i]);
            }
            continue;
        }
        
        // 🔥 关键修复：在生成token之前再次检查max_tokens，确保不会超出限制
        // 这是为了防止在高并发下，批处理提前结束时，未完成的请求已经生成了超过maxTokens的tokens
        if (batch[i].generatedTokens.size() >= batch[i].maxTokens) {
            CLLM_DEBUG("Request %zu reached max tokens limit BEFORE adding token (%zu >= %d), marking as completed", 
                      i, batch[i].generatedTokens.size(), batch[i].maxTokens);
            batch[i].isCompleted = true;
            
            // Phase 7: 触发完成回调
            if (scheduler_) {
                scheduler_->triggerResponseCallback(batch[i].requestId, batch[i]);
            }
            continue;
        }
        
        batch[i].generatedTokens.push_back(nextToken);
        CLLM_DEBUG("Request %zu - Generated tokens now: %zu", i, batch[i].generatedTokens.size());
        
        // 🔒 安全兜底：防止生成数量超过 maxTokens
        if (batch[i].maxTokens > 0 &&
            batch[i].generatedTokens.size() > static_cast<size_t>(batch[i].maxTokens)) {
            CLLM_WARN("Request %zu - Generated tokens exceeded maxTokens (%zu > %d), truncating",
                      i, batch[i].generatedTokens.size(), batch[i].maxTokens);
            batch[i].generatedTokens.resize(static_cast<size_t>(batch[i].maxTokens));
        }
        
        // Check if we should complete the request
        const bool eosReached = (batch[i].eosTokenId >= 0 && nextToken == batch[i].eosTokenId);
        const bool maxTokensReached = (batch[i].generatedTokens.size() >= batch[i].maxTokens);

        if (eosReached) {
            CLLM_DEBUG("Request %zu - Reached EOS token (%d), completing", i, batch[i].eosTokenId);
            batch[i].isCompleted = true;
            
            // 🔥 优化: 立即释放序列ID和KV缓存，避免阻塞后续批处理
            // 注意: 这里不能直接调用modelExecutor_，需要通过scheduler_
            // Phase 7: 触发完成回调
            if (scheduler_) {
                scheduler_->triggerResponseCallback(batch[i].requestId, batch[i]);
            }
        } else if (maxTokensReached) {
            CLLM_DEBUG("Request %zu - Reached max tokens (%zu), completing", i, batch[i].generatedTokens.size());
            batch[i].isCompleted = true;
            
            // 🔥 优化: 立即释放序列ID和KV缓存，避免阻塞后续批处理
            // Phase 7: 触发完成回调
            if (scheduler_) {
                scheduler_->triggerResponseCallback(batch[i].requestId, batch[i]);
            }
        } else {
            CLLM_DEBUG("Request %zu - Continuing generation", i);
        }
    }
    
    CLLM_DEBUG("updateRequestStates completed");
}

}
