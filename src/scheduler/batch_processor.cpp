#include "cllm/scheduler/batch_processor.h"
#include "cllm/common/request_state.h"
#include "cllm/scheduler/scheduler.h"
#include "cllm/batch/manager.h"
#include "cllm/sampler.h"
#include "cllm/model/executor.h"
#include "cllm/common/logger.h"
#include "cllm/common/config.h"
#include <algorithm>

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
    
    while (!isBatchComplete(batch)) {
        auto activeRequests = getActiveRequests(batch);
        
        if (activeRequests.empty()) {
            break;
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
        
        processIteration(batch);
    }
    
    CLLM_INFO("Batch processing completed after %d iterations", iterationCount);
}

bool SchedulerBatchProcessor::isBatchComplete(const std::vector<RequestState>& batch) const {
    for (size_t i = 0; i < batch.size(); ++i) {
        const auto& req = batch[i];
        bool completed = req.isCompleted || req.isFailed || 
                        req.generatedTokens.size() >= req.maxTokens;
        CLLM_DEBUG("isBatchComplete - Request %zu: isCompleted=%d, isFailed=%d, generatedTokens=%zu, maxTokens=%d, completed=%d", 
                  i, req.isCompleted, req.isFailed, req.generatedTokens.size(), req.maxTokens, completed);
        if (!completed) {
            return false;
        }
    }
    return true;
}

std::vector<RequestState> SchedulerBatchProcessor::getActiveRequests(
    const std::vector<RequestState>& batch
) const {
    std::vector<RequestState> active;
    
    for (const auto& req : batch) {
        if (!req.isCompleted && !req.isFailed && 
            req.generatedTokens.size() < req.maxTokens) {
            active.push_back(req);
        }
    }
    
    return active;
}

void SchedulerBatchProcessor::processIteration(std::vector<RequestState>& batch) {
    CLLM_DEBUG("processIteration called with batch size: %zu", batch.size());
    
    std::vector<RequestState> activeRequests = getActiveRequests(batch);
    
    CLLM_DEBUG("Active requests count: %zu", activeRequests.size());
    
    if (activeRequests.empty()) {
        CLLM_DEBUG("No active requests, returning");
        return;
    }
    
    // Log active requests details
    for (size_t i = 0; i < activeRequests.size(); ++i) {
        CLLM_DEBUG("Active request %zu details:", i);
        CLLM_DEBUG("  Request ID: %llu", activeRequests[i].requestId);
        CLLM_DEBUG("  Generated tokens: %zu", activeRequests[i].generatedTokens.size());
        CLLM_DEBUG("  Tokenized prompt size: %zu", activeRequests[i].tokenizedPrompt.size());
        CLLM_DEBUG("  Max tokens: %d", activeRequests[i].maxTokens);
        CLLM_DEBUG("  Temperature: %f", activeRequests[i].temperature);
        CLLM_DEBUG("  TopK: %d", activeRequests[i].topK);
        CLLM_DEBUG("  TopP: %f", activeRequests[i].topP);
        
        if (activeRequests[i].generatedTokens.empty()) {
            CLLM_DEBUG("  First iteration, using full prompt");
            std::string promptTokens;
            for (size_t j = 0; j < std::min(activeRequests[i].tokenizedPrompt.size(), (size_t)10); ++j) {
                promptTokens += " " + std::to_string(activeRequests[i].tokenizedPrompt[j]);
            }
            if (activeRequests[i].tokenizedPrompt.size() > 10) {
                promptTokens += " ...";
            }
            CLLM_DEBUG("  Prompt tokens: [%s]", promptTokens.c_str());
        } else {
            CLLM_DEBUG("  Using last token: %d", activeRequests[i].generatedTokens.back());
        }
    }
    
    CLLM_DEBUG("Calling batchManager_->prepareBatchInput...");
    BatchInput input = batchManager_->prepareBatchInput(activeRequests);
    
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
    
    CLLM_DEBUG("Calling executor_->forward(input)...");
    BatchOutput output = executor_->forward(input);
    
    CLLM_DEBUG("Model forward pass completed");
    
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
            CLLM_DEBUG("Request %zu reached max tokens limit (%zu >= %d), marking as completed", 
                      i, batch[i].generatedTokens.size(), batch[i].maxTokens);
            batch[i].isCompleted = true;
            continue;
        }
        
        CLLM_DEBUG("Request %zu - Getting logits from output (using output index %zu)", i, activeIdx);
        
        // 从 ModelExecutor 获取模型的 vocab size（用于正确提取 logits）
        size_t modelVocabSize = executor_ ? executor_->getConfig().vocabSize : 32000;
        FloatArray fullLogits = output.getLogitsForRequest(activeIdx, modelVocabSize);
        
        CLLM_DEBUG("Request %zu - Full logits size: %zu (model vocab_size)", i, fullLogits.size());
        
        if (fullLogits.empty()) {
            CLLM_ERROR("Request %zu got empty logits from model!", i);
            batch[i].isFailed = true;
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
        for (size_t j = 0; j < logits.size(); ++j) {
            logits[j] = fullLogits[j];
        }
        
        if (fullLogits.size() > tokenizerVocabSize) {
            CLLM_DEBUG("Request %zu - Clipped logits from %zu to %zu (tokenizer vocab_size)", 
                      i, fullLogits.size(), logits.size());
        }
        
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
            continue;
        }
        
        batch[i].generatedTokens.push_back(nextToken);
        CLLM_DEBUG("Request %zu - Generated tokens now: %zu", i, batch[i].generatedTokens.size());
        
        // Check if we should complete the request
        const bool eosReached = (batch[i].eosTokenId >= 0 && nextToken == batch[i].eosTokenId);
        const bool maxTokensReached = (batch[i].generatedTokens.size() >= batch[i].maxTokens);

        if (eosReached) {
            CLLM_DEBUG("Request %zu - Reached EOS token (%d), completing", i, batch[i].eosTokenId);
            batch[i].isCompleted = true;
        } else if (maxTokensReached) {
            CLLM_DEBUG("Request %zu - Reached max tokens (%zu), completing", i, batch[i].generatedTokens.size());
            batch[i].isCompleted = true;
        } else {
            CLLM_DEBUG("Request %zu - Continuing generation", i);
        }
    }
    
    CLLM_DEBUG("updateRequestStates completed");
}

}
