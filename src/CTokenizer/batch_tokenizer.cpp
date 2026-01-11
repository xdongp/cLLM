#include "cllm/CTokenizer/batch_tokenizer.h"
#include "cllm/CTokenizer/performance_config.h"
#include <algorithm>
#include <stdexcept>

namespace cllm {

int BatchTokenizer::getOptimalThreadCount(int maxParallel, size_t taskCount) {
    // 如果指定了最大并行数
    if (maxParallel > 0) {
        return std::min(maxParallel, static_cast<int>(taskCount));
    }
    
    // 自动检测：使用 CPU 核心数
    int hwThreads = static_cast<int>(std::thread::hardware_concurrency());
    if (hwThreads <= 0) {
        hwThreads = 4;  // 默认值
    }
    
    // 不要创建超过任务数量的线程
    return std::min(hwThreads, static_cast<int>(taskCount));
}

BatchTokenizer::BatchEncodeResult BatchTokenizer::batchEncode(
    CTokenizer* tokenizer,
    const std::vector<std::string>& texts,
    bool addSpecialTokens,
    int maxParallel
) {
    BatchEncodeResult result;
    
    if (!tokenizer) {
        throw std::invalid_argument("Tokenizer cannot be null");
    }
    
    if (texts.empty()) {
        return result;
    }
    
    const size_t numTexts = texts.size();
    result.tokenized.resize(numTexts);
    result.success.resize(numTexts, false);
    result.errors.resize(numTexts);
    
    // 如果只有一个文本或单线程，直接处理
    if (numTexts == 1 || maxParallel == 1) {
        for (size_t i = 0; i < numTexts; ++i) {
            try {
                result.tokenized[i] = tokenizer->encode(texts[i], addSpecialTokens);
                result.success[i] = true;
            } catch (const std::exception& e) {
                result.success[i] = false;
                result.errors[i] = e.what();
            }
        }
        return result;
    }
    
    // 多线程并行处理
    const int numThreads = getOptimalThreadCount(maxParallel, numTexts);
    std::vector<std::future<void>> futures;
    futures.reserve(numThreads);
    
    // 计算每个线程处理的任务范围
    const size_t tasksPerThread = (numTexts + numThreads - 1) / numThreads;
    
    for (int t = 0; t < numThreads; ++t) {
        const size_t startIdx = t * tasksPerThread;
        const size_t endIdx = std::min(startIdx + tasksPerThread, numTexts);
        
        if (startIdx >= numTexts) {
            break;
        }
        
        futures.push_back(std::async(std::launch::async, [&, startIdx, endIdx]() {
            for (size_t i = startIdx; i < endIdx; ++i) {
                try {
                    result.tokenized[i] = tokenizer->encode(texts[i], addSpecialTokens);
                    result.success[i] = true;
                } catch (const std::exception& e) {
                    result.success[i] = false;
                    result.errors[i] = e.what();
                } catch (...) {
                    result.success[i] = false;
                    result.errors[i] = "Unknown error during encoding";
                }
            }
        }));
    }
    
    // 等待所有任务完成
    for (auto& future : futures) {
        future.get();
    }
    
    return result;
}

BatchTokenizer::BatchDecodeResult BatchTokenizer::batchDecode(
    CTokenizer* tokenizer,
    const std::vector<std::vector<llama_token>>& tokenSequences,
    bool skipSpecialTokens,
    int maxParallel
) {
    BatchDecodeResult result;
    
    if (!tokenizer) {
        throw std::invalid_argument("Tokenizer cannot be null");
    }
    
    if (tokenSequences.empty()) {
        return result;
    }
    
    const size_t numSequences = tokenSequences.size();
    result.decoded.resize(numSequences);
    result.success.resize(numSequences, false);
    result.errors.resize(numSequences);
    
    // 如果只有一个序列或单线程，直接处理
    if (numSequences == 1 || maxParallel == 1) {
        for (size_t i = 0; i < numSequences; ++i) {
            try {
                result.decoded[i] = tokenizer->decode(tokenSequences[i], skipSpecialTokens);
                result.success[i] = true;
            } catch (const std::exception& e) {
                result.success[i] = false;
                result.errors[i] = e.what();
            }
        }
        return result;
    }
    
    // 多线程并行处理
    const int numThreads = getOptimalThreadCount(maxParallel, numSequences);
    std::vector<std::future<void>> futures;
    futures.reserve(numThreads);
    
    // 计算每个线程处理的任务范围
    const size_t tasksPerThread = (numSequences + numThreads - 1) / numThreads;
    
    for (int t = 0; t < numThreads; ++t) {
        const size_t startIdx = t * tasksPerThread;
        const size_t endIdx = std::min(startIdx + tasksPerThread, numSequences);
        
        if (startIdx >= numSequences) {
            break;
        }
        
        futures.push_back(std::async(std::launch::async, [&, startIdx, endIdx]() {
            for (size_t i = startIdx; i < endIdx; ++i) {
                try {
                    result.decoded[i] = tokenizer->decode(tokenSequences[i], skipSpecialTokens);
                    result.success[i] = true;
                } catch (const std::exception& e) {
                    result.success[i] = false;
                    result.errors[i] = e.what();
                } catch (...) {
                    result.success[i] = false;
                    result.errors[i] = "Unknown error during decoding";
                }
            }
        }));
    }
    
    // 等待所有任务完成
    for (auto& future : futures) {
        future.get();
    }
    
    return result;
}

// 带配置的批处理编码
BatchTokenizer::BatchEncodeResult BatchTokenizer::batchEncode(
    CTokenizer* tokenizer,
    const std::vector<std::string>& texts,
    const TokenizerPerformanceConfig& config,
    bool addSpecialTokens
) {
    int maxParallel = config.numThreads == 0 ? 
        -1 : static_cast<int>(config.numThreads);
    
    return batchEncode(tokenizer, texts, addSpecialTokens, maxParallel);
}

// 带配置的批处理解码
BatchTokenizer::BatchDecodeResult BatchTokenizer::batchDecode(
    CTokenizer* tokenizer,
    const std::vector<std::vector<llama_token>>& tokensList,
    const TokenizerPerformanceConfig& config,
    bool skipSpecialTokens
) {
    int maxParallel = config.numThreads == 0 ? 
        -1 : static_cast<int>(config.numThreads);
    
    return batchDecode(tokenizer, tokensList, skipSpecialTokens, maxParallel);
}

} // namespace cllm
