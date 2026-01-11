/**
 * @file config.h
 * @brief 调度器配置结构
 * @author cLLM Team
 * @date 2024-01-09
 */

#ifndef CLLM_SCHEDULER_CONFIG_H
#define CLLM_SCHEDULER_CONFIG_H

#include <cstddef>

namespace cllm {

/**
 * @brief 调度器配置结构
 * 
 * 包含调度器运行所需的各种参数配置。
 */
struct SchedulerConfig {
    size_t maxBatchSize = 8;               ///< 最大批处理大小
    size_t maxContextLength = 2048;        ///< 最大上下文长度
    float contextUsageThreshold = 0.75f;   ///< 上下文使用阈值
    float defaultTemperature = 0.7f;       ///< 默认温度参数
    float defaultTopP = 0.9f;              ///< 默认Top-P参数
    int defaultTopK = 50;                  ///< 默认Top-K参数
    size_t defaultMaxTokens = 100;         ///< 默认最大token数
    float requestTimeout = 300.0f;         ///< 请求超时时间（秒）
    
    size_t schedulerLoopInterval = 100;    ///< 调度器循环间隔（微秒）
    size_t idleLoopInterval = 10000;       ///< 空闲循环间隔（微秒）
};

}

#endif
