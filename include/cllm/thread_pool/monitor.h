/**
 * @file monitor.h
 * @brief 线程池监控器，监控线程池的运行状态
 * @author cLLM Team
 * @date 2024-01-01
 */

#pragma once

#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <cstddef>

namespace cllm {

/**
 * @brief 线程池统计信息结构
 * 
 * 记录线程池的运行状态和性能指标。
 */
struct ThreadPoolStats {
    size_t total_threads;       ///< 总线程数
    size_t running_threads;     ///< 运行中的线程数
    size_t idle_threads;        ///< 空闲线程数
    size_t queued_tasks;        ///< 队列中的任务数
    size_t total_tasks;         ///< 总任务数
    size_t completed_tasks;     ///< 已完成任务数
    double avg_task_time_ms;    ///< 平均任务执行时间（毫秒）
};

class ThreadPoolManager;

/**
 * @brief 线程池监控器类
 * 
 * 定期收集线程池的统计信息，提供性能监控功能。
 */
class ThreadPoolMonitor {
public:
    /**
     * @brief 构造函数
     * @param manager 线程池管理器引用
     */
    explicit ThreadPoolMonitor(ThreadPoolManager& manager);
    
    /**
     * @brief 析构函数，自动停止监控
     */
    ~ThreadPoolMonitor();
    
    ThreadPoolMonitor(const ThreadPoolMonitor&) = delete;
    ThreadPoolMonitor& operator=(const ThreadPoolMonitor&) = delete;
    
    /**
     * @brief 开始监控
     * @param interval_ms 监控间隔（毫秒）
     */
    void startMonitoring(size_t interval_ms);
    
    /**
     * @brief 停止监控
     */
    void stopMonitoring();
    
    /**
     * @brief 获取最新的统计信息
     * @return 线程池统计信息
     */
    ThreadPoolStats getStats() const;
    
private:
    /**
     * @brief 监控循环函数
     */
    void monitoringLoop();
    
    std::thread monitoring_thread_;  ///< 监控线程
    std::atomic<bool> stop_;         ///< 停止标志
    ThreadPoolManager& manager_;     ///< 线程池管理器引用
    
    mutable std::mutex stats_mutex_; ///< 统计信息互斥锁
    ThreadPoolStats stats_;          ///< 统计信息
};

}  // namespace cllm
