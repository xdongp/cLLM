/**
 * @file monitor.cpp
 * @brief 线程池监控器实现
 * @author cLLM Team
 * @date 2024-01-01
 */

#include "cllm/thread_pool/monitor.h"
#include "cllm/thread_pool/manager.h"
#include <thread>

namespace cllm {

ThreadPoolMonitor::ThreadPoolMonitor(ThreadPoolManager& manager)
    : stop_(false), manager_(manager) {
    stats_ = ThreadPoolStats{};
}

ThreadPoolMonitor::~ThreadPoolMonitor() {
    stopMonitoring();
}

void ThreadPoolMonitor::startMonitoring(size_t interval_ms) {
    if (monitoring_thread_.joinable()) {
        return;
    }
    
    stop_ = false;
    monitoring_thread_ = std::thread([this, interval_ms]() {
        while (!stop_) {
            monitoringLoop();
            std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
        }
    });
}

void ThreadPoolMonitor::stopMonitoring() {
    stop_ = true;
    if (monitoring_thread_.joinable()) {
        monitoring_thread_.join();
    }
}

ThreadPoolStats ThreadPoolMonitor::getStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void ThreadPoolMonitor::monitoringLoop() {
    ThreadPoolStats new_stats;
    
    new_stats.total_threads = manager_.getThreadCount();
    new_stats.running_threads = manager_.getTasksRunning();
    new_stats.idle_threads = new_stats.total_threads - new_stats.running_threads;
    new_stats.queued_tasks = manager_.getTasksQueued();
    new_stats.total_tasks = manager_.getTasksTotal();
    new_stats.completed_tasks = manager_.getAllSubmittedTasks() - new_stats.running_threads - new_stats.queued_tasks;
    new_stats.avg_task_time_ms = 0.0;
    
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_ = new_stats;
    }
}

}  // namespace cllm
