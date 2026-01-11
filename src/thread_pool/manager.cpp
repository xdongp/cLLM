/**
 * @file manager.cpp
 * @brief 线程池管理器实现
 * @author cLLM Team
 * @date 2024-01-01
 */

#include "cllm/thread_pool/manager.h"

namespace cllm {

ThreadPoolManager::ThreadPoolManager(size_t num_threads)
    : pool_(num_threads), paused_(false), total_submitted_tasks_(0) {
}

ThreadPoolManager::~ThreadPoolManager() {
    shutdown();
}

void ThreadPoolManager::submitTask(std::function<void()> task) {
    try {
        pool_.detach_task(task);
        total_submitted_tasks_.fetch_add(1);
    } catch (const std::exception& e) {
        throw ThreadPoolException(std::string("Failed to submit task: ") + e.what());
    }
}

void ThreadPoolManager::waitForAll() {
    pool_.wait();
}

void ThreadPoolManager::shutdown() {
    pool_.wait();
}

size_t ThreadPoolManager::getThreadCount() const {
    return pool_.get_thread_count();
}

size_t ThreadPoolManager::getTasksTotal() const {
    return pool_.get_tasks_total();
}

size_t ThreadPoolManager::getTasksRunning() const {
    return pool_.get_tasks_running();
}

size_t ThreadPoolManager::getTasksQueued() const {
    return pool_.get_tasks_queued();
}

void ThreadPoolManager::pause() {
    pool_.pause();
    paused_ = true;
}

void ThreadPoolManager::resume() {
    pool_.unpause();
    paused_ = false;
}

bool ThreadPoolManager::isPaused() const {
    return paused_;
}

size_t ThreadPoolManager::getAllSubmittedTasks() const {
    return total_submitted_tasks_.load();
}

}  // namespace cllm
