/**
 * @file manager.h
 * @brief 线程池管理器，封装BS::thread_pool提供简化接口
 * @author cLLM Team
 * @date 2024-01-01
 */

#pragma once

#include <functional>
#include <future>
#include <cstddef>
#include <stdexcept>
#include <BS_thread_pool.hpp>
// 包含新的通用异常头文件
#include "cllm/common/exceptions.h"

namespace cllm {

// ThreadPoolException 已在 cllm/common/exceptions.h 中定义，保留别名
using ThreadPoolException = cllm::ThreadPoolException;

/**
 * @brief 线程池管理器类
 * 
 * 封装BS::thread_pool，提供简化的任务提交接口。
 * 支持任务提交、线程池暂停/恢复、统计信息查询等功能。
 */
class ThreadPoolManager {
public:
    /**
     * @brief 构造函数
     * @param num_threads 线程数量
     */
    explicit ThreadPoolManager(size_t num_threads);
    
    /**
     * @brief 析构函数，自动调用shutdown()
     */
    ~ThreadPoolManager();
    
    ThreadPoolManager(const ThreadPoolManager&) = delete;
    ThreadPoolManager& operator=(const ThreadPoolManager&) = delete;
    
    /**
     * @brief 提交无返回值的任务
     * @param task 任务函数
     * @throws ThreadPoolException 如果线程池已暂停或提交失败
     */
    void submitTask(std::function<void()> task);
    
    /**
     * @brief 提交有返回值的任务
     * @tparam F 函数类型
     * @tparam Args 参数类型
     * @param f 任务函数
     * @param args 任务参数
     * @return future对象，用于获取任务结果
     */
    template<typename F, typename... Args>
    auto submitTaskWithResult(F&& f, Args&&... args) 
        -> std::future<decltype(f(args...))>;
    
    /**
     * @brief 等待所有任务完成
     */
    void waitForAll();
    
    /**
     * @brief 关闭线程池，等待所有任务完成
     */
    void shutdown();
    
    /**
     * @brief 获取线程池的总线程数
     * @return 线程数
     */
    size_t getThreadCount() const;
    
    /**
     * @brief 获取总任务数
     * @return 总任务数（包括已完成、运行中和队列中的任务）
     */
    size_t getTasksTotal() const;
    
    /**
     * @brief 获取当前运行中的任务数
     * @return 运行中的任务数
     */
    size_t getTasksRunning() const;
    
    /**
     * @brief 获取当前队列中的任务数
     * @return 队列中的任务数
     */
    size_t getTasksQueued() const;
    
    /**
     * @brief 暂停线程池，不再执行新任务
     */
    void pause();
    
    /**
     * @brief 恢复线程池，继续执行任务
     */
    void resume();
    
    /**
     * @brief 判断线程池是否处于暂停状态
     * @return true 如果已暂停，false 否则
     */
    bool isPaused() const;
    
    /**
     * @brief 获取所有提交的任务总数（包括已完成、运行中和队列中的任务）
     * @return 总任务数
     */
    size_t getAllSubmittedTasks() const;
    
private:
    BS::thread_pool<BS::tp::pause> pool_;  ///< 底层线程池
    bool paused_;  ///< 暂停状态标志
    std::atomic<size_t> total_submitted_tasks_;  ///< 所有提交的任务总数
};

template<typename F, typename... Args>
auto ThreadPoolManager::submitTaskWithResult(F&& f, Args&&... args) 
    -> std::future<decltype(f(args...))> {
    return pool_.submit_task(std::forward<F>(f), std::forward<Args>(args)...);
}

}  // namespace cllm
