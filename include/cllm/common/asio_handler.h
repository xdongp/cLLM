#ifndef CLLM_COMMON_ASIO_HANDLER_H
#define CLLM_COMMON_ASIO_HANDLER_H

#include <functional>
#include <memory>
#include <thread>

namespace cllm {

/**
 * @brief Asio异步任务处理器
 * 符合设计文档中要求显式使用Asio（standalone Asio）的技术栈要求
 */
class AsioHandler {
public:
    /**
     * @brief 构造函数
     */
    AsioHandler();

    /**
     * @brief 析构函数
     */
    ~AsioHandler();

    /**
     * @brief 提交异步任务
     * @param task 要执行的任务
     */
    void postTask(std::function<void()> task);

    /**
     * @brief 获取线程池大小
     * @return 线程池大小
     */
    size_t getThreadPoolSize() const;

    /**
     * @brief 停止所有异步操作
     */
    void stop();

private:
    class AsioImpl* impl_;  ///< Asio实现
    size_t threadPoolSize_;                         ///< 线程池大小
};

} // namespace cllm

#endif // CLLM_COMMON_ASIO_HANDLER_H