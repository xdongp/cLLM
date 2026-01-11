#pragma once
#include <memory>
#include <string>
#include <vector>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>

namespace cllm {

class Logger {
public:
    static Logger& instance();
    
    template<typename... Args>
    void trace(const char* fmt, const Args&... args) {
        logger_->trace(fmt, args...);
    }
    
    template<typename... Args>
    void debug(const char* fmt, const Args&... args) {
        logger_->debug(fmt, args...);
    }
    
    template<typename... Args>
    void info(const char* fmt, const Args&... args) {
        logger_->info(fmt, args...);
    }
    
    template<typename... Args>
    void warn(const char* fmt, const Args&... args) {
        logger_->warn(fmt, args...);
    }
    
    template<typename... Args>
    void error(const char* fmt, const Args&... args) {
        logger_->error(fmt, args...);
    }
    
    template<typename... Args>
    void critical(const char* fmt, const Args&... args) {
        logger_->critical(fmt, args...);
    }
    
    void setLevel(spdlog::level::level_enum level);
    void addFileSink(const std::string& filename);
    void flush();

private:
    Logger();
    std::shared_ptr<spdlog::logger> logger_;
};

// 全局日志宏
#define CLLM_TRACE(...)    cllm::Logger::instance().trace(__VA_ARGS__)
#define CLLM_DEBUG(...)    cllm::Logger::instance().debug(__VA_ARGS__)
#define CLLM_INFO(...)     cllm::Logger::instance().info(__VA_ARGS__)
#define CLLM_WARN(...)     cllm::Logger::instance().warn(__VA_ARGS__)
#define CLLM_ERROR(...)    cllm::Logger::instance().error(__VA_ARGS__)
#define CLLM_CRITICAL(...) cllm::Logger::instance().critical(__VA_ARGS__)

} // namespace cllm