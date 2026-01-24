#pragma once
#include <memory>
#include <string>
#include <vector>
#include <cstdarg>
#include <cstdio>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>

namespace cllm {

class Logger {
public:
    static Logger& instance();
    
    // printf风格的格式化支持
    void trace(const char* fmt, ...) {
        va_list args;
        va_start(args, fmt);
        log_with_printf_style(spdlog::level::trace, fmt, args);
        va_end(args);
    }
    
    void debug(const char* fmt, ...) {
        va_list args;
        va_start(args, fmt);
        log_with_printf_style(spdlog::level::debug, fmt, args);
        va_end(args);
    }
    
    void info(const char* fmt, ...) {
        va_list args;
        va_start(args, fmt);
        log_with_printf_style(spdlog::level::info, fmt, args);
        va_end(args);
    }
    
    void warn(const char* fmt, ...) {
        va_list args;
        va_start(args, fmt);
        log_with_printf_style(spdlog::level::warn, fmt, args);
        va_end(args);
    }
    
    void error(const char* fmt, ...) {
        va_list args;
        va_start(args, fmt);
        log_with_printf_style(spdlog::level::err, fmt, args);
        va_end(args);
    }
    
    void critical(const char* fmt, ...) {
        va_list args;
        va_start(args, fmt);
        log_with_printf_style(spdlog::level::critical, fmt, args);
        va_end(args);
    }
    
    void setLevel(spdlog::level::level_enum level);
    void addFileSink(const std::string& filename);
    void flush();
    bool shouldLog(spdlog::level::level_enum level) const;

private:
    Logger();
    void log_with_printf_style(spdlog::level::level_enum level, const char* fmt, va_list args);
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