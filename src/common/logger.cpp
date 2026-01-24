#include "cllm/common/logger.h"
#include <memory>
#include <vector>

namespace cllm {

Logger::Logger() {
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
    
    logger_ = std::make_shared<spdlog::logger>("cllm", console_sink);
    logger_->set_level(spdlog::level::info);
    spdlog::register_logger(logger_);
}

Logger& Logger::instance() {
    static Logger instance;
    return instance;
}

void Logger::log_with_printf_style(spdlog::level::level_enum level, const char* fmt, va_list args) {
    // 首先计算需要的缓冲区大小
    va_list args_copy;
    va_copy(args_copy, args);
    int size = std::vsnprintf(nullptr, 0, fmt, args_copy);
    va_end(args_copy);
    
    if (size < 0) {
        logger_->log(level, "日志格式化错误");
        return;
    }
    
    // 分配缓冲区并格式化字符串
    std::vector<char> buffer(size + 1);
    std::vsnprintf(buffer.data(), buffer.size(), fmt, args);
    
    // 使用格式化后的字符串作为日志消息
    logger_->log(level, "{}", buffer.data());
}

void Logger::setLevel(spdlog::level::level_enum level) {
    logger_->set_level(level);
}

void Logger::addFileSink(const std::string& filename) {
    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(filename);
    file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] %v");
    logger_->sinks().push_back(file_sink);
}

void Logger::flush() {
    logger_->flush();
}

bool Logger::shouldLog(spdlog::level::level_enum level) const {
    return logger_ && logger_->should_log(level);
}

} // namespace cllm