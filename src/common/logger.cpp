#include "cllm/common/logger.h"
#include <memory>

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

} // namespace cllm