/**
 * @file json.cpp
 * @brief 简单的JSON解析器实现（使用nlohmann/json）
 * @author cLLM Team
 * @date 2026-01-09
 */

#include "cllm/common/json.h"
#include <nlohmann/json.hpp>
#include "cllm/common/logger.h"
#include <stdexcept>
#include <sstream>

namespace cllm {

JsonValue JsonParser::parse(const std::string& jsonStr) {
    try {
        nlohmann::json parsedJson = nlohmann::json::parse(jsonStr);
        return parsedJson;
    } catch (const std::exception& e) {
        CLLM_ERROR("Failed to parse JSON: %s", e.what());
        throw std::runtime_error("JSON parsing failed: " + std::string(e.what()));
    }
}

std::string JsonParser::stringify(const JsonValue& jsonObj) {
    try {
        return jsonObj.dump();
    } catch (const std::exception& e) {
        CLLM_ERROR("Failed to stringify JSON: %s", e.what());
        throw std::runtime_error("JSON stringify failed: " + std::string(e.what()));
    }
}

} // namespace cllm
