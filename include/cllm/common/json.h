/**
 * @file json.h
 * @brief 简单的JSON解析器
 * @author cLLM Team
 * @date 2026-01-09
 */

#ifndef CLLM_COMMON_JSON_H
#define CLLM_COMMON_JSON_H

#include <string>
#include <nlohmann/json.hpp>

namespace cllm {
    using JsonValue = nlohmann::json;
    
    /**
     * @brief 简单的JSON解析器
     */
    class JsonParser {
    public:
        /**
         * @brief 解析JSON字符串
         * @param jsonStr 要解析的JSON字符串
         * @return 解析后的JSON对象
         */
        static JsonValue parse(const std::string& jsonStr);
        
        /**
         * @brief 将JSON对象转换为字符串
         * @param jsonObj JSON对象
         * @return JSON字符串
         */
        static std::string stringify(const JsonValue& jsonObj);
    };
}

#endif // CLLM_COMMON_JSON_H
