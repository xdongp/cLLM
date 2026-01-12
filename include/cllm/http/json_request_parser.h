#pragma once

#include <string>
#include <functional>
#include <nlohmann/json.hpp>

namespace cllm {

class JsonRequestParser {
public:
    template<typename T>
    static bool getField(const nlohmann::json& json, const std::string& key, T& value, bool required = false);
    
    template<typename T>
    static bool getFieldWithDefault(const nlohmann::json& json, const std::string& key, T& value, const T& defaultValue);
    
    static bool validateJson(const std::string& body, nlohmann::json& json);
    
    static bool validateRequiredFields(const nlohmann::json& json, const std::vector<std::string>& requiredFields);
    
    static std::string getLastError();
    
private:
    static thread_local std::string lastError_;
};

template<typename T>
bool JsonRequestParser::getField(const nlohmann::json& json, const std::string& key, T& value, bool required) {
    if (!json.contains(key)) {
        if (required) {
            lastError_ = "Required field '" + key + "' is missing";
        }
        return !required;
    }
    
    try {
        value = json[key].get<T>();
        return true;
    } catch (const nlohmann::json::exception& e) {
        lastError_ = std::string("Failed to parse field '") + key + "': " + e.what();
        return false;
    }
}

template<typename T>
bool JsonRequestParser::getFieldWithDefault(const nlohmann::json& json, const std::string& key, T& value, const T& defaultValue) {
    if (!json.contains(key)) {
        value = defaultValue;
        return true;
    }
    
    try {
        value = json[key].get<T>();
        return true;
    } catch (const nlohmann::json::exception& e) {
        lastError_ = std::string("Failed to parse field '") + key + "': " + e.what();
        value = defaultValue;
        return false;
    }
}

}
