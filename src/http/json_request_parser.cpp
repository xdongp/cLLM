#include "cllm/http/json_request_parser.h"

namespace cllm {

thread_local std::string JsonRequestParser::lastError_;

bool JsonRequestParser::validateJson(const std::string& body, nlohmann::json& json) {
    try {
        json = nlohmann::json::parse(body);
        return true;
    } catch (const nlohmann::json::parse_error& e) {
        lastError_ = std::string("JSON parse error: ") + e.what();
        return false;
    } catch (const nlohmann::json::exception& e) {
        lastError_ = std::string("JSON error: ") + e.what();
        return false;
    }
}

bool JsonRequestParser::validateRequiredFields(const nlohmann::json& json, const std::vector<std::string>& requiredFields) {
    for (const auto& field : requiredFields) {
        if (!json.contains(field)) {
            lastError_ = "Required field '" + field + "' is missing";
            return false;
        }
    }
    return true;
}

std::string JsonRequestParser::getLastError() {
    return lastError_;
}

}
