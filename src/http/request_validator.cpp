#include "cllm/http/request_validator.h"
#include <algorithm>

namespace cllm {

RequestValidator::RequestValidator() {
}

RequestValidator::~RequestValidator() {
}

bool RequestValidator::validateRequired(const HttpRequest& request, const std::string& field) {
    if (!request.hasHeader(field) && !request.hasQuery(field)) {
        setError("Required field '" + field + "' is missing");
        return false;
    }
    return true;
}

bool RequestValidator::validateRequired(const nlohmann::json& jsonBody, const std::string& field) {
    if (!jsonBody.contains(field)) {
        setError("Required field '" + field + "' is missing");
        return false;
    }
    return true;
}

bool RequestValidator::validateType(const std::string& value, const std::string& expectedType) {
    if (expectedType == "int") {
        try {
            std::stoi(value);
            return true;
        } catch (...) {
            setError("Invalid integer value: " + value);
            return false;
        }
    } else if (expectedType == "float") {
        try {
            std::stof(value);
            return true;
        } catch (...) {
            setError("Invalid float value: " + value);
            return false;
        }
    } else if (expectedType == "string") {
        return true;
    } else if (expectedType == "bool") {
        if (value == "true" || value == "false") {
            return true;
        }
        setError("Invalid boolean value: " + value);
        return false;
    }
    
    setError("Unknown type: " + expectedType);
    return false;
}

bool RequestValidator::validateType(const nlohmann::json& jsonBody, const std::string& field, const std::string& expectedType) {
    if (!jsonBody.contains(field)) {
        setError("Field '" + field + "' is missing");
        return false;
    }
    
    const auto& value = jsonBody[field];
    
    if (expectedType == "string") {
        if (!value.is_string()) {
            setError("Field '" + field + "' should be a string");
            return false;
        }
    } else if (expectedType == "number") {
        if (!value.is_number()) {
            setError("Field '" + field + "' should be a number");
            return false;
        }
    } else if (expectedType == "integer") {
        if (!value.is_number_integer()) {
            setError("Field '" + field + "' should be an integer");
            return false;
        }
    } else if (expectedType == "boolean") {
        if (!value.is_boolean()) {
            setError("Field '" + field + "' should be a boolean");
            return false;
        }
    } else if (expectedType == "array") {
        if (!value.is_array()) {
            setError("Field '" + field + "' should be an array");
            return false;
        }
    } else if (expectedType == "object") {
        if (!value.is_object()) {
            setError("Field '" + field + "' should be an object");
            return false;
        }
    } else {
        setError("Unknown type: " + expectedType);
        return false;
    }
    
    return true;
}

bool RequestValidator::validateRange(int value, int min, int max) {
    if (value < min || value > max) {
        setError("Value " + std::to_string(value) + " is out of range [" + 
                     std::to_string(min) + ", " + std::to_string(max) + "]");
        return false;
    }
    return true;
}

bool RequestValidator::validateRange(float value, float min, float max) {
    if (value < min || value > max) {
        setError("Value " + std::to_string(value) + " is out of range [" + 
                     std::to_string(min) + ", " + std::to_string(max) + "]");
        return false;
    }
    return true;
}

bool RequestValidator::validateLength(const std::string& value, size_t minLength, size_t maxLength) {
    size_t length = value.length();
    if (length < minLength || length > maxLength) {
        setError("String length " + std::to_string(length) + " is out of range [" + 
                     std::to_string(minLength) + ", " + std::to_string(maxLength) + "]");
        return false;
    }
    return true;
}

bool RequestValidator::validateSize(const nlohmann::json& array, size_t minSize, size_t maxSize) {
    if (!array.is_array()) {
        setError("Value is not an array");
        return false;
    }
    
    size_t size = array.size();
    if (size < minSize || size > maxSize) {
        setError("Array size " + std::to_string(size) + " is out of range [" + 
                     std::to_string(minSize) + ", " + std::to_string(maxSize) + "]");
        return false;
    }
    return true;
}

bool RequestValidator::validatePattern(const std::string& value, const std::string& pattern) {
    try {
        std::regex regex(pattern);
        if (!std::regex_match(value, regex)) {
            setError("Value '" + value + "' does not match pattern '" + pattern + "'");
            return false;
        }
        return true;
    } catch (const std::regex_error& e) {
        setError("Invalid regex pattern: " + pattern);
        return false;
    }
}

bool RequestValidator::validateEnum(const std::string& value, const std::vector<std::string>& allowedValues) {
    auto it = std::find(allowedValues.begin(), allowedValues.end(), value);
    if (it == allowedValues.end()) {
        setError("Value '" + value + "' is not in allowed values");
        return false;
    }
    return true;
}

bool RequestValidator::validateCustom(const std::string& value, std::function<bool(const std::string&)> validator) {
    if (!validator(value)) {
        setError("Custom validation failed for value: " + value);
        return false;
    }
    return true;
}

bool RequestValidator::validateJson(const std::string& requestBody, nlohmann::json& jsonBody) {
    try {
        jsonBody = nlohmann::json::parse(requestBody);
        return true;
    } catch (const nlohmann::json::parse_error& e) {
        setError("JSON parse error: " + std::string(e.what()));
        return false;
    } catch (const nlohmann::json::exception& e) {
        setError("JSON error: " + std::string(e.what()));
        return false;
    }
}

bool RequestValidator::validateRequiredFields(const nlohmann::json& jsonBody, const std::vector<std::string>& fields) {
    for (const auto& field : fields) {
        if (!jsonBody.contains(field)) {
            setError("Required field '" + field + "' is missing");
            return false;
        }
    }
    return true;
}

std::string RequestValidator::getLastError() const {
    return lastError_;
}

void RequestValidator::clearLastError() {
    lastError_.clear();
}

void RequestValidator::setError(const std::string& error) {
    lastError_ = error;
}

}
