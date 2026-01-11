#include "cllm/http/request_validator.h"

namespace cllm {

RequestValidator::RequestValidator() {
}

RequestValidator::~RequestValidator() {
}

bool RequestValidator::validateRequired(const HttpRequest& request, const std::string& field) {
    if (!request.hasHeader(field) && !request.hasQuery(field)) {
        lastError_ = "Required field '" + field + "' is missing";
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
            lastError_ = "Invalid integer value: " + value;
            return false;
        }
    } else if (expectedType == "float") {
        try {
            std::stof(value);
            return true;
        } catch (...) {
            lastError_ = "Invalid float value: " + value;
            return false;
        }
    } else if (expectedType == "string") {
        return true;
    } else if (expectedType == "bool") {
        if (value == "true" || value == "false") {
            return true;
        }
        lastError_ = "Invalid boolean value: " + value;
        return false;
    }
    
    lastError_ = "Unknown type: " + expectedType;
    return false;
}

bool RequestValidator::validateRange(int value, int min, int max) {
    if (value < min || value > max) {
        lastError_ = "Value " + std::to_string(value) + " is out of range [" + 
                     std::to_string(min) + ", " + std::to_string(max) + "]";
        return false;
    }
    return true;
}

bool RequestValidator::validateRange(float value, float min, float max) {
    if (value < min || value > max) {
        lastError_ = "Value " + std::to_string(value) + " is out of range [" + 
                     std::to_string(min) + ", " + std::to_string(max) + "]";
        return false;
    }
    return true;
}

std::string RequestValidator::getLastError() const {
    return lastError_;
}

}
