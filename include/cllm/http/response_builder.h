#pragma once

#include <string>
#include <map>
#include <nlohmann/json.hpp>
#include "cllm/http/response.h"

namespace cllm {

class ResponseBuilder {
public:
    ResponseBuilder();
    ~ResponseBuilder();
    
    static HttpResponse success(const nlohmann::json& data = {});
    static HttpResponse success(const std::string& message, const nlohmann::json& data = {});
    static HttpResponse error(int statusCode, const std::string& message);
    static HttpResponse badRequest(const std::string& message = "Bad request");
    static HttpResponse unauthorized(const std::string& message = "Unauthorized");
    static HttpResponse forbidden(const std::string& message = "Forbidden");
    static HttpResponse notFound(const std::string& message = "Not found");
    static HttpResponse internalError(const std::string& message = "Internal server error");
    static HttpResponse serviceUnavailable(const std::string& message = "Service unavailable");
    
    static HttpResponse json(const nlohmann::json& data, int statusCode = 200);
    static HttpResponse text(const std::string& text, int statusCode = 200);
    
    static HttpResponse streaming(const std::string& data);
    
private:
    static std::string getContentType();
};

}
