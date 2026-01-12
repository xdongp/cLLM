#include "cllm/http/response_builder.h"
#include "cllm/common/config.h"
#include <nlohmann/json.hpp>

namespace cllm {

ResponseBuilder::ResponseBuilder() : statusCode_(200) {
}

ResponseBuilder::~ResponseBuilder() {
}

ResponseBuilder& ResponseBuilder::setStatus(int code) {
    statusCode_ = code;
    return *this;
}

ResponseBuilder& ResponseBuilder::setHeader(const std::string& name, const std::string& value) {
    headers_[name] = value;
    return *this;
}

ResponseBuilder& ResponseBuilder::setBody(const std::string& body) {
    body_ = body;
    return *this;
}

HttpResponse ResponseBuilder::build() {
    HttpResponse response;
    response.setStatusCode(statusCode_);
    
    for (const auto& header : headers_) {
        response.setHeader(header.first, header.second);
    }
    
    response.setBody(body_);
    return response;
}

ResponseBuilder ResponseBuilder::ok() {
    ResponseBuilder builder;
    builder.setStatus(200);
    return builder;
}

ResponseBuilder ResponseBuilder::error(int code, const std::string& message) {
    ResponseBuilder builder;
    builder.setStatus(code);
    
    nlohmann::json errorJson;
    errorJson["error"]["code"] = code;
    errorJson["error"]["message"] = message;
    builder.setBody(errorJson.dump());
    builder.setHeader("Content-Type", cllm::Config::instance().apiResponseContentTypeJson());
    
    return builder;
}

}
