#include "cllm/http/response.h"
#include <nlohmann/json.hpp>
#include <sstream>

namespace cllm {

HttpResponse::HttpResponse() : statusCode_(200), streaming_(false) {
    setContentType("text/plain");
}

HttpResponse::~HttpResponse() {
}

void HttpResponse::setStatusCode(int code) {
    statusCode_ = code;
}

void HttpResponse::setHeader(const std::string& name, const std::string& value) {
    headers_[name] = value;
}

void HttpResponse::setBody(const std::string& body) {
    body_ = body;
}

void HttpResponse::setContentType(const std::string& contentType) {
    setHeader("Content-Type", contentType);
}

int HttpResponse::getStatusCode() const {
    return statusCode_;
}

std::string HttpResponse::getHeader(const std::string& name) const {
    auto it = headers_.find(name);
    if (it != headers_.end()) {
        return it->second;
    }
    return "";
}

std::string HttpResponse::getBody() const {
    return body_;
}

std::string HttpResponse::getContentType() const {
    return getHeader("Content-Type");
}

std::map<std::string, std::string> HttpResponse::getAllHeaders() const {
    return headers_;
}

void HttpResponse::setError(int code, const std::string& message) {
    setStatusCode(code);
    
    nlohmann::json errorJson;
    errorJson["error"]["code"] = code;
    errorJson["error"]["message"] = message;
    setBody(errorJson.dump());
    setContentType("application/json");
}

HttpResponse HttpResponse::ok(const std::string& body) {
    HttpResponse response;
    response.setStatusCode(200);
    response.setBody(body);
    return response;
}

HttpResponse HttpResponse::notFound() {
    HttpResponse response;
    response.setError(404, "Not Found");
    return response;
}

HttpResponse HttpResponse::badRequest(const std::string& message) {
    HttpResponse response;
    response.setError(400, message);
    return response;
}

HttpResponse HttpResponse::internalError(const std::string& message) {
    HttpResponse response;
    response.setError(500, message);
    return response;
}

// Streaming support methods
void HttpResponse::enableStreaming() {
    streaming_ = true;
    setContentType("text/event-stream");
    setHeader("Cache-Control", "no-cache");
    setHeader("Connection", "keep-alive");
}

bool HttpResponse::isStreaming() const {
    return streaming_;
}

void HttpResponse::addChunk(const std::string& chunk) {
    chunks_.push_back(chunk);
}

std::vector<std::string> HttpResponse::getChunks() const {
    return chunks_;
}

}
