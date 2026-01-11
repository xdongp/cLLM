#include "cllm/http/request.h"

namespace cllm {

HttpRequest::HttpRequest() {
}

HttpRequest::~HttpRequest() {
}

std::string HttpRequest::getMethod() const {
    return method_;
}

std::string HttpRequest::getPath() const {
    return path_;
}

std::string HttpRequest::getHeader(const std::string& name) const {
    auto it = headers_.find(name);
    if (it != headers_.end()) {
        return it->second;
    }
    return "";
}

std::string HttpRequest::getBody() const {
    return body_;
}

std::string HttpRequest::getQuery(const std::string& key) const {
    auto it = queries_.find(key);
    if (it != queries_.end()) {
        return it->second;
    }
    return "";
}

void HttpRequest::setMethod(const std::string& method) {
    method_ = method;
}

void HttpRequest::setPath(const std::string& path) {
    path_ = path;
}

void HttpRequest::setHeader(const std::string& name, const std::string& value) {
    headers_[name] = value;
}

void HttpRequest::setBody(const std::string& body) {
    body_ = body;
}

void HttpRequest::setQuery(const std::string& key, const std::string& value) {
    queries_[key] = value;
}

bool HttpRequest::hasHeader(const std::string& name) const {
    return headers_.find(name) != headers_.end();
}

bool HttpRequest::hasQuery(const std::string& key) const {
    return queries_.find(key) != queries_.end();
}

std::map<std::string, std::string> HttpRequest::getAllHeaders() const {
    return headers_;
}

std::map<std::string, std::string> HttpRequest::getAllQueries() const {
    return queries_;
}

}
