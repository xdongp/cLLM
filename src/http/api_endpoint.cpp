#include "cllm/http/api_endpoint.h"

namespace cllm {

ApiEndpoint::ApiEndpoint(
    const std::string& name,
    const std::string& path,
    const std::string& method
) : name_(name), path_(path), method_(method) {
}

ApiEndpoint::~ApiEndpoint() {
}

std::string ApiEndpoint::getName() const {
    return name_;
}

std::string ApiEndpoint::getPath() const {
    return path_;
}

std::string ApiEndpoint::getMethod() const {
    return method_;
}

void ApiEndpoint::setHandler(HandlerFunc handler) {
    handler_ = handler;
}

ApiEndpoint::HandlerFunc ApiEndpoint::getHandler() const {
    return handler_;
}

void ApiEndpoint::validateRequest(const HttpRequest& request) {
}

}
