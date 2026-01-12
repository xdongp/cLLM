#include "cllm/http/response_builder.h"
#include "cllm/common/config.h"

namespace cllm {

ResponseBuilder::ResponseBuilder() {
}

ResponseBuilder::~ResponseBuilder() {
}

HttpResponse ResponseBuilder::success(const nlohmann::json& data) {
    nlohmann::json responseJson = {
        {"success", true},
        {"data", data}
    };
    
    return json(responseJson, 200);
}

HttpResponse ResponseBuilder::success(const std::string& message, const nlohmann::json& data) {
    nlohmann::json responseJson = {
        {"success", true},
        {"message", message},
        {"data", data}
    };
    
    return json(responseJson, 200);
}

HttpResponse ResponseBuilder::error(int statusCode, const std::string& message) {
    nlohmann::json responseJson = {
        {"success", false},
        {"error", message}
    };
    
    return json(responseJson, statusCode);
}

HttpResponse ResponseBuilder::badRequest(const std::string& message) {
    return error(400, message);
}

HttpResponse ResponseBuilder::unauthorized(const std::string& message) {
    return error(401, message);
}

HttpResponse ResponseBuilder::forbidden(const std::string& message) {
    return error(403, message);
}

HttpResponse ResponseBuilder::notFound(const std::string& message) {
    return error(404, message);
}

HttpResponse ResponseBuilder::internalError(const std::string& message) {
    return error(500, message);
}

HttpResponse ResponseBuilder::serviceUnavailable(const std::string& message) {
    return error(503, message);
}

HttpResponse ResponseBuilder::json(const nlohmann::json& data, int statusCode) {
    HttpResponse response;
    response.setStatusCode(statusCode);
    response.setBody(data.dump());
    response.setContentType(getContentType());
    return response;
}

HttpResponse ResponseBuilder::text(const std::string& text, int statusCode) {
    HttpResponse response;
    response.setStatusCode(statusCode);
    response.setBody(text);
    response.setContentType("text/plain");
    return response;
}

HttpResponse ResponseBuilder::streaming(const std::string& data) {
    HttpResponse response;
    response.setStatusCode(200);
    response.setBody(data);
    response.setContentType("text/event-stream");
    return response;
}

std::string ResponseBuilder::getContentType() {
    return cllm::Config::instance().apiResponseContentTypeJson();
}

}
