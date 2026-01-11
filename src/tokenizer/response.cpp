#include "cllm/tokenizer/response.h"

namespace cllm {

GenerationResponse::GenerationResponse()
    : finished_(false), responseTime_(0.0f) {
}

GenerationResponse::~GenerationResponse() {
}

void GenerationResponse::setRequestId(const std::string& requestId) {
    requestId_ = requestId;
}

void GenerationResponse::setText(const std::string& text) {
    text_ = text;
}

void GenerationResponse::setTokens(const std::vector<int>& tokens) {
    tokens_ = tokens;
}

void GenerationResponse::setFinished(bool finished) {
    finished_ = finished;
}

void GenerationResponse::setError(const std::string& error) {
    error_ = error;
}

void GenerationResponse::setResponseTime(float responseTime) {
    responseTime_ = responseTime;
}

std::string GenerationResponse::getRequestId() const {
    return requestId_;
}

std::string GenerationResponse::getText() const {
    return text_;
}

std::vector<int> GenerationResponse::getTokens() const {
    return tokens_;
}

bool GenerationResponse::isFinished() const {
    return finished_;
}

std::string GenerationResponse::getError() const {
    return error_;
}

float GenerationResponse::getResponseTime() const {
    return responseTime_;
}

}
