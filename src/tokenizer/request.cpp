#include "cllm/tokenizer/request.h"

namespace cllm {

GenerationRequest::GenerationRequest()
    : stream_(false) {
}

GenerationRequest::~GenerationRequest() {
}

void GenerationRequest::setRequestId(const std::string& requestId) {
    requestId_ = requestId;
}

void GenerationRequest::setPrompt(const std::string& prompt) {
    prompt_ = prompt;
}

void GenerationRequest::setConfig(const TokenizerConfig& config) {
    config_ = config;
}

void GenerationRequest::setStream(bool stream) {
    stream_ = stream;
}

std::string GenerationRequest::getRequestId() const {
    return requestId_;
}

std::string GenerationRequest::getPrompt() const {
    return prompt_;
}

TokenizerConfig GenerationRequest::getConfig() const {
    return config_;
}

bool GenerationRequest::isStream() const {
    return stream_;
}

std::vector<int> GenerationRequest::getEncodedPrompt() const {
    return encodedPrompt_;
}

void GenerationRequest::setEncodedPrompt(const std::vector<int>& encodedPrompt) {
    encodedPrompt_ = encodedPrompt;
}

}
