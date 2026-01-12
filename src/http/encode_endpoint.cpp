#include "cllm/http/encode_endpoint.h"
#include "cllm/http/json_request_parser.h"
#include "cllm/http/response_builder.h"
#include "cllm/tokenizer/i_tokenizer.h"
#include "cllm/common/logger.h"
#include "cllm/common/config.h"
#include <nlohmann/json.hpp>
#include <sstream>

namespace cllm {

EncodeEndpoint::EncodeEndpoint(ITokenizer* tokenizer)
    : ApiEndpoint(cllm::Config::instance().apiEndpointEncodeName(), cllm::Config::instance().apiEndpointEncodePath(), cllm::Config::instance().apiEndpointEncodeMethod()),
      tokenizer_(tokenizer) {
}

EncodeEndpoint::~EncodeEndpoint() {
}

void EncodeEndpoint::setTokenizer(ITokenizer* tokenizer) {
    tokenizer_ = tokenizer;
}

EncodeEndpoint::EncodeRequest EncodeEndpoint::parseRequest(const HttpRequest& request) {
    EncodeRequest req;
    
    nlohmann::json jsonBody;
    
    if (!JsonRequestParser::validateJson(request.getBody(), jsonBody)) {
        CLLM_WARN("Failed to parse JSON request body: %s, using empty text", JsonRequestParser::getLastError().c_str());
    }
    
    JsonRequestParser::getFieldWithDefault(jsonBody, "text", req.text, std::string(""));
    
    return req;
}

HttpResponse EncodeEndpoint::handle(const HttpRequest& request) {
    try {
        EncodeRequest req = parseRequest(request);
        
        if (req.text.empty()) {
            return ResponseBuilder::badRequest("Missing required field: text");
        }
        
        if (tokenizer_ == nullptr) {
            return ResponseBuilder::internalError("Tokenizer is not initialized");
        }
        
        std::vector<int> tokens = tokenizer_->encode(req.text, true);
        
        nlohmann::json responseJson = {
            {"tokens", tokens},
            {"length", tokens.size()}
        };
        
        return ResponseBuilder::success(responseJson);
    } catch (const std::exception& e) {
        return ResponseBuilder::internalError(e.what());
    }
}

}
