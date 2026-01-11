#include "cllm/http/encode_endpoint.h"
#include "cllm/tokenizer/i_tokenizer.h"
#include "cllm/common/logger.h"
#include <nlohmann/json.hpp>
#include <sstream>

namespace cllm {

EncodeEndpoint::EncodeEndpoint(ITokenizer* tokenizer)
    : ApiEndpoint("encode", "/encode", "POST"),
      tokenizer_(tokenizer) {
}

EncodeEndpoint::~EncodeEndpoint() {
}

void EncodeEndpoint::setTokenizer(ITokenizer* tokenizer) {
    tokenizer_ = tokenizer;
}

EncodeEndpoint::EncodeRequest EncodeEndpoint::parseRequest(const HttpRequest& request) {
    EncodeRequest req;
    
    std::string body = request.getBody();
    req.text = "";
    
    try {
        nlohmann::json jsonBody = nlohmann::json::parse(body);
        
        if (jsonBody.contains("text") && jsonBody["text"].is_string()) {
            req.text = jsonBody["text"].get<std::string>();
        }
    } catch (const nlohmann::json::exception& e) {
        CLLM_WARN("Failed to parse JSON request body: %s, using empty text", e.what());
    }
    
    return req;
}

HttpResponse EncodeEndpoint::handle(const HttpRequest& request) {
    try {
        EncodeRequest req = parseRequest(request);
        
        if (req.text.empty()) {
            return HttpResponse::badRequest("Missing required field: text");
        }
        
        if (tokenizer_ == nullptr) {
            return HttpResponse::internalError("Tokenizer is not initialized");
        }
        
        std::vector<int> tokens = tokenizer_->encode(req.text, true);
        
        std::ostringstream oss;
        oss << "{";
        oss << "\"tokens\":[";
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (i > 0) {
                oss << ",";
            }
            oss << tokens[i];
        }
        oss << "],";
        oss << "\"length\":" << tokens.size();
        oss << "}";
        
        HttpResponse response;
        response.setStatusCode(200);
        response.setBody(oss.str());
        response.setContentType("application/json");
        
        return response;
    } catch (const std::exception& e) {
        return HttpResponse::internalError(e.what());
    }
}

}
