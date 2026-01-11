#include "cllm/http/encode_endpoint.h"
#include "cllm/tokenizer/tokenizer.h"
#include <sstream>
#include <stdexcept>

namespace cllm {

EncodeEndpoint::EncodeEndpoint(Tokenizer* tokenizer)
    : ApiEndpoint("encode", "/encode", "POST"),
      tokenizer_(tokenizer) {
}

EncodeEndpoint::~EncodeEndpoint() {
}

void EncodeEndpoint::setTokenizer(Tokenizer* tokenizer) {
    tokenizer_ = tokenizer;
}

EncodeEndpoint::EncodeRequest EncodeEndpoint::parseRequest(const HttpRequest& request) {
    EncodeRequest req;
    
    std::string body = request.getBody();
    
    req.text = "";
    
    size_t textPos = body.find("\"text\"");
    if (textPos != std::string::npos) {
        size_t start = body.find("\"", textPos + 7);
        size_t end = body.find("\"", start + 1);
        if (start != std::string::npos && end != std::string::npos) {
            req.text = body.substr(start + 1, end - start - 1);
        }
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
