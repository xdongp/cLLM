#include "cllm/http/health_endpoint.h"
#include "cllm/common/config.h"
#include <sstream>

namespace cllm {

HealthEndpoint::HealthEndpoint()
    : ApiEndpoint(cllm::Config::instance().apiEndpointHealthName(), cllm::Config::instance().apiEndpointHealthPath(), cllm::Config::instance().apiEndpointHealthMethod()) {
}

HealthEndpoint::~HealthEndpoint() {
}

HttpResponse HealthEndpoint::handle(const HttpRequest& request) {
    std::ostringstream oss;
    oss << "{\"status\":\"healthy\",\"model_loaded\":true}";
    
    HttpResponse response;
    response.setStatusCode(200);
    response.setBody(oss.str());
    response.setContentType(cllm::Config::instance().apiResponseContentTypeJson());
    
    return response;
}

}
