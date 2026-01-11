#include "cllm/http/health_endpoint.h"
#include <sstream>

namespace cllm {

HealthEndpoint::HealthEndpoint()
    : ApiEndpoint("health", "/health", "GET") {
}

HealthEndpoint::~HealthEndpoint() {
}

HttpResponse HealthEndpoint::handle(const HttpRequest& request) {
    std::ostringstream oss;
    oss << "{\"status\":\"healthy\",\"model_loaded\":true}";
    
    HttpResponse response;
    response.setStatusCode(200);
    response.setBody(oss.str());
    response.setContentType("application/json");
    
    return response;
}

}
