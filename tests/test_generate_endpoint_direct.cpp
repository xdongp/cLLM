#include <iostream>
#include <string>
#include <memory>
#include <json/json.h>

#include "cllm/http/generate_endpoint.h"
#include "cllm/scheduler/scheduler.h"
#include "cllm/model/executor.h"
#include "cllm/tokenizer/manager.h"
#include "cllm/http/request.h"
#include "cllm/http/response.h"
#include "cllm/config/config.h"

int main() {
    try {
        std::cout << "[TEST] Starting direct generate endpoint test..." << std::endl;
        
        // Initialize configuration
        cllm::Config::instance().load("../config/sampler_config.yaml");
        cllm::Config::instance().load("../config/test_config.yaml");
        
        // Create a mock model executor with placeholder weights
        std::unique_ptr<cllm::ModelExecutor> modelExecutor = 
            std::make_unique<cllm::ModelExecutor>("", "", true, false);
        
        // Create a mock tokenizer manager
        std::unique_ptr<cllm::TokenizerManager> tokenizerManager = 
            std::make_unique<cllm::TokenizerManager>("../tests", modelExecutor.get());
        
        cllm::ITokenizer* tokenizer = tokenizerManager->getTokenizer();
        if (!tokenizer) {
            std::cerr << "[TEST ERROR] Failed to create tokenizer" << std::endl;
            return 1;
        }
        
        // Create scheduler
        std::unique_ptr<cllm::Scheduler> scheduler = 
            std::make_unique<cllm::Scheduler>(modelExecutor.get(), 8, 2048);
        scheduler->start();
        
        // Create generate endpoint
        std::shared_ptr<cllm::GenerateEndpoint> generateEndpoint = 
            std::make_shared<cllm::GenerateEndpoint>(scheduler.get(), tokenizer);
        
        // Create test request
        cllm::HttpRequest request;
        request.setMethod("POST");
        request.setPath("/generate");
        
        // Set request body
        std::string requestBody = R"({
            "prompt": "hello",
            "max_tokens": 5,
            "temperature": 0.7,
            "top_p": 0.9
        })";
        request.setBody(requestBody);
        
        std::cout << "[TEST] Sending generate request with prompt: 'hello'" << std::endl;
        
        // Process request
        cllm::HttpResponse response = generateEndpoint->handle(request);
        
        // Check response
        std::cout << "[TEST] Response status: " << response.getStatusCode() << std::endl;
        std::cout << "[TEST] Response body: " << response.getBody() << std::endl;
        
        // Parse JSON response
        Json::Value root;
        Json::Reader reader;
        bool parsingSuccessful = reader.parse(response.getBody(), root);
        
        if (parsingSuccessful && root.isMember("text")) {
            std::string generatedText = root["text"].asString();
            std::cout << "[TEST] Generated text: " << generatedText << std::endl;
            std::cout << "[TEST] Test completed successfully!" << std::endl;
        } else {
            std::cerr << "[TEST ERROR] Failed to parse response or missing 'text' field" << std::endl;
            return 1;
        }
        
        // Cleanup
        scheduler->stop();
        
    } catch (const std::exception& e) {
        std::cerr << "[TEST ERROR] Exception occurred: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}