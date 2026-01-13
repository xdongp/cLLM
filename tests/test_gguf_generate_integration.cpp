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
#include "cllm/model/gguf_loader_new.h"

int main(int argc, char* argv[]) {
    try {
        std::cout << "[TEST] Starting GGUF generate endpoint integration test..." << std::endl;
        
        // Check if a GGUF model path is provided
        if (argc < 2) {
            std::cerr << "[TEST ERROR] Please provide a path to a GGUF model file" << std::endl;
            std::cerr << "Usage: " << argv[0] << " <gguf_model_path>" << std::endl;
            return 1;
        }
        
        std::string ggufModelPath = argv[1];
        std::cout << "[TEST] Using GGUF model: " << ggufModelPath << std::endl;
        
        // Initialize configuration
        cllm::Config::instance().load("../config/sampler_config.yaml");
        cllm::Config::instance().load("../config/test_config.yaml");
        
        // Create a model executor with GGUF model
        std::unique_ptr<cllm::ModelExecutor> modelExecutor = 
            std::make_unique<cllm::ModelExecutor>(ggufModelPath, "", true, false);
        
        // Load the model
        std::cout << "[TEST] Loading model..." << std::endl;
        modelExecutor->loadModel();
        
        if (!modelExecutor->isLoaded()) {
            std::cerr << "[TEST ERROR] Failed to load model" << std::endl;
            return 1;
        }
        
        std::cout << "[TEST] Model loaded successfully" << std::endl;
        
        // Create a tokenizer manager
        std::unique_ptr<cllm::TokenizerManager> tokenizerManager = 
            std::make_unique<cllm::TokenizerManager>("../tests", modelExecutor.get());
        
        cllm::ITokenizer* tokenizer = tokenizerManager->getTokenizer();
        if (!tokenizer) {
            std::cerr << "[TEST ERROR] Failed to create tokenizer" << std::endl;
            return 1;
        }
        
        std::cout << "[TEST] Tokenizer created successfully" << std::endl;
        
        // Create scheduler
        std::unique_ptr<cllm::Scheduler> scheduler = 
            std::make_unique<cllm::Scheduler>(modelExecutor.get(), 8, 2048);
        scheduler->start();
        
        std::cout << "[TEST] Scheduler started" << std::endl;
        
        // Create generate endpoint
        std::shared_ptr<cllm::GenerateEndpoint> generateEndpoint = 
            std::make_shared<cllm::GenerateEndpoint>(scheduler.get(), tokenizer);
        
        // Create test request with input "hello"
        cllm::HttpRequest request;
        request.setMethod("POST");
        request.setPath("/generate");
        
        // Set request body
        std::string requestBody = R"({
            "prompt": "hello",
            "max_tokens": 10,
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": false
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
            std::cout << "[TEST] Integration test completed successfully!" << std::endl;
        } else {
            std::cerr << "[TEST ERROR] Failed to parse response or missing 'text' field" << std::endl;
            if (root.isMember("error")) {
                std::cerr << "[TEST ERROR] Error message: " << root["error"].asString() << std::endl;
            }
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