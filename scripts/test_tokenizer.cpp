#include <iostream>
#include <vector>
#include <string>
#include "cllm/tokenizer/hf_tokenizer.h"
#include "cllm/tokenizer/tokenizer_manager.h"

int main() {
    std::string modelPath = "/Users/dannypan/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B";
    
    std::cout << "Testing HFTokenizer..." << std::endl;
    
    // Test 1: Direct HFTokenizer
    {
        cllm::HFTokenizer tokenizer(cllm::ModelType::QWEN);
        if (!tokenizer.load(modelPath)) {
            std::cerr << "Failed to load tokenizer" << std::endl;
            return 1;
        }
        
        // Encode
        std::string text = "hello world";
        std::vector<int> ids = tokenizer.encode(text, false);
        std::cout << "Encoded '" << text << "' to " << ids.size() << " tokens: [";
        for (size_t i = 0; i < ids.size() && i < 10; ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << ids[i];
        }
        std::cout << "]" << std::endl;
        
        // Decode
        std::string decoded = tokenizer.decode(ids, true);
        std::cout << "Decoded back to: '" << decoded << "'" << std::endl;
        
        // Test with some artificial tokens
        std::vector<int> testTokens = {9707, 11, 1879, 0};  // hello, ,, world, !
        std::string decoded2 = tokenizer.decode(testTokens, true);
        std::cout << "Decoded test tokens [9707, 11, 1879, 0] to: '" << decoded2 << "'" << std::endl;
    }
    
    std::cout << "\nTokenizer test completed!" << std::endl;
    return 0;
}
