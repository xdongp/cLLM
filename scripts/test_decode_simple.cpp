#include <iostream>
#include <vector>
#include <fstream>
#include <tokenizers_cpp.h>

int main() {
    std::string modelPath = "/Users/dannypan/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B/tokenizer.json";
    
    std::cout << "Loading tokenizer from: " << modelPath << std::endl;
    
    try {
        // Load tokenizer
        std::ifstream f(modelPath);
        std::string json_blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        auto tokenizer = tokenizers::Tokenizer::FromBlobJSON(json_blob);
        
        std::cout << "Tokenizer loaded, vocab size: " << tokenizer->GetVocabSize() << std::endl;
        
        // Test 1: Encode and decode
        std::string text = "hello world";
        auto encoding = tokenizer->Encode(text);
        std::cout << "\nTest 1: Encode '" << text << "'" << std::endl;
        std::cout << "Tokens: [";
        for (size_t i = 0; i < encoding.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << encoding[i];
        }
        std::cout << "]" << std::endl;
        
        // Decode back
        std::vector<int32_t> ids(encoding.begin(), encoding.end());
        std::string decoded = tokenizer->Decode(ids);
        std::cout << "Decoded: '" << decoded << "'" << std::endl;
        
        // Test 2: Decode specific tokens
        std::vector<int32_t> testTokens = {14990, 1879};  // hello, world
        std::cout << "\nTest 2: Decode tokens [14990, 1879]" << std::endl;
        std::string decoded2 = tokenizer->Decode(testTokens);
        std::cout << "Decoded: '" << decoded2 << "'" << std::endl;
        
        // Test 3: Single token
        std::vector<int32_t> singleToken = {14990};
        std::cout << "\nTest 3: Decode single token [14990]" << std::endl;
        std::string decoded3 = tokenizer->Decode(singleToken);
        std::cout << "Decoded: '" << decoded3 << "'" << std::endl;
        
        std::cout << "\nAll tests passed!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
