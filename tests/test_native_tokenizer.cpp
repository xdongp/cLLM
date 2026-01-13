#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>

// Mock the necessary components for testing

// Mock ITokenizer interface
class ITokenizer {
public:
    virtual ~ITokenizer() = default;
    virtual bool load(const std::string& modelPath) = 0;
    virtual std::vector<int> encode(const std::string& text, bool addSpecialTokens) = 0;
    virtual std::string decode(const std::vector<int>& ids, bool skipSpecialTokens) = 0;
    virtual int getVocabSize() const = 0;
};

// Mock ModelType enum
enum class ModelType {
    LLAMA,
    QWEN,
    OTHER
};

// Simplified version of NativeTokenizer with the fix
class NativeTokenizerTest : public ITokenizer {
public:
    NativeTokenizerTest() {}
    ~NativeTokenizerTest() override = default;

    bool load(const std::string& modelPath) override {
        // Test mode: use built-in vocabulary
        vocab_ = {
            {"hello", 100},
            {"world", 101},
            {"!", 102}
        };
        idToToken_ = {
            {100, "hello"},
            {101, "world"},
            {102, "!"}
        };
        return true;
    }

    std::vector<int> encode(const std::string& text, bool addSpecialTokens) override {
        std::vector<int> ids;
        
        // Improved encoding logic from the fix
        std::stringstream ss(text);
        std::string word;
        
        while (ss >> word) {
            if (word == "hello") {
                ids.push_back(100);
            } else if (word == "world") {
                ids.push_back(101);
            } else {
                // Hash-based mapping to avoid single ID mapping
                size_t hash = std::hash<std::string>{}(word);
                ids.push_back(100 + (hash % 3)); // 100-102
            }
        }
        
        if (ids.empty()) {
            ids.push_back(100);
        }
        
        return ids;
    }

    std::string decode(const std::vector<int>& ids, bool skipSpecialTokens) override {
        std::string text;
        for (int id : ids) {
            auto it = idToToken_.find(id);
            if (it != idToToken_.end()) {
                text += it->second + " ";
            } else {
                // Improved decoding logic from the fix
                if (id >= 100 && id < 200) {
                    text += "word_" + std::to_string(id - 100) + " ";
                } else {
                    int mappedId = 100 + (id % 3);
                    auto mapIt = idToToken_.find(mappedId);
                    if (mapIt != idToToken_.end()) {
                        text += mapIt->second + " ";
                    } else {
                        text += "[UNK] ";
                    }
                }
            }
        }
        if (!text.empty()) text.pop_back();
        return text;
    }

    int getVocabSize() const override {
        return static_cast<int>(vocab_.size());
    }

private:
    std::unordered_map<std::string, int> vocab_;
    std::unordered_map<int, std::string> idToToken_;
};

int main() {
    std::cout << "Testing NativeTokenizer Fix..." << std::endl;
    
    NativeTokenizerTest tokenizer;
    tokenizer.load("test");
    
    // Test encoding
    std::vector<std::string> testTexts = {
        "hello",
        "hello world",
        "test input",
        "another example"
    };
    
    for (const auto& text : testTexts) {
        std::vector<int> ids = tokenizer.encode(text, false);
        std::string decoded = tokenizer.decode(ids, false);
        
        std::cout << "\nInput: " << text << std::endl;
        std::cout << "Encoded: ";
        for (int id : ids) {
            std::cout << id << " ";
        }
        std::cout << std::endl;
        std::cout << "Decoded: " << decoded << std::endl;
    }
    
    return 0;
}
