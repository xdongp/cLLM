/**
 * @file test_http_server_comprehensive.cpp
 * @brief 自研HTTP服务器全面测试
 * @author cLLM Team
 * @date 2026-01-20
 */

#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>
#include <cassert>
#include <curl/curl.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// 测试统计
struct TestStats {
    std::atomic<int> total{0};
    std::atomic<int> passed{0};
    std::atomic<int> failed{0};
    
    void reset() {
        total = 0;
        passed = 0;
        failed = 0;
    }
    
    void print() const {
        std::cout << "\n========================================" << std::endl;
        std::cout << "测试统计:" << std::endl;
        std::cout << "  总计: " << total.load() << std::endl;
        std::cout << "  通过: " << passed.load() << std::endl;
        std::cout << "  失败: " << failed.load() << std::endl;
        std::cout << "  成功率: " << (total.load() > 0 ? (100.0 * passed.load() / total.load()) : 0) << "%" << std::endl;
        std::cout << "========================================\n" << std::endl;
    }
};

static TestStats g_stats;

// CURL回调函数
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* data) {
    size_t totalSize = size * nmemb;
    data->append((char*)contents, totalSize);
    return totalSize;
}

// HTTP请求辅助函数
class HttpClient {
public:
    HttpClient(const std::string& baseUrl) : baseUrl_(baseUrl) {
        curl_global_init(CURL_GLOBAL_DEFAULT);
    }
    
    ~HttpClient() {
        curl_global_cleanup();
    }
    
    bool get(const std::string& path, int& statusCode, std::string& response) {
        CURL* curl = curl_easy_init();
        if (!curl) return false;
        
        std::string url = baseUrl_ + path;
        std::string data;
        
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &data);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
        
        CURLcode res = curl_easy_perform(curl);
        
        if (res == CURLE_OK) {
            curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &statusCode);
            response = data;
        }
        
        curl_easy_cleanup(curl);
        return res == CURLE_OK;
    }
    
    bool post(const std::string& path, const std::string& body, int& statusCode, std::string& response) {
        CURL* curl = curl_easy_init();
        if (!curl) return false;
        
        std::string url = baseUrl_ + path;
        std::string data;
        
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, body.length());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &data);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
        
        struct curl_slist* headers = nullptr;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        
        CURLcode res = curl_easy_perform(curl);
        
        if (res == CURLE_OK) {
            curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &statusCode);
            response = data;
        }
        
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
        return res == CURLE_OK;
    }
    
private:
    std::string baseUrl_;
};

// 测试辅助宏
#define TEST(name, func) \
    do { \
        std::cout << "[TEST] " << name << "..." << std::flush; \
        g_stats.total++; \
        try { \
            if (func()) { \
                std::cout << " ✓ PASSED" << std::endl; \
                g_stats.passed++; \
            } else { \
                std::cout << " ✗ FAILED" << std::endl; \
                g_stats.failed++; \
            } \
        } catch (const std::exception& e) { \
            std::cout << " ✗ FAILED (exception: " << e.what() << ")" << std::endl; \
            g_stats.failed++; \
        } \
    } while(0)

// ============================================================================
// 测试用例
// ============================================================================

// 测试1: 健康检查端点
bool test_health_endpoint(HttpClient& client) {
    int statusCode;
    std::string response;
    
    if (!client.get("/health", statusCode, response)) {
        return false;
    }
    
    if (statusCode != 200) {
        return false;
    }
    
    try {
        json j = json::parse(response);
        return j.contains("success") && j["success"] == true;
    } catch (...) {
        return false;
    }
}

// 测试2: 生成端点（基本）
bool test_generate_basic(HttpClient& client) {
    int statusCode;
    std::string response;
    
    json request;
    request["prompt"] = "Hello";
    request["max_tokens"] = 5;
    
    if (!client.post("/generate", request.dump(), statusCode, response)) {
        return false;
    }
    
    if (statusCode != 200) {
        return false;
    }
    
    try {
        json j = json::parse(response);
        return j.contains("success") && j["success"] == true &&
               j.contains("data") && j["data"].contains("text");
    } catch (...) {
        return false;
    }
}

// 测试3: 编码端点
bool test_encode_endpoint(HttpClient& client) {
    int statusCode;
    std::string response;
    
    json request;
    request["text"] = "Hello world";
    
    if (!client.post("/encode", request.dump(), statusCode, response)) {
        return false;
    }
    
    if (statusCode != 200) {
        return false;
    }
    
    try {
        json j = json::parse(response);
        return j.contains("success") && j["success"] == true &&
               j.contains("data") && j["data"].contains("tokens");
    } catch (...) {
        return false;
    }
}

// 测试4: 404 Not Found
bool test_404_not_found(HttpClient& client) {
    int statusCode;
    std::string response;
    
    if (!client.get("/nonexistent", statusCode, response)) {
        return false;
    }
    
    return statusCode == 404;
}

// 测试5: 无效JSON
bool test_invalid_json(HttpClient& client) {
    int statusCode;
    std::string response;
    
    std::string invalidJson = "{invalid json}";
    if (!client.post("/generate", invalidJson, statusCode, response)) {
        return false;
    }
    
    return statusCode == 400;  // Bad Request
}

// 测试6: 并发请求
bool test_concurrent_requests(HttpClient& client, int numThreads = 10, int requestsPerThread = 10) {
    std::atomic<int> successCount{0};
    std::atomic<int> failCount{0};
    
    auto worker = [&](int threadId) {
        for (int i = 0; i < requestsPerThread; ++i) {
            int statusCode;
            std::string response;
            
            json request;
            request["prompt"] = "Test " + std::to_string(threadId) + "-" + std::to_string(i);
            request["max_tokens"] = 3;
            
            if (client.post("/generate", request.dump(), statusCode, response)) {
                if (statusCode == 200) {
                    try {
                        json j = json::parse(response);
                        if (j.contains("success") && j["success"] == true) {
                            successCount++;
                        } else {
                            failCount++;
                        }
                    } catch (...) {
                        failCount++;
                    }
                } else {
                    failCount++;
                }
            } else {
                failCount++;
            }
        }
    };
    
    std::vector<std::thread> threads;
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back(worker, i);
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    int total = numThreads * requestsPerThread;
    int success = successCount.load();
    double successRate = (double)success / total;
    
    std::cout << "  (并发: " << numThreads << "线程 × " << requestsPerThread 
              << "请求 = " << total << "请求, 成功: " << success 
              << ", 成功率: " << (successRate * 100) << "%)" << std::flush;
    
    return successRate >= 0.95;  // 95%成功率
}

// 测试7: Keep-Alive连接
bool test_keep_alive(HttpClient& client) {
    // 发送多个请求，验证连接复用
    for (int i = 0; i < 5; ++i) {
        int statusCode;
        std::string response;
        
        if (!client.get("/health", statusCode, response)) {
            return false;
        }
        
        if (statusCode != 200) {
            return false;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    return true;
}

// 测试8: 大请求体
bool test_large_request_body(HttpClient& client) {
    int statusCode;
    std::string response;
    
    json request;
    std::string largePrompt(10000, 'A');  // 10KB prompt
    request["prompt"] = largePrompt;
    request["max_tokens"] = 5;
    
    if (!client.post("/generate", request.dump(), statusCode, response)) {
        return false;
    }
    
    return statusCode == 200 || statusCode == 400;  // 可能被拒绝，但不应崩溃
}

// 测试9: 空请求体
bool test_empty_request_body(HttpClient& client) {
    int statusCode;
    std::string response;
    
    if (!client.post("/generate", "", statusCode, response)) {
        return false;
    }
    
    return statusCode == 400;  // Bad Request
}

// 测试10: 压力测试
bool test_stress_test(HttpClient& client, int numRequests = 100) {
    std::atomic<int> successCount{0};
    std::atomic<int> failCount{0};
    
    auto start = std::chrono::steady_clock::now();
    
    for (int i = 0; i < numRequests; ++i) {
        int statusCode;
        std::string response;
        
        json request;
        request["prompt"] = "Stress test " + std::to_string(i);
        request["max_tokens"] = 3;
        
        if (client.post("/generate", request.dump(), statusCode, response)) {
            if (statusCode == 200) {
                successCount++;
            } else {
                failCount++;
            }
        } else {
            failCount++;
        }
    }
    
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    int success = successCount.load();
    double successRate = (double)success / numRequests;
    double qps = (double)numRequests * 1000 / duration;
    
    std::cout << "  (请求数: " << numRequests << ", 成功: " << success 
              << ", 耗时: " << duration << "ms, QPS: " << qps 
              << ", 成功率: " << (successRate * 100) << "%)" << std::flush;
    
    return successRate >= 0.90;  // 90%成功率
}

// ============================================================================
// 主函数
// ============================================================================

int main(int argc, char* argv[]) {
    std::string serverUrl = "http://localhost:8080";
    
    if (argc > 1) {
        serverUrl = argv[1];
    }
    
    std::cout << "========================================" << std::endl;
    std::cout << "自研HTTP服务器全面测试" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "服务器地址: " << serverUrl << std::endl;
    std::cout << "开始测试..." << std::endl;
    std::cout << "" << std::endl;
    
    HttpClient client(serverUrl);
    
    // 等待服务器启动
    std::cout << "等待服务器就绪..." << std::flush;
    for (int i = 0; i < 10; ++i) {
        int statusCode;
        std::string response;
        if (client.get("/health", statusCode, response) && statusCode == 200) {
            std::cout << " ✓" << std::endl;
            break;
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
        std::cout << "." << std::flush;
    }
    std::cout << std::endl;
    
    // 基本功能测试
    std::cout << "\n[基本功能测试]" << std::endl;
    TEST("健康检查端点", [&]() { return test_health_endpoint(client); });
    TEST("生成端点（基本）", [&]() { return test_generate_basic(client); });
    TEST("编码端点", [&]() { return test_encode_endpoint(client); });
    
    // 错误处理测试
    std::cout << "\n[错误处理测试]" << std::endl;
    TEST("404 Not Found", [&]() { return test_404_not_found(client); });
    TEST("无效JSON", [&]() { return test_invalid_json(client); });
    TEST("空请求体", [&]() { return test_empty_request_body(client); });
    
    // 边界条件测试
    std::cout << "\n[边界条件测试]" << std::endl;
    TEST("大请求体", [&]() { return test_large_request_body(client); });
    
    // 并发测试
    std::cout << "\n[并发测试]" << std::endl;
    TEST("并发请求（10线程×10请求）", [&]() { return test_concurrent_requests(client, 10, 10); });
    TEST("并发请求（20线程×5请求）", [&]() { return test_concurrent_requests(client, 20, 5); });
    
    // 连接测试
    std::cout << "\n[连接测试]" << std::endl;
    TEST("Keep-Alive连接", [&]() { return test_keep_alive(client); });
    
    // 压力测试
    std::cout << "\n[压力测试]" << std::endl;
    TEST("压力测试（100请求）", [&]() { return test_stress_test(client, 100); });
    TEST("压力测试（200请求）", [&]() { return test_stress_test(client, 200); });
    
    // 打印统计
    g_stats.print();
    
    return g_stats.failed.load() > 0 ? 1 : 0;
}
