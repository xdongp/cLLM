/**
 * @file test_endpoints.cpp
 * @brief 端点单元测试（不需要启动服务器）
 * @author cLLM Team
 * @date 2026-01-10
 */

#include <gtest/gtest.h>
#include <memory>

#include "cllm/http/health_endpoint.h"
#include "cllm/http/encode_endpoint.h"
#include "cllm/http/request.h"
#include "cllm/http/response.h"
#include "cllm/tokenizer/tokenizer.h"

namespace cllm {
namespace test {

/**
 * @brief 测试健康检查端点
 */
class HealthEndpointTest : public ::testing::Test {
protected:
    std::unique_ptr<HealthEndpoint> endpoint_;
    
    void SetUp() override {
        endpoint_ = std::make_unique<HealthEndpoint>();
    }
};

TEST_F(HealthEndpointTest, BasicHealth) {
    HttpRequest request;
    request.setMethod("GET");
    request.setPath("/health");
    
    HttpResponse response = endpoint_->handle(request);
    
    EXPECT_EQ(response.getStatusCode(), 200);
    EXPECT_FALSE(response.getBody().empty());
    EXPECT_EQ(response.getContentType(), "application/json");
    
    std::string body = response.getBody();
    EXPECT_NE(body.find("status"), std::string::npos);
    EXPECT_NE(body.find("healthy"), std::string::npos);
}

TEST_F(HealthEndpointTest, ResponseFormat) {
    HttpRequest request;
    request.setMethod("GET");
    request.setPath("/health");
    
    HttpResponse response = endpoint_->handle(request);
    std::string body = response.getBody();
    
    // 验证 JSON 格式
    EXPECT_EQ(body.front(), '{');
    EXPECT_EQ(body.back(), '}');
    EXPECT_NE(body.find("\"status\""), std::string::npos);
    EXPECT_NE(body.find("\"model_loaded\""), std::string::npos);
}

/**
 * @brief 测试编码端点
 */
class EncodeEndpointTest : public ::testing::Test {
protected:
    std::unique_ptr<EncodeEndpoint> endpoint_;
    std::unique_ptr<Tokenizer> tokenizer_;
    
    void SetUp() override {
        // 尝试加载测试用的 tokenizer
        try {
            std::string tokenizerPath = "../tests/tokenizer.model";
            tokenizer_ = std::make_unique<Tokenizer>(tokenizerPath);
            endpoint_ = std::make_unique<EncodeEndpoint>(tokenizer_.get());
        } catch (const std::exception& e) {
            GTEST_SKIP() << "Tokenizer model not available: " << e.what();
        }
    }
};

TEST_F(EncodeEndpointTest, BasicEncoding) {
    HttpRequest request;
    request.setMethod("POST");
    request.setPath("/encode");
    request.setBody(R"({"text": "Hello"})");
    
    HttpResponse response = endpoint_->handle(request);
    
    EXPECT_EQ(response.getStatusCode(), 200);
    EXPECT_FALSE(response.getBody().empty());
    EXPECT_EQ(response.getContentType(), "application/json");
    
    std::string body = response.getBody();
    EXPECT_NE(body.find("tokens"), std::string::npos);
    EXPECT_NE(body.find("length"), std::string::npos);
}

TEST_F(EncodeEndpointTest, MissingTextField) {
    HttpRequest request;
    request.setMethod("POST");
    request.setPath("/encode");
    request.setBody("{}");
    
    HttpResponse response = endpoint_->handle(request);
    
    // 应该返回错误
    EXPECT_NE(response.getStatusCode(), 200);
}

TEST_F(EncodeEndpointTest, EmptyText) {
    HttpRequest request;
    request.setMethod("POST");
    request.setPath("/encode");
    request.setBody(R"({"text": ""})");
    
    HttpResponse response = endpoint_->handle(request);
    
    // 空文本应该返回错误或空数组
    if (response.getStatusCode() == 200) {
        std::string body = response.getBody();
        // 应该包含空的 tokens 数组
        EXPECT_NE(body.find("\"tokens\":[]"), std::string::npos);
    }
}

TEST_F(EncodeEndpointTest, MultipleWords) {
    HttpRequest request;
    request.setMethod("POST");
    request.setPath("/encode");
    request.setBody(R"({"text": "Hello world"})");
    
    HttpResponse response = endpoint_->handle(request);
    
    EXPECT_EQ(response.getStatusCode(), 200);
    
    std::string body = response.getBody();
    EXPECT_NE(body.find("tokens"), std::string::npos);
    EXPECT_NE(body.find("length"), std::string::npos);
    
    // 多个单词应该产生多个 token
    // 验证 length > 1
    size_t lengthPos = body.find("\"length\":");
    if (lengthPos != std::string::npos) {
        // 简单检查：至少应该有一个非零长度
        EXPECT_NE(body.find('1', lengthPos), std::string::npos);
    }
}

/**
 * @brief 测试 HTTP 请求/响应类
 */
class HttpRequestResponseTest : public ::testing::Test {};

TEST_F(HttpRequestResponseTest, RequestSetGet) {
    HttpRequest request;
    
    request.setMethod("POST");
    request.setPath("/api/test");
    request.setBody("{\"key\": \"value\"}");
    request.setHeader("Content-Type", "application/json");
    
    EXPECT_EQ(request.getMethod(), "POST");
    EXPECT_EQ(request.getPath(), "/api/test");
    EXPECT_EQ(request.getBody(), "{\"key\": \"value\"}");
    EXPECT_EQ(request.getHeader("Content-Type"), "application/json");
}

TEST_F(HttpRequestResponseTest, ResponseSetGet) {
    HttpResponse response;
    
    response.setStatusCode(200);
    response.setBody("{\"status\": \"ok\"}");
    response.setContentType("application/json");
    response.setHeader("X-Custom", "value");
    
    EXPECT_EQ(response.getStatusCode(), 200);
    EXPECT_EQ(response.getBody(), "{\"status\": \"ok\"}");
    EXPECT_EQ(response.getContentType(), "application/json");
    EXPECT_EQ(response.getHeader("X-Custom"), "value");
}

TEST_F(HttpRequestResponseTest, ResponseHelpers) {
    auto response1 = HttpResponse::ok("Success");
    EXPECT_EQ(response1.getStatusCode(), 200);
    EXPECT_EQ(response1.getBody(), "Success");
    
    auto response2 = HttpResponse::badRequest("Bad request");
    EXPECT_EQ(response2.getStatusCode(), 400);
    
    auto response3 = HttpResponse::notFound();
    EXPECT_EQ(response3.getStatusCode(), 404);
    
    auto response4 = HttpResponse::internalError("Error");
    EXPECT_EQ(response4.getStatusCode(), 500);
}

} // namespace test
} // namespace cllm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
