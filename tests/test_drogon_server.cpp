#include <gtest/gtest.h>
#include <drogon/drogon_test.h>
#include <drogon/HttpAppFramework.h>
#include <cllm/http/drogon_server.h>
#include <cllm/http/handler.h>

using namespace cllm;
using namespace drogon;

class TestHttpHandler : public HttpHandler {
public:
    TestHttpHandler() : callCount(0) {}
    
    HttpResponse handleRequest(const HttpRequest& request) override {
        callCount++;
        lastRequest = request;
        
        if (request.getPath() == "/health") {
            HttpResponse response;
            response.setStatusCode(200);
            response.setBody(R"({"status":"healthy"})");
            return response;
        }
        return HttpResponse::notFound();
    }
    
    int callCount;
    HttpRequest lastRequest;
};

class DrogonServerTest : public ::testing::Test {
protected:
    void SetUp() override {
        handler_ = std::make_shared<TestHttpHandler>();
        DrogonServer::init("127.0.0.1", 8080, handler_.get());
        app().run();
    }

    void TearDown() override {
        DrogonServer::stop();
        app().quit();
    }

    TestClientPtr client_ = TestClient::newHttpClient("http://127.0.0.1:8080");
    std::shared_ptr<TestHttpHandler> handler_;
};

TEST_F(DrogonServerTest, HealthCheck) {
    auto resp = client_->sendRequest(
        HttpRequest::newHttpRequest("/health", HttpMethod::Get));
    
    EXPECT_EQ(resp->getStatusCode(), HttpStatusCode::k200OK);
    EXPECT_EQ(handler_->callCount, 1);
    EXPECT_EQ(resp->getBody(), R"({"status":"healthy"})");
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}