/**
 * @file main.cpp
 * @brief cLLM 服务器主入口
 * @author cLLM Team
 * @date 2026-01-10
 */

#include "cllm/http/drogon_server.h"

#include <signal.h>
#include <getopt.h>

#include <memory>
#include <string>

#include "cllm/http/handler.h"
#include "cllm/http/health_endpoint.h"
#include "cllm/http/generate_endpoint.h"
#include "cllm/http/encode_endpoint.h"
#include "cllm/scheduler/scheduler.h"
#include "cllm/model/executor.h"
#include "cllm/tokenizer/tokenizer.h"
#include "cllm/common/config.h"
#include "cllm/common/logger.h"
#include "cllm/common/asio_handler.h"

// 全局变量用于信号处理
static std::unique_ptr<cllm::Scheduler> g_scheduler;
static std::unique_ptr<cllm::ModelExecutor> g_modelExecutor;
static std::unique_ptr<cllm::Tokenizer> g_tokenizer;

/**
 * @brief 信号处理函数
 * @param signal 信号编号
 */
void signalHandler(int signal) {
    CLLM_INFO("Received signal {}, shutting down gracefully...", signal);
    
    // 停止调度器
    if (g_scheduler) {
        CLLM_INFO("Stopping scheduler...");
        g_scheduler->stop();
    }
    
    // 停止服务器
    CLLM_INFO("Stopping HTTP server...");
    cllm::DrogonServer::stop();
    
    CLLM_INFO("Shutdown complete");
    cllm::Logger::instance().flush();
    exit(0);
}

/**
 * @brief 打印使用说明
 * @param programName 程序名称
 */
void printUsage(const char* programName) {
    CLLM_INFO("Usage: {} [options]", programName);
    CLLM_INFO("");
    CLLM_INFO("Options:");
    CLLM_INFO("  --model-path PATH        Path to model directory (required)");
    CLLM_INFO("  --port PORT              Server port (default: 8080)");
    CLLM_INFO("  --host HOST              Server host (default: 0.0.0.0)");
    CLLM_INFO("  --quantization TYPE      Quantization type: fp16, int8, int4 (default: fp16)");
    CLLM_INFO("  --max-batch-size SIZE    Maximum batch size (default: 8)");
    CLLM_INFO("  --max-context-length LEN Maximum context length (default: 2048)");
    CLLM_INFO("  --use-libtorch           Use LibTorch backend (default: false, use Kylin)");
    CLLM_INFO("  --config PATH            Path to config file (optional)");
    CLLM_INFO("  --log-level LEVEL        Log level: trace, debug, info, warn, error (default: info)");
    CLLM_INFO("  --log-file PATH          Log file path (optional)");
    CLLM_INFO("  --help                   Show this help message");
    CLLM_INFO("");
    CLLM_INFO("Examples:");
    CLLM_INFO("  {} --model-path /path/to/model", programName);
    CLLM_INFO("  {} --model-path /path/to/model --port 9000 --use-libtorch", programName);
}

/**
 * @brief 打印服务器横幅
 */
void printBanner() {
    CLLM_INFO("");
    CLLM_INFO("  _____ _      _      __  __ ");
    CLLM_INFO(" / ____| |    | |    |  \\/  |");
    CLLM_INFO("| |    | |    | |    | \\  / |");
    CLLM_INFO("| |    | |    | |    | |\\/| |");
    CLLM_INFO("| |____| |____| |____| |  | |");
    CLLM_INFO(" \\_____|______|______|_|  |_|");
    CLLM_INFO("");
}

/**
 * @brief 主函数
 * @param argc 参数个数
 * @param argv 参数数组
 * @return 程序退出码
 */
int main(int argc, char* argv[]) {
    // 初始化日志系统
    cllm::Logger::instance().setLevel(spdlog::level::info);
    
    // 配置参数
    std::string modelPath;
    std::string host = "0.0.0.0";
    int port = 8080;
    std::string quantization = "fp16";
    size_t maxBatchSize = 8;
    size_t maxContextLength = 2048;
    bool useLibTorch = false;
    std::string configPath;
    std::string logLevel = "info";
    std::string logFile;
    
    // 解析命令行参数
    static struct option long_options[] = {
        {"model-path", required_argument, 0, 'm'},
        {"port", required_argument, 0, 'p'},
        {"host", required_argument, 0, 'h'},
        {"quantization", required_argument, 0, 'q'},
        {"max-batch-size", required_argument, 0, 'b'},
        {"max-context-length", required_argument, 0, 'c'},
        {"use-libtorch", no_argument, 0, 'l'},
        {"config", required_argument, 0, 'f'},
        {"log-level", required_argument, 0, 'L'},
        {"log-file", required_argument, 0, 'F'},
        {"help", no_argument, 0, '?'},
        {0, 0, 0, 0}
    };
    
    int opt;
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "m:p:h:q:b:c:lf:L:F:?", 
                              long_options, &option_index)) != -1) {
        switch (opt) {
            case 'm':
                modelPath = optarg;
                break;
            case 'p':
                port = std::atoi(optarg);
                break;
            case 'h':
                host = optarg;
                break;
            case 'q':
                quantization = optarg;
                break;
            case 'b':
                maxBatchSize = std::atoi(optarg);
                break;
            case 'c':
                maxContextLength = std::atoi(optarg);
                break;
            case 'l':
                useLibTorch = true;
                break;
            case 'f':
                configPath = optarg;
                break;
            case 'L':
                logLevel = optarg;
                break;
            case 'F':
                logFile = optarg;
                break;
            case '?':
                printUsage(argv[0]);
                return 0;
            default:
                printUsage(argv[0]);
                return 1;
        }
    }
    
    // 设置日志级别
    if (logLevel == "trace") {
        cllm::Logger::instance().setLevel(spdlog::level::trace);
    } else if (logLevel == "debug") {
        cllm::Logger::instance().setLevel(spdlog::level::debug);
    } else if (logLevel == "info") {
        cllm::Logger::instance().setLevel(spdlog::level::info);
    } else if (logLevel == "warn") {
        cllm::Logger::instance().setLevel(spdlog::level::warn);
    } else if (logLevel == "error") {
        cllm::Logger::instance().setLevel(spdlog::level::err);
    }
    
    // 设置日志文件
    if (!logFile.empty()) {
        cllm::Logger::instance().addFileSink(logFile);
        CLLM_INFO("Logging to file: {}", logFile);
    }
    
    // 检查必需参数
    if (modelPath.empty()) {
        CLLM_ERROR("Model path is required. Use --model-path to specify");
        printUsage(argv[0]);
        return 1;
    }
    
    printBanner();
    
    CLLM_INFO("========================================");
    CLLM_INFO("cLLM - High-Performance LLM Inference Engine");
    CLLM_INFO("========================================");
    CLLM_INFO("Starting cLLM Server...");
    CLLM_INFO("Configuration:");
    CLLM_INFO("  - Model Path: {}", modelPath);
    CLLM_INFO("  - Host: {}", host);
    CLLM_INFO("  - Port: {}", port);
    CLLM_INFO("  - Quantization: {}", quantization);
    CLLM_INFO("  - Max Batch Size: {}", maxBatchSize);
    CLLM_INFO("  - Max Context Length: {}", maxContextLength);
    CLLM_INFO("  - Backend: {}", useLibTorch ? "LibTorch" : "Kylin");
    CLLM_INFO("  - Log Level: {}", logLevel);
    
    try {
        // 加载配置文件（如果提供）
        if (!configPath.empty()) {
            CLLM_INFO("Loading config from: {}", configPath);
            cllm::Config::instance().load(configPath);
        }
        
        // 注册信号处理器
        signal(SIGINT, signalHandler);
        signal(SIGTERM, signalHandler);
        
        // 初始化 Asio 处理器（满足技术栈要求）
        CLLM_INFO("Initializing Asio handler...");
        cllm::AsioHandler asioHandler;
        CLLM_INFO("Asio thread pool size: {}", asioHandler.getThreadPoolSize());
        
        // 提交一个异步任务作为示例
        asioHandler.postTask([]() {
            CLLM_DEBUG("Asio async task executed successfully");
        });
        
        // 初始化模型执行器
        CLLM_INFO("Initializing model executor...");
        g_modelExecutor = std::make_unique<cllm::ModelExecutor>(
            modelPath,
            quantization,
            true,  // enableSIMD
            useLibTorch
        );
        
        // 加载模型
        CLLM_INFO("Loading model...");
        g_modelExecutor->loadModel();
        CLLM_INFO("Model loaded successfully");
        
        // 初始化分词器
        CLLM_INFO("Initializing tokenizer...");
        std::string tokenizerPath = modelPath + "/tokenizer.model";
        g_tokenizer = std::make_unique<cllm::Tokenizer>(tokenizerPath);
        CLLM_INFO("Tokenizer initialized");
        CLLM_INFO("  - Vocab size: {}", g_tokenizer->getVocabSize());
        
        // 初始化调度器
        CLLM_INFO("Initializing scheduler...");
        g_scheduler = std::make_unique<cllm::Scheduler>(
            g_modelExecutor.get(),
            maxBatchSize,
            maxContextLength
        );
        
        // 启动调度器
        CLLM_INFO("Starting scheduler...");
        g_scheduler->start();
        CLLM_INFO("Scheduler started");
        
        // 创建 HTTP 处理器
        CLLM_INFO("Setting up HTTP endpoints...");
        auto httpHandler = std::make_unique<cllm::HttpHandler>();
        
        // 注册端点
        auto healthEndpoint = std::make_unique<cllm::HealthEndpoint>();
        httpHandler->get("/health", [endpoint = healthEndpoint.get()](const cllm::HttpRequest& req) {
            return endpoint->handle(req);
        });
        
        auto generateEndpoint = std::make_unique<cllm::GenerateEndpoint>(
            g_scheduler.get(),
            g_tokenizer.get()
        );
        httpHandler->post("/generate", [endpoint = generateEndpoint.get()](const cllm::HttpRequest& req) {
            return endpoint->handle(req);
        });
        
        httpHandler->post("/generate_stream", [endpoint = generateEndpoint.get()](const cllm::HttpRequest& req) {
            return endpoint->handle(req);
        });
        
        auto encodeEndpoint = std::make_unique<cllm::EncodeEndpoint>(g_tokenizer.get());
        httpHandler->post("/encode", [endpoint = encodeEndpoint.get()](const cllm::HttpRequest& req) {
            return endpoint->handle(req);
        });
        
        CLLM_INFO("Registered endpoints:");
        CLLM_INFO("  - GET  /health");
        CLLM_INFO("  - POST /generate");
        CLLM_INFO("  - POST /generate_stream");
        CLLM_INFO("  - POST /encode");
        
        // 初始化并启动 Drogon 服务器
        CLLM_INFO("Initializing Drogon HTTP server...");
        cllm::DrogonServer::init(host, port, httpHandler.get());
        
        CLLM_INFO("========================================");
        CLLM_INFO("✓ cLLM Server is ready!");
        CLLM_INFO("Listening on http://{}:{}", host, port);
        CLLM_INFO("Press Ctrl+C to stop the server");
        CLLM_INFO("========================================");
        
        // 启动服务器（阻塞）
        cllm::DrogonServer::start();
        
        // 保持端点对象生命周期
        healthEndpoint.reset();
        generateEndpoint.reset();
        encodeEndpoint.reset();
        
    } catch (const std::exception& e) {
        CLLM_ERROR("Failed to start server: {}", e.what());
        cllm::Logger::instance().flush();
        return 1;
    }
    
    cllm::Logger::instance().flush();
    return 0;
}