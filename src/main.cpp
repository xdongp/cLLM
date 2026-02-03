/**
 * @file main.cpp
 * @brief cLLM 服务器主入口
 * @author cLLM Team
 * @date 2026-01-10
 */

#include "cllm/http/http_server.h"

#include <signal.h>
#include <getopt.h>

#include <memory>
#include <string>
#include <filesystem>
#include <optional>
#include <vector>
#include <thread>
#include <chrono>

#include "cllm/http/handler.h"
#include "cllm/http/health_endpoint.h"
#include "cllm/http/generate_endpoint.h"
#include "cllm/http/encode_endpoint.h"
#include "cllm/http/benchmark_endpoint.h"
#include "cllm/http/model_info_endpoint.h"
#include "cllm/scheduler/scheduler.h"
#include "cllm/model/executor.h"
#include "cllm/tokenizer/manager.h"
#include "cllm/common/config.h"
#include "cllm/common/logger.h"
#include "cllm/common/asio_handler.h"
#include "cllm/http/response_builder.h"
#include "cllm/common/json.h"

// 全局变量用于信号处理
static std::unique_ptr<cllm::Scheduler> g_scheduler;
static std::unique_ptr<cllm::ModelExecutor> g_modelExecutor;
static std::unique_ptr<cllm::TokenizerManager> g_tokenizerManager;

// API Endpoint 路径常量（硬编码，简化配置）
namespace ApiEndpoints {
    constexpr const char* ROOT = "/";
    constexpr const char* HEALTH = "/health";
    constexpr const char* GENERATE = "/generate";
    constexpr const char* GENERATE_STREAM = "/generate_stream";
    constexpr const char* ENCODE = "/encode";
    constexpr const char* BENCHMARK = "/benchmark";
    constexpr const char* MODEL_INFO = "/model/info";
}

/**
 * @brief 信号处理函数
 * @param signal 信号编号
 */
void signalHandler(int signal) {
    CLLM_INFO("Received signal %d, shutting down gracefully...", signal);
    
    // 停止调度器
    if (g_scheduler) {
        CLLM_INFO("Stopping scheduler...");
        g_scheduler->stop();
    }
    
    // 停止服务器
    CLLM_INFO("Stopping HTTP server...");
    cllm::HttpServer::stop();
    
    CLLM_INFO("Shutdown complete");
    cllm::Logger::instance().flush();
    exit(0);
}

/**
 * @brief 打印使用说明
 * @param programName 程序名称
 */
void printUsage(const char* programName) {
    CLLM_INFO("Usage: %s [options]", programName);
    CLLM_INFO("");
    CLLM_INFO("Options:");
    CLLM_INFO("  --model-path PATH        Path to model directory (required)");
    CLLM_INFO("  --port PORT              Server port (default: 8080)");
    CLLM_INFO("  --host HOST              Server host (default: 0.0.0.0)");
    CLLM_INFO("  --quantization TYPE      Quantization type: fp16, int8, int4, q4_k_m, q5_k_m (default: fp16)");
    CLLM_INFO("  --max-batch-size SIZE    Maximum batch size (default: 8)");
    CLLM_INFO("  --max-context-length LEN Maximum context length (default: 2048)");
    CLLM_INFO("  --use-libtorch           Use LibTorch backend (default: false, use Kylin)");
    CLLM_INFO("  --config PATH            Path to config file (optional)");
    CLLM_INFO("  --log-level LEVEL        Log level: trace, debug, info, warn, error (default: info)");
    CLLM_INFO("  --log-file PATH          Log file path (optional)");
    CLLM_INFO("  --help                   Show this help message");
    CLLM_INFO("");
    CLLM_INFO("Examples:");
    CLLM_INFO("  %s --model-path /path/to/model", programName);
    CLLM_INFO("  %s --model-path /path/to/model --port 9000 --use-libtorch", programName);
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
    
    // 配置参数（先收集 CLI 覆盖项；Config 文件加载后再计算最终值）
    std::optional<std::string> modelPathOpt;
    std::optional<std::string> hostOpt;
    std::optional<int> portOpt;
    std::optional<std::string> quantizationOpt;
    std::optional<size_t> maxBatchSizeOpt;
    std::optional<size_t> maxContextLengthOpt;
    bool useLibTorchOpt = false;
    bool useLibTorchOptSet = false;
    std::optional<std::string> logLevelOpt;
    std::optional<std::string> logFileOpt;
    std::optional<std::string> backendTypeOpt;

    std::string configPath;
    
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
                modelPathOpt = optarg;
                break;
            case 'p':
                portOpt = std::atoi(optarg);
                break;
            case 'h':
                hostOpt = optarg;
                break;
            case 'q':
                quantizationOpt = optarg;
                break;
            case 'b':
                maxBatchSizeOpt = static_cast<size_t>(std::atoi(optarg));
                break;
            case 'c':
                maxContextLengthOpt = static_cast<size_t>(std::atoi(optarg));
                break;
            case 'l':
                useLibTorchOpt = true;
                useLibTorchOptSet = true;
                break;
            case 'f':
                configPath = optarg;
                break;
            case 'L':
                logLevelOpt = optarg;
                break;
            case 'F':
                logFileOpt = optarg;
                break;
            case '?':
                printUsage(argv[0]);
                return 0;
            default:
                printUsage(argv[0]);
                return 1;
        }
    }
    
    printBanner();

    try {
        // 加载配置文件：优先使用 --config，否则自动加载 config/config.yaml
        {
            namespace fs = std::filesystem;
            std::string selectedConfigPath;

            if (!configPath.empty()) {
                selectedConfigPath = configPath;
            } else {
                const fs::path candidates[] = {
                    fs::path("config") / "config.yaml",
                    fs::path("../config") / "config.yaml",
                    fs::path("../../config") / "config.yaml"
                };

                for (const auto& p : candidates) {
                    if (fs::exists(p)) {
                        selectedConfigPath = p.string();
                        break;
                    }
                }
            }

            if (selectedConfigPath.empty() || !fs::exists(selectedConfigPath.c_str())) {
                throw std::runtime_error("Config file not found. Please pass --config or ensure config/config.yaml exists.");
            }

            CLLM_INFO("Loading config from: %s", selectedConfigPath.c_str());
            cllm::Config::instance().load(selectedConfigPath.c_str());
        }

        // 计算最终配置（Config + CLI 覆盖）
        std::string modelPath = modelPathOpt.value_or(cllm::Config::instance().serverModelPath());
        std::string host = hostOpt.value_or(cllm::Config::instance().serverHost());
        int port = portOpt.value_or(cllm::Config::instance().serverPort());
        std::string quantization = quantizationOpt.value_or(cllm::Config::instance().serverQuantization());
        size_t maxBatchSize = maxBatchSizeOpt.value_or(static_cast<size_t>(cllm::Config::instance().serverMaxBatchSize()));
        size_t maxContextLength = maxContextLengthOpt.value_or(static_cast<size_t>(cllm::Config::instance().serverMaxContextLength()));
        std::string backendType = backendTypeOpt.value_or(cllm::Config::instance().backendType());
        bool useLibTorch = useLibTorchOptSet ? useLibTorchOpt : (backendType == "libtorch" || backendType == "LibTorch" || cllm::Config::instance().serverUseLibTorch());
        if (useLibTorch) {
            backendType = "libtorch";
        }
        std::string logLevel = logLevelOpt.value_or(cllm::Config::instance().loggingLevel());
        std::string logFile = logFileOpt.value_or(cllm::Config::instance().loggingFile());

        // 设置日志级别（使用最终值）
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
            cllm::Logger::instance().addFileSink(logFile.c_str());
            CLLM_INFO("Logging to file: %s", logFile.c_str());
        }

        // 检查必需参数（允许从配置提供 model.path）
        // 允许空模型路径用于测试目的（将使用占位权重）
        if (modelPath == "/path/to/model") {
            CLLM_ERROR("Model path is required. Use --model-path or set model.path in config");
            printUsage(argv[0]);
            return 1;
        }
        
        if (modelPath.empty()) {
            CLLM_INFO("Using empty model path - will use placeholder weights for testing");
        }

        CLLM_INFO("========================================");
        CLLM_INFO("cLLM - High-Performance LLM Inference Engine");
        CLLM_INFO("========================================");
        CLLM_INFO("Starting cLLM Server...");
        CLLM_INFO("Configuration:");
        CLLM_INFO("  - Model Path: %s", modelPath.c_str());
        CLLM_INFO("  - Host: %s", host.c_str());
        CLLM_INFO("  - Port: %d", port);
        CLLM_INFO("  - Quantization: %s", quantization.c_str());
        CLLM_INFO("  - Max Batch Size: %zu", maxBatchSize);
        CLLM_INFO("  - Max Context Length: %zu", maxContextLength);
        CLLM_INFO("  - Backend: %s", (backendType.empty() ? (useLibTorch ? "LibTorch" : "Auto") : backendType.c_str()));
        CLLM_INFO("  - Log Level: %s", logLevel.c_str());

        // 注册信号处理器
        signal(SIGINT, signalHandler);
        signal(SIGTERM, signalHandler);
        
        // 初始化 Asio 处理器（满足技术栈要求）
        CLLM_INFO("Initializing Asio handler...");
        cllm::AsioHandler asioHandler;
        CLLM_INFO("Asio thread pool size: %zu", asioHandler.getThreadPoolSize());
        
        // 提交一个异步任务作为示例
        asioHandler.postTask([]() {
            CLLM_DEBUG("Asio async task executed successfully");
        });
        
        // 初始化模型执行器
        CLLM_INFO("Initializing model executor...");

        namespace fs = std::filesystem;

        std::string tokenizerModelDir;
        std::string backendModelPath;

        if (modelPath.empty()) {
            // Empty model path case - use placeholder weights and default tokenizer
            CLLM_INFO("Empty model path provided - using placeholder weights and default tokenizer");
            tokenizerModelDir = "";
            backendModelPath = "";
        } else {
            // 1) Check if model path is a directory
            const bool isDir = fs::is_directory(modelPath);

            // 2) 解析 tokenizer 目录（优先 HuggingFace tokenizer.json）
            auto resolveTokenizerDir = [](const fs::path& baseDir) -> std::string {
                if (fs::exists(baseDir / "tokenizer.json")) {
                    return baseDir.string();
                }

                std::vector<fs::path> matches;
                for (const auto& ent : fs::directory_iterator(baseDir)) {
                    if (!ent.is_directory()) {
                        continue;
                    }
                    const fs::path cand = ent.path() / "tokenizer.json";
                    if (fs::exists(cand)) {
                        matches.push_back(ent.path());
                    }
                }

                if (matches.size() == 1) {
                    return matches.front().string();
                }

                // 回退：用 baseDir（TokenizerManager 会再尝试其他方式/或给出错误）
                return baseDir.string();
            };

            if (isDir) {
                tokenizerModelDir = resolveTokenizerDir(fs::path(modelPath));
            } else {
                fs::path file(modelPath);
                tokenizerModelDir = resolveTokenizerDir(file.parent_path());
            }

            // 3) 解析后端权重文件路径
            auto listFilesWithExt = [](const fs::path& dir, const std::string& ext) -> std::vector<fs::path> {
                std::vector<fs::path> out;
                for (const auto& ent : fs::directory_iterator(dir)) {
                    if (!ent.is_regular_file()) {
                        continue;
                    }
                    const fs::path p = ent.path();
                    if (p.extension() == ext) {
                        out.push_back(p);
                    }
                }
                return out;
            };

            if (isDir) {
                fs::path dir(modelPath);

                // 检测是否为 HuggingFace 模型目录（优先级最高）
                // HuggingFace 目录包含 config.json 和 model.safetensors
                bool isHFModelDir = false;
                if (backendType == "kylin" || backendType == "Kylin") {
                    fs::path configJson = dir / "config.json";
                    fs::path safetensors = dir / "model.safetensors";
                    if (fs::exists(configJson) && fs::exists(safetensors)) {
                        isHFModelDir = true;
                        backendModelPath = dir.string();  // 使用目录路径
                        CLLM_INFO("Detected HuggingFace model directory: %s", backendModelPath.c_str());
                    }
                }

                if (!useLibTorch && !isHFModelDir) {
                    auto ggufs = listFilesWithExt(dir, ".gguf");
                    if (ggufs.size() == 1) {
                        backendModelPath = ggufs.front().string();
                    } else if (ggufs.size() > 1) {
                        std::string msg = "Multiple .gguf files found. Pass a .gguf path directly via --model-path. Candidates: ";
                        for (const auto& p : ggufs) {
                            msg += p.filename().string();
                            msg += " ";
                        }
                        throw std::runtime_error(msg);
                    }
                }

                if (useLibTorch) {
                    // LibTorch 需要 TorchScript .pt
                    auto pts = listFilesWithExt(dir, ".pt");
                    if (pts.size() == 1) {
                        backendModelPath = pts.front().string();
                    } else {
                        std::string msg = "LibTorch backend requires exactly one .pt in directory (or pass a .pt path). Found: ";
                        msg += std::to_string(pts.size());
                        if (!pts.empty()) {
                            msg += " (";
                            for (const auto& p : pts) {
                                msg += p.filename().string();
                                msg += " ";
                            }
                            msg += ")";
                        }
                        throw std::runtime_error(msg);
                    }
                } else if (!backendModelPath.empty()) {
                    // 已选择 .gguf 或 HuggingFace 目录
                } else if (!isHFModelDir) {
                    // Kylin 需要 .bin（仅在非 HF 模型目录时检测）
                    auto bins = listFilesWithExt(dir, ".bin");
                    if (bins.empty()) {
                        throw std::runtime_error("Kylin backend requires .bin model file. No .bin found in: " + dir.string());
                    }

                    // 若目录下只有一个 .bin，直接用；否则按 quantization 关键词筛选
                    std::vector<fs::path> filtered;
                    if (bins.size() == 1) {
                        filtered = bins;
                    } else {
                        for (const auto& p : bins) {
                            const std::string name = p.filename().string();
                            if (quantization == "fp16" && name.find("fp16") != std::string::npos) {
                                filtered.push_back(p);
                            } else if (quantization == "fp32" && name.find("fp32") != std::string::npos) {
                                filtered.push_back(p);
                            } else if (quantization == "int8" && name.find("int8") != std::string::npos) {
                                filtered.push_back(p);
                            } else if (quantization == "int4" && name.find("int4") != std::string::npos) {
                                filtered.push_back(p);
                            }
                        }
                    }

                    if (filtered.size() == 1) {
                        backendModelPath = filtered.front().string();
                    } else {
                        std::string msg = "Could not uniquely select .bin for quantization=" + quantization +
                                          ". Pass a .bin path directly via --model-path. Candidates: ";
                        for (const auto& p : bins) {
                            msg += p.filename().string();
                            msg += " ";
                        }
                        throw std::runtime_error(msg);
                    }
                }
            } else {
                backendModelPath = modelPath;
            }
        }

        if (!backendModelPath.empty()) {
            namespace fs = std::filesystem;
            if (fs::path(backendModelPath).extension() == ".gguf") {
                tokenizerModelDir = backendModelPath;
                if (backendType.empty()) {
                    backendType = "llama_cpp";
                }
            }
        }

        CLLM_INFO("Resolved paths:");
        CLLM_INFO("  - Backend model file: %s", backendModelPath.c_str());
        CLLM_INFO("  - Tokenizer dir: %s", tokenizerModelDir.c_str());
        CLLM_INFO("  - Backend type: %s", backendType.empty() ? "auto" : backendType.c_str());

        cllm::ModelConfig initialConfig;
        initialConfig.maxSequenceLength = maxContextLength;
        initialConfig.ropeNctxOrig = maxContextLength;
        initialConfig.llamaBatchSize = static_cast<size_t>(cllm::Config::instance().backendLlamaCppBatchSize());
        initialConfig.llamaNumThreads = cllm::Config::instance().backendLlamaCppNumThreads();
        initialConfig.llamaGpuLayers = cllm::Config::instance().backendLlamaCppGpuLayers();
        initialConfig.llamaUseMmap = cllm::Config::instance().backendLlamaCppUseMmap();
        initialConfig.llamaUseMlock = cllm::Config::instance().backendLlamaCppUseMlock();

        g_modelExecutor = std::make_unique<cllm::ModelExecutor>(
            backendModelPath,
            quantization,
            true,  // enableSIMD
            useLibTorch,
            backendType,
            &initialConfig
        );

        // 加载模型（实际权重加载由 InferenceEngine 后端负责，这里主要做 warmup / 标记）
        CLLM_INFO("Loading model...");
        g_modelExecutor->loadModel();
        CLLM_INFO("Model loaded successfully");

        // 初始化分词器（TokenizerManager 会自动选择 HFTokenizer 或 NativeTokenizer）
        CLLM_INFO("Initializing tokenizer...");
        g_tokenizerManager = std::make_unique<cllm::TokenizerManager>(tokenizerModelDir, g_modelExecutor.get());
        cllm::ITokenizer* tokenizer = g_tokenizerManager->getTokenizer();
        CLLM_INFO("Tokenizer initialized");
        CLLM_INFO("  - Vocab size: %zu", tokenizer->getVocabSize());
        
        // 将 tokenizer 的 vocab_size 设置到 ModelExecutor 的 config 中
        // 这样在采样时可以使用正确的 tokenizer vocab_size 来限制 logits 范围
        size_t tokenizerVocabSize = tokenizer->getVocabSize();
        g_modelExecutor->setTokenizerVocabSize(tokenizerVocabSize);
        CLLM_INFO("Model config updated with tokenizer vocab_size: %zu", tokenizerVocabSize);
        
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
        
        // 根路径：显示所有可用 endpoints
        httpHandler->get(ApiEndpoints::ROOT, [](const cllm::HttpRequest& req) {
            nlohmann::json response;
            response["name"] = "cLLM Server";
            response["version"] = "1.0.0";
            response["endpoints"] = {
                {{"method", "GET"},  {"path", ApiEndpoints::ROOT},           {"description", "显示所有可用端点"}},
                {{"method", "GET"},  {"path", ApiEndpoints::HEALTH},         {"description", "健康检查"}},
                {{"method", "POST"}, {"path", ApiEndpoints::GENERATE},       {"description", "文本生成（非流式）"}},
                {{"method", "POST"}, {"path", ApiEndpoints::GENERATE_STREAM},{"description", "文本生成（流式）"}},
                {{"method", "POST"}, {"path", ApiEndpoints::ENCODE},         {"description", "文本编码（tokenize）"}},
                {{"method", "POST"}, {"path", ApiEndpoints::BENCHMARK},      {"description", "性能基准测试"}},
                {{"method", "GET"},  {"path", ApiEndpoints::MODEL_INFO},     {"description", "模型信息"}}
            };
            return cllm::ResponseBuilder::success(response);
        });
        
        // 健康检查端点
        auto healthEndpoint = std::make_unique<cllm::HealthEndpoint>();
        httpHandler->get(ApiEndpoints::HEALTH, [endpoint = healthEndpoint.get()](const cllm::HttpRequest& req) {
            return endpoint->handle(req);
        });
        
        // 文本生成端点（非流式）
        auto generateEndpoint = std::make_unique<cllm::GenerateEndpoint>(
            g_scheduler.get(),
            tokenizer
        );
        httpHandler->post(ApiEndpoints::GENERATE, [endpoint = generateEndpoint.get()](const cllm::HttpRequest& req) {
            return endpoint->handle(req);
        });
        
        // 流式端点：使用真流式处理（边生成边发送）
        httpHandler->postStreaming(ApiEndpoints::GENERATE_STREAM, 
            [endpoint = generateEndpoint.get()](const cllm::HttpRequest& req, cllm::StreamingWriteCallback writeCallback) {
                endpoint->handleStreamingCallback(req, writeCallback);
            });
        
        // 文本编码端点
        auto encodeEndpoint = std::make_unique<cllm::EncodeEndpoint>(tokenizer);
        httpHandler->post(ApiEndpoints::ENCODE, [endpoint = encodeEndpoint.get()](const cllm::HttpRequest& req) {
            return endpoint->handle(req);
        });
        
        // 性能基准测试端点
        auto benchmarkEndpoint = std::make_unique<cllm::BenchmarkEndpoint>(
            g_modelExecutor.get(), tokenizer, 8, 2048);
        httpHandler->post(ApiEndpoints::BENCHMARK, [endpoint = benchmarkEndpoint.get()](const cllm::HttpRequest& req) {
            return endpoint->handle(req);
        });
        
        // 模型信息端点
        auto modelInfoEndpoint = std::make_unique<cllm::ModelInfoEndpoint>(
            g_modelExecutor.get(), modelPath);
        httpHandler->get(ApiEndpoints::MODEL_INFO, [endpoint = modelInfoEndpoint.get()](const cllm::HttpRequest& req) {
            return endpoint->handle(req);
        });
        
        CLLM_INFO("Registered endpoints:");
        CLLM_INFO("  - GET  %s", ApiEndpoints::ROOT);
        CLLM_INFO("  - GET  %s", ApiEndpoints::HEALTH);
        CLLM_INFO("  - POST %s", ApiEndpoints::GENERATE);
        CLLM_INFO("  - POST %s", ApiEndpoints::GENERATE_STREAM);
        CLLM_INFO("  - POST %s", ApiEndpoints::ENCODE);
        CLLM_INFO("  - POST %s", ApiEndpoints::BENCHMARK);
        CLLM_INFO("  - GET  %s", ApiEndpoints::MODEL_INFO);
        
        // 初始化并启动 HTTP 服务器（自研高性能服务器）
        CLLM_INFO("Initializing HTTP server...");
        cllm::HttpServer::init(host, port, httpHandler.get());
        
        CLLM_INFO("========================================");
        CLLM_INFO("✓ cLLM Server is ready!");
        CLLM_INFO("Listening on http://%s:%d", host.c_str(), port);
        CLLM_INFO("Press Ctrl+C to stop the server");
        CLLM_INFO("========================================");
        
        // Start server (启动后台线程)
        cllm::HttpServer::start();
        
        // 保持服务器运行（主线程等待）
        while (cllm::HttpServer::isRunning()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        
        // Keep endpoint objects alive
        healthEndpoint.reset();
        generateEndpoint.reset();
        encodeEndpoint.reset();
        benchmarkEndpoint.reset();
    } catch (const std::exception& e) {
        CLLM_ERROR("Failed to start server: %s", e.what());
        cllm::Logger::instance().flush();
        return 1;
    }
    
    cllm::Logger::instance().flush();
    return 0;
}