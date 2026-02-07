/**
 * @file kylin_test_framework.h
 * @brief Kylin Backend 测试框架核心
 * 
 * 提供模块化的测试结构，支持：
 * - 分阶段测试执行
 * - 详细的日志记录
 * - 成功/失败判断
 * - 测试报告生成
 */

#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <chrono>
#include <memory>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <algorithm>

namespace kylin_test {

// 测试状态枚举
enum class TestStatus {
    NOT_RUN,      // 未运行
    RUNNING,      // 运行中
    PASSED,       // 通过
    FAILED,       // 失败
    SKIPPED,      // 跳过
    ERROR         // 错误
};

// 日志级别
enum class LogLevel {
    DEBUG,
    INFO,
    WARN,
    ERROR,
    PASS,
    FAIL
};

// 测试结果结构
struct TestResult {
    std::string name;
    TestStatus status = TestStatus::NOT_RUN;
    std::string message;
    double duration_ms = 0.0;
    std::vector<std::string> logs;
    
    void addLog(LogLevel level, const std::string& msg) {
        std::string prefix;
        switch (level) {
            case LogLevel::DEBUG: prefix = "[DEBUG]"; break;
            case LogLevel::INFO:  prefix = "[INFO] "; break;
            case LogLevel::WARN:  prefix = "[WARN] "; break;
            case LogLevel::ERROR: prefix = "[ERROR]"; break;
            case LogLevel::PASS:  prefix = "[PASS] "; break;
            case LogLevel::FAIL:  prefix = "[FAIL] "; break;
        }
        logs.push_back(prefix + " " + msg);
    }
};

// 测试基类
class TestCase {
public:
    TestCase(const std::string& name, const std::string& description = "")
        : name_(name), description_(description) {}
    
    virtual ~TestCase() = default;
    
    // 执行测试
    TestResult run() {
        TestResult result;
        result.name = name_;
        result.status = TestStatus::RUNNING;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        try {
            log(LogLevel::INFO, "Starting test: " + name_);
            if (!description_.empty()) {
                log(LogLevel::INFO, "Description: " + description_);
            }
            
            // 执行测试逻辑
            execute();
            
            result.status = TestStatus::PASSED;
            result.message = "Test passed";
            log(LogLevel::PASS, "Test completed successfully");
            
        } catch (const std::exception& e) {
            result.status = TestStatus::FAILED;
            result.message = std::string("Exception: ") + e.what();
            log(LogLevel::FAIL, result.message);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        result.duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
        result.logs = logs_;
        
        return result;
    }
    
    // 跳过测试
    TestResult skip(const std::string& reason) {
        TestResult result;
        result.name = name_;
        result.status = TestStatus::SKIPPED;
        result.message = reason;
        log(LogLevel::WARN, "Test skipped: " + reason);
        result.logs = logs_;
        return result;
    }
    
    const std::string& getName() const { return name_; }
    const std::string& getDescription() const { return description_; }

protected:
    // 子类实现具体的测试逻辑
    virtual void execute() = 0;
    
    // 日志记录
    void log(LogLevel level, const std::string& msg) {
        logs_.push_back(formatLog(level, msg));
        if (verbose_) {
            std::cout << logs_.back() << std::endl;
        }
    }
    
    // 断言宏的辅助函数
    void assertTrue(bool condition, const std::string& msg) {
        if (!condition) {
            throw std::runtime_error("Assertion failed: " + msg);
        }
        log(LogLevel::DEBUG, "Assertion passed: " + msg);
    }
    
    void assertFalse(bool condition, const std::string& msg) {
        assertTrue(!condition, msg);
    }
    
    void assertEquals(int expected, int actual, const std::string& msg) {
        if (expected != actual) {
            throw std::runtime_error(msg + " (expected: " + std::to_string(expected) + 
                                   ", actual: " + std::to_string(actual) + ")");
        }
        log(LogLevel::DEBUG, msg + " (value: " + std::to_string(actual) + ")");
    }
    
    void assertNear(float expected, float actual, float tolerance, const std::string& msg) {
        if (std::abs(expected - actual) > tolerance) {
            throw std::runtime_error(msg + " (expected: " + std::to_string(expected) + 
                                   ", actual: " + std::to_string(actual) + 
                                   ", tolerance: " + std::to_string(tolerance) + ")");
        }
        log(LogLevel::DEBUG, msg + " (value: " + std::to_string(actual) + ")");
    }
    
    void assertNotNull(const void* ptr, const std::string& msg) {
        if (ptr == nullptr) {
            throw std::runtime_error("Null pointer: " + msg);
        }
        log(LogLevel::DEBUG, "Pointer not null: " + msg);
    }
    
    void assertValidLogits(const std::vector<float>& logits, const std::string& msg) {
        size_t nanCount = 0, infCount = 0;
        for (const auto& val : logits) {
            if (std::isnan(val)) nanCount++;
            else if (std::isinf(val)) infCount++;
        }
        
        if (nanCount > 0 || infCount > 0) {
            throw std::runtime_error(msg + " - Invalid logits: " + 
                                   std::to_string(nanCount) + " NaN, " +
                                   std::to_string(infCount) + " Inf");
        }
        log(LogLevel::DEBUG, msg + " - All logits valid (" + std::to_string(logits.size()) + " values)");
    }
    
    void assertTokenDiversity(const std::vector<int>& tokens, int minUnique, const std::string& msg) {
        std::vector<int> uniqueTokens = tokens;
        std::sort(uniqueTokens.begin(), uniqueTokens.end());
        uniqueTokens.erase(std::unique(uniqueTokens.begin(), uniqueTokens.end()), uniqueTokens.end());
        
        if (static_cast<int>(uniqueTokens.size()) < minUnique) {
            throw std::runtime_error(msg + " - Token diversity too low: " +
                                   std::to_string(uniqueTokens.size()) + " unique tokens, " +
                                   "expected at least " + std::to_string(minUnique));
        }
        log(LogLevel::DEBUG, msg + " - Token diversity: " + 
            std::to_string(uniqueTokens.size()) + " unique tokens");
    }

public:
    static void setVerbose(bool verbose) { verbose_ = verbose; }

private:
    std::string name_;
    std::string description_;
    std::vector<std::string> logs_;
    static bool verbose_;
    
    std::string formatLog(LogLevel level, const std::string& msg) {
        std::string prefix;
        switch (level) {
            case LogLevel::DEBUG: prefix = "[DEBUG]"; break;
            case LogLevel::INFO:  prefix = "[INFO] "; break;
            case LogLevel::WARN:  prefix = "[WARN] "; break;
            case LogLevel::ERROR: prefix = "[ERROR]"; break;
            case LogLevel::PASS:  prefix = "[PASS] "; break;
            case LogLevel::FAIL:  prefix = "[FAIL] "; break;
        }
        return prefix + " " + msg;
    }
};

inline bool TestCase::verbose_ = false;

// 测试套件
class TestSuite {
public:
    TestSuite(const std::string& name) : name_(name) {}
    
    void addTest(std::shared_ptr<TestCase> test) {
        tests_.push_back(test);
    }
    
    std::vector<TestResult> runAll() {
        std::vector<TestResult> results;
        std::cout << "\n========================================" << std::endl;
        std::cout << "Running Test Suite: " << name_ << std::endl;
        std::cout << "========================================" << std::endl;
        
        for (auto& test : tests_) {
            auto result = test->run();
            results.push_back(result);
            printResult(result);
        }
        
        printSummary(results);
        return results;
    }
    
    std::vector<TestResult> runByName(const std::string& testName) {
        std::vector<TestResult> results;
        for (auto& test : tests_) {
            if (test->getName() == testName) {
                auto result = test->run();
                results.push_back(result);
                printResult(result);
            }
        }
        return results;
    }

private:
    std::string name_;
    std::vector<std::shared_ptr<TestCase>> tests_;
    
    void printResult(const TestResult& result) {
        std::cout << "\n[Test: " << result.name << "]" << std::endl;
        std::cout << "  Status: " << statusToString(result.status) << std::endl;
        std::cout << "  Duration: " << std::fixed << std::setprecision(2) 
                  << result.duration_ms << " ms" << std::endl;
        if (!result.message.empty()) {
            std::cout << "  Message: " << result.message << std::endl;
        }
        for (const auto& log : result.logs) {
            std::cout << "  " << log << std::endl;
        }
    }
    
    void printSummary(const std::vector<TestResult>& results) {
        int passed = 0, failed = 0, skipped = 0, errors = 0;
        double totalDuration = 0.0;
        
        for (const auto& r : results) {
            switch (r.status) {
                case TestStatus::PASSED: passed++; break;
                case TestStatus::FAILED: failed++; break;
                case TestStatus::SKIPPED: skipped++; break;
                case TestStatus::ERROR: errors++; break;
                default: break;
            }
            totalDuration += r.duration_ms;
        }
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "Test Summary" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Total: " << results.size() << std::endl;
        std::cout << "Passed: " << passed << std::endl;
        std::cout << "Failed: " << failed << std::endl;
        std::cout << "Skipped: " << skipped << std::endl;
        std::cout << "Errors: " << errors << std::endl;
        std::cout << "Total Duration: " << std::fixed << std::setprecision(2) 
                  << totalDuration << " ms" << std::endl;
        std::cout << "========================================" << std::endl;
    }
    
    std::string statusToString(TestStatus status) {
        switch (status) {
            case TestStatus::NOT_RUN: return "NOT RUN";
            case TestStatus::RUNNING: return "RUNNING";
            case TestStatus::PASSED: return "PASSED ✓";
            case TestStatus::FAILED: return "FAILED ✗";
            case TestStatus::SKIPPED: return "SKIPPED ○";
            case TestStatus::ERROR: return "ERROR !";
            default: return "UNKNOWN";
        }
    }
};

// 测试报告生成器
class TestReport {
public:
    static void generateHTML(const std::string& filename, 
                            const std::vector<TestResult>& results,
                            const std::string& suiteName) {
        std::ofstream file(filename);
        file << "<!DOCTYPE html>\n<html>\n<head>\n";
        file << "<title>Kylin Backend Test Report</title>\n";
        file << "<style>\n";
        file << "body { font-family: Arial, sans-serif; margin: 20px; }\n";
        file << ".passed { color: green; }\n";
        file << ".failed { color: red; }\n";
        file << ".skipped { color: orange; }\n";
        file << "table { border-collapse: collapse; width: 100%; }\n";
        file << "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n";
        file << "th { background-color: #4CAF50; color: white; }\n";
        file << "tr:nth-child(even) { background-color: #f2f2f2; }\n";
        file << "</style>\n</head>\n<body>\n";
        
        file << "<h1>Kylin Backend Test Report</h1>\n";
        file << "<p>Suite: " << suiteName << "</p>\n";
        file << "<p>Generated: " << getCurrentTime() << "</p>\n";
        
        // 汇总
        int passed = 0, failed = 0, skipped = 0;
        for (const auto& r : results) {
            switch (r.status) {
                case TestStatus::PASSED: passed++; break;
                case TestStatus::FAILED: failed++; break;
                case TestStatus::SKIPPED: skipped++; break;
                default: break;
            }
        }
        
        file << "<h2>Summary</h2>\n";
        file << "<p>Total: " << results.size() << " | ";
        file << "<span class='passed'>Passed: " << passed << "</span> | ";
        file << "<span class='failed'>Failed: " << failed << "</span> | ";
        file << "<span class='skipped'>Skipped: " << skipped << "</span></p>\n";
        
        // 详细结果
        file << "<h2>Detailed Results</h2>\n";
        file << "<table>\n";
        file << "<tr><th>Test Name</th><th>Status</th><th>Duration (ms)</th><th>Message</th></tr>\n";
        
        for (const auto& r : results) {
            std::string statusClass;
            switch (r.status) {
                case TestStatus::PASSED: statusClass = "passed"; break;
                case TestStatus::FAILED: statusClass = "failed"; break;
                case TestStatus::SKIPPED: statusClass = "skipped"; break;
                default: statusClass = "";
            }
            
            file << "<tr>";
            file << "<td>" << r.name << "</td>";
            file << "<td class='" << statusClass << "'>" << statusToString(r.status) << "</td>";
            file << "<td>" << std::fixed << std::setprecision(2) << r.duration_ms << "</td>";
            file << "<td>" << r.message << "</td>";
            file << "</tr>\n";
        }
        
        file << "</table>\n</body>\n</html>\n";
    }

private:
    static std::string getCurrentTime() {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
        return ss.str();
    }
    
    static std::string statusToString(TestStatus status) {
        switch (status) {
            case TestStatus::PASSED: return "PASSED";
            case TestStatus::FAILED: return "FAILED";
            case TestStatus::SKIPPED: return "SKIPPED";
            case TestStatus::ERROR: return "ERROR";
            default: return "UNKNOWN";
        }
    }
};

} // namespace kylin_test
