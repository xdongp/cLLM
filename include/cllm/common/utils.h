/**
 * @file utils.h
 * @brief 通用工具函数，提供字符串处理、文件操作等功能
 * @author cLLM Team
 * @date 2024-01-01
 */

#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <unordered_map>

namespace cllm {

/**
 * @brief 获取当前时间戳字符串
 * @return 格式化的时间戳字符串
 */
std::string getCurrentTimestamp();

/**
 * @brief 格式化字节大小
 * @param bytes 字节数
 * @return 格式化后的字符串（例："1.5 MB"）
 */
std::string formatSize(size_t bytes);

/**
 * @brief 格式化时间间隔
 * @param duration 时间间隔（毫秒）
 * @return 格式化后的字符串（例："1.5s"）
 */
std::string formatDuration(std::chrono::milliseconds duration);

/**
 * @brief 分割字符串
 * @param str 要分割的字符串
 * @param delimiter 分隔符
 * @return 分割后的字符串数组
 */
std::vector<std::string> splitString(const std::string& str, char delimiter);

/**
 * @brief 去除字符串首尾空白
 * @param str 要处理的字符串
 * @return 去除空白后的字符串
 */
std::string trimString(const std::string& str);

/**
 * @brief 检查文件是否存在
 * @param path 文件路径
 * @return true 如果文件存在，false 否则
 */
bool fileExists(const std::string& path);

/**
 * @brief 读取文件内容
 * @param path 文件路径
 * @return 文件内容
 * @throws std::runtime_error 如果文件不存在或读取失败
 */
std::string readFile(const std::string& path);

/**
 * @brief 写入文件内容
 * @param path 文件路径
 * @param content 要写入的内容
 * @throws std::runtime_error 如果写入失败
 */
void writeFile(const std::string& path, const std::string& content);

/**
 * @brief 解析JSON文件
 * @param path JSON文件路径
 * @return 解析后的JSON对象
 * @throws std::runtime_error 如果文件不存在或解析失败
 */
std::unordered_map<std::string, std::string> parseJsonFile(const std::string& path);

}  // namespace cllm
