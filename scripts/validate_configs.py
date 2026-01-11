#!/usr/bin/env python3
"""
配置文件验证脚本
验证cLLM配置文件的一致性、合法性和安全性
"""

import yaml
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

# 颜色输出
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_error(msg: str):
    print(f"{Colors.RED}✗ {msg}{Colors.END}")

def print_warning(msg: str):
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.END}")

def print_success(msg: str):
    print(f"{Colors.GREEN}✓ {msg}{Colors.END}")

def print_info(msg: str):
    print(f"{Colors.BLUE}ℹ {msg}{Colors.END}")

def print_header(msg: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{msg}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")

# 验证规则
class ConfigValidator:
    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.configs: Dict[str, Any] = {}
        
    def load_configs(self) -> bool:
        """加载所有配置文件"""
        config_files = [
            'model_config.yaml',
            'sampler_config.yaml',
            'scheduler_config.yaml',
            'cache_config.yaml',
            'server_config.yaml',
            'test_config.yaml',
            'production.yaml'
        ]
        
        print_header("加载配置文件")
        for config_file in config_files:
            path = self.config_dir / config_file
            if not path.exists():
                if config_file == 'production.yaml':
                    print_warning(f"配置文件不存在: {config_file} (可选)")
                else:
                    print_error(f"配置文件不存在: {config_file}")
                    self.errors.append(f"Missing config file: {config_file}")
                continue
            
            try:
                with open(path, 'r') as f:
                    self.configs[config_file] = yaml.safe_load(f)
                print_success(f"加载成功: {config_file}")
            except Exception as e:
                print_error(f"加载失败 {config_file}: {e}")
                self.errors.append(f"Failed to load {config_file}: {e}")
                return False
        
        return True
    
    def validate_vocab_size(self):
        """验证vocab_size配置"""
        print_header("验证 vocab_size")
        
        model_config = self.configs.get('model_config.yaml', {})
        test_config = self.configs.get('test_config.yaml', {})
        production_config = self.configs.get('production.yaml', {})
        
        model_vocab = model_config.get('model', {}).get('vocab_size')
        test_vocab = test_config.get('model', {}).get('vocab_size')
        prod_vocab = production_config.get('model', {}).get('vocab_size')
        
        # 期望值为Qwen3的151936
        expected_vocab = 151936
        
        if model_vocab == expected_vocab:
            print_success(f"model_config.yaml: vocab_size = {model_vocab} ✓")
        elif model_vocab == 32000:
            print_error(f"model_config.yaml: vocab_size = {model_vocab} (应该是 {expected_vocab})")
            self.errors.append(f"model_config.yaml has incorrect vocab_size: {model_vocab}")
        else:
            print_warning(f"model_config.yaml: vocab_size = {model_vocab} (非标准值)")
            self.warnings.append(f"Unusual vocab_size in model_config: {model_vocab}")
        
        if test_vocab == expected_vocab:
            print_success(f"test_config.yaml: vocab_size = {test_vocab} ✓")
        elif test_vocab:
            print_warning(f"test_config.yaml: vocab_size = {test_vocab} (与期望不符)")
        
        if prod_vocab == expected_vocab:
            print_success(f"production.yaml: vocab_size = {prod_vocab} ✓")
        elif prod_vocab:
            print_error(f"production.yaml: vocab_size = {prod_vocab} (应该是 {expected_vocab})")
    
    def validate_sampler_consistency(self):
        """验证采样器配置一致性"""
        print_header("验证采样器配置一致性")
        
        sampler_config = self.configs.get('sampler_config.yaml', {}).get('sampler', {})
        scheduler_config = self.configs.get('scheduler_config.yaml', {}).get('scheduler', {})
        test_config = self.configs.get('test_config.yaml', {}).get('sampler', {})
        
        # 检查greedy_threshold
        sampler_threshold = sampler_config.get('greedy_threshold')
        test_threshold = test_config.get('greedy_threshold')
        
        if sampler_threshold == 0.0:
            print_success(f"sampler_config.yaml: greedy_threshold = {sampler_threshold} ✓")
        else:
            print_warning(f"sampler_config.yaml: greedy_threshold = {sampler_threshold} (建议 0.0)")
            self.warnings.append(f"greedy_threshold is {sampler_threshold}, recommended 0.0")
        
        if test_threshold == 0.0:
            print_success(f"test_config.yaml: greedy_threshold = {test_threshold} ✓")
        
        # 检查冗余配置
        redundant_params = ['default_temperature', 'default_top_k', 'default_top_p']
        for param in redundant_params:
            if param in scheduler_config:
                print_warning(f"scheduler_config中存在冗余配置: {param}")
                self.warnings.append(f"Redundant parameter in scheduler_config: {param}")
    
    def validate_cache_config(self):
        """验证缓存配置"""
        print_header("验证缓存配置")
        
        cache_config = self.configs.get('cache_config.yaml', {}).get('cache', {})
        
        max_size = cache_config.get('default_max_size', 0)
        max_memory = cache_config.get('default_max_memory_mb', 0)
        enable_memory_limit = cache_config.get('enable_memory_limit', False)
        
        # 检查缓存大小
        if max_size >= 1000:
            print_success(f"default_max_size = {max_size} ✓")
        elif max_size <= 10:
            print_error(f"default_max_size = {max_size} (过小,建议 >= 1000)")
            self.errors.append(f"Cache size too small: {max_size}")
        else:
            print_warning(f"default_max_size = {max_size} (可以更大)")
        
        # 检查内存限制
        if max_memory > 0 and enable_memory_limit:
            print_success(f"内存限制已启用: {max_memory} MB ✓")
        elif max_memory == 0 and not enable_memory_limit:
            print_error("内存限制未启用 (存在OOM风险)")
            self.errors.append("Cache memory limit not enabled")
        else:
            print_warning(f"内存配置不一致: max_memory={max_memory}, enable={enable_memory_limit}")
    
    def validate_batch_size(self):
        """验证批处理大小配置"""
        print_header("验证批处理配置")
        
        scheduler_config = self.configs.get('scheduler_config.yaml', {}).get('scheduler', {})
        server_config = self.configs.get('server_config.yaml', {}).get('resources', {})
        test_config = self.configs.get('test_config.yaml', {}).get('resources', {})
        production_config = self.configs.get('production.yaml', {}).get('inference', {}).get('batch', {})
        
        scheduler_batch = scheduler_config.get('max_batch_size')
        server_batch = server_config.get('max_batch_size')
        test_batch = test_config.get('max_batch_size')
        prod_batch = production_config.get('max_size')
        
        # 检查是否有冗余定义
        if scheduler_batch and server_batch:
            if scheduler_batch == server_batch:
                print_warning(f"存在冗余配置: scheduler和server都定义了max_batch_size={scheduler_batch}")
                self.warnings.append("Redundant max_batch_size in scheduler and server")
            else:
                print_error(f"批处理大小不一致: scheduler={scheduler_batch}, server={server_batch}")
                self.errors.append(f"Inconsistent batch size: {scheduler_batch} vs {server_batch}")
        
        # 检查生产配置优化
        if prod_batch and prod_batch >= 32:
            print_success(f"production.yaml: batch.max_size = {prod_batch} (已优化) ✓")
        elif prod_batch:
            print_warning(f"production.yaml: batch.max_size = {prod_batch} (建议 >= 32)")
    
    def validate_security(self):
        """验证安全配置"""
        print_header("验证安全配置")
        
        server_config = self.configs.get('server_config.yaml', {}).get('server', {})
        production_config = self.configs.get('production.yaml', {}).get('server', {})
        
        server_host = server_config.get('host')
        prod_host = production_config.get('host')
        
        # 检查host配置
        if server_host == "0.0.0.0":
            print_warning("server_config.yaml: host = 0.0.0.0 (允许所有IP访问,存在安全风险)")
            self.warnings.append("Server host is 0.0.0.0, security risk")
        elif server_host == "127.0.0.1":
            print_success(f"server_config.yaml: host = {server_host} (安全) ✓")
        
        if prod_host == "127.0.0.1":
            print_success(f"production.yaml: host = {prod_host} (安全) ✓")
        elif prod_host == "0.0.0.0":
            print_warning(f"production.yaml: host = {prod_host} (生产环境应谨慎使用)")
    
    def validate_performance(self):
        """验证性能相关配置"""
        print_header("验证性能配置")
        
        cache_config = self.configs.get('cache_config.yaml', {}).get('cache', {})
        cleanup_interval = cache_config.get('cleanup_interval', 0)
        
        if cleanup_interval >= 5000:
            print_success(f"cleanup_interval = {cleanup_interval} ms (已优化) ✓")
        elif cleanup_interval <= 1000:
            print_warning(f"cleanup_interval = {cleanup_interval} ms (过于频繁,建议 >= 5000)")
            self.warnings.append(f"Cleanup interval too frequent: {cleanup_interval}ms")
        
        # 检查eviction_threshold
        eviction_threshold = cache_config.get('eviction_threshold', 0.9)
        if eviction_threshold <= 0.85:
            print_success(f"eviction_threshold = {eviction_threshold} (提前触发) ✓")
        elif eviction_threshold >= 0.9:
            print_warning(f"eviction_threshold = {eviction_threshold} (建议降低到0.85)")
    
    def generate_report(self):
        """生成验证报告"""
        print_header("验证报告")
        
        total_checks = len(self.errors) + len(self.warnings)
        
        if self.errors:
            print(f"\n{Colors.RED}{Colors.BOLD}发现 {len(self.errors)} 个错误:{Colors.END}")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")
        
        if self.warnings:
            print(f"\n{Colors.YELLOW}{Colors.BOLD}发现 {len(self.warnings)} 个警告:{Colors.END}")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")
        
        if not self.errors and not self.warnings:
            print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 所有配置验证通过!{Colors.END}")
            return 0
        elif not self.errors:
            print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠ 配置基本正确,但有 {len(self.warnings)} 个警告需要关注{Colors.END}")
            return 0
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}❌ 配置验证失败,需要修复 {len(self.errors)} 个错误{Colors.END}")
            return 1

def main():
    # 获取项目根目录
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    config_dir = project_root / "config"
    
    if not config_dir.exists():
        print_error(f"配置目录不存在: {config_dir}")
        return 1
    
    print(f"{Colors.BOLD}cLLM 配置验证工具{Colors.END}")
    print(f"配置目录: {config_dir}")
    
    # 创建验证器
    validator = ConfigValidator(config_dir)
    
    # 加载配置
    if not validator.load_configs():
        print_error("配置加载失败")
        return 1
    
    # 执行验证
    validator.validate_vocab_size()
    validator.validate_sampler_consistency()
    validator.validate_cache_config()
    validator.validate_batch_size()
    validator.validate_security()
    validator.validate_performance()
    
    # 生成报告
    return validator.generate_report()

if __name__ == "__main__":
    sys.exit(main())
