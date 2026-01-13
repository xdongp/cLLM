#!/usr/bin/env python3
"""测试 GGUF 文件格式"""
import struct
import sys
import os

# 设置文件路径
filepath = "/Users/dannypan/PycharmProjects/xllm/cpp/cLLM/model/Qwen/qwen3-0.6b-q8_0.gguf"

if not os.path.exists(filepath):
    print(f"错误: 文件不存在: {filepath}")
    sys.exit(1)

print("=" * 70)
print("GGUF 文件格式验证")
print("=" * 70)
print(f"文件: {filepath}")
print(f"文件大小: {os.path.getsize(filepath)} 字节")
print()

try:
    with open(filepath, 'rb') as f:
        # 1. 读取文件头
        print("--- 文件头分析 ---")
        magic = f.read(4)
        print(f"魔数: {magic} = {magic.hex()}")
        
        if magic != b'GGUF':
            print("❌ 错误: 不是有效的GGUF文件!")
            sys.exit(1)
        print("✅ 魔数正确")
        
        version = struct.unpack('<I', f.read(4))[0]
        print(f"版本: {version}")
        
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        print(f"张量数量: {tensor_count}")
        
        metadata_count = struct.unpack('<Q', f.read(8))[0]
        print(f"元数据数量: {metadata_count}")
        
        header_end = f.tell()
        print(f"文件头结束位置: {header_end} 字节")
        print()
        
        # 2. 分析第一个元数据条目（这是出错的地方）
        print("=" * 70)
        print("第一个元数据条目详细分析（位置 24-52）")
        print("=" * 70)
        
        pos_24 = f.tell()
        print(f"当前位置: {pos_24} 字节")
        
        # 读取位置 24-60 的原始数据用于调试
        raw_data = f.read(36)
        f.seek(pos_24)
        print(f"原始数据 (36字节): {raw_data.hex()}")
        print(f"原始数据 (ASCII尝试): {repr(raw_data)}")
        print()
        
        # 解析第一个元数据条目
        print("--- 解析元数据条目 #0 ---")
        
        # 读取键长度
        key_len_bytes = f.read(4)
        key_len = struct.unpack('<I', key_len_bytes)[0]
        print(f"键长度: {key_len} (0x{key_len:08x})")
        print(f"键长度字节: {key_len_bytes.hex()} = {repr(key_len_bytes)}")
        
        if key_len > 1000:
            print(f"❌ 键长度异常大: {key_len}")
        else:
            # 读取键名
            key_bytes = f.read(key_len)
            key = key_bytes.decode('utf-8', errors='replace')
            print(f"键名: '{key}'")
            print(f"键名字节: {key_bytes.hex()}")
            
            # 读取值类型
            value_type_pos = f.tell()
            value_type_bytes = f.read(4)
            value_type = struct.unpack('<I', value_type_bytes)[0]
            
            print(f"值类型位置: {value_type_pos} 字节")
            print(f"值类型: {value_type} (0x{value_type:08x})")
            print(f"值类型字节: {value_type_bytes.hex()} = {repr(value_type_bytes)}")
            print(f"值类型ASCII解释: '{value_type_bytes.decode('ascii', errors='replace')}'")
            
            if value_type > 11:
                print(f"❌ 错误: 值类型 {value_type} 无效!")
                print(f"   最大有效值: 11")
                print(f"   这看起来像是读取了字符串 'true' 而不是值类型")
                print(f"   可能的原因:")
                print(f"   1. 文件格式不匹配（不是标准GGUF v3）")
                print(f"   2. 文件位置计算错误")
                print(f"   3. 字节序问题")
            else:
                print(f"✅ 值类型有效")
                
                # 根据类型读取值
                if value_type == 11:  # STRING
                    str_length = struct.unpack('<I', f.read(4))[0]
                    str_value = f.read(str_length).decode('utf-8', errors='replace')
                    print(f"值 (STRING): '{str_value}' (长度: {str_length})")
                elif value_type == 10:  # BOOL
                    value = struct.unpack('<B', f.read(1))[0] != 0
                    print(f"值 (BOOL): {value}")
                elif value_type == 5:  # INT32
                    value = struct.unpack('<i', f.read(4))[0]
                    print(f"值 (INT32): {value}")
                elif value_type == 4:  # UINT32
                    value = struct.unpack('<I', f.read(4))[0]
                    print(f"值 (UINT32): {value}")
        
        print()
        print("=" * 70)
        print("验证完成")
        print("=" * 70)
        
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
