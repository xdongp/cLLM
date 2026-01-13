#!/usr/bin/env python3
"""
精确分析GGUF文件头结构
输出每个字段的详细信息，包括位置、值和解释
"""

import struct
import sys

def main():
    if len(sys.argv) != 2:
        print(f"用法: {sys.argv[0]} <GGUF文件路径>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    with open(file_path, 'rb') as f:
        print("=== GGUF文件头精确分析 ===")
        print(f"文件: {file_path}")
        
        # 读取并分析每个字节
        pos = 0
        
        # 1. 魔数 (4 bytes)
        magic = f.read(4)
        magic_int = struct.unpack('<I', magic)[0]
        magic_str = magic.decode('utf-8')
        print(f"[{pos}-{pos+4}] 魔数: 0x{magic_int:08x} = '{magic_str}'")
        pos += 4
        
        if magic != b'GGUF':
            print("错误: 不是有效的GGUF文件")
            sys.exit(1)
        
        # 2. 版本号 (1 byte)
        version = f.read(1)
        version_int = struct.unpack('<B', version)[0]
        print(f"[{pos}-{pos+1}] 版本号: 0x{version_int:02x} = {version_int}")
        pos += 1
        
        if version_int != 3:
            print(f"警告: 不是GGUF版本3，当前版本: {version_int}")
        
        # 3. 版本号填充 (3 bytes)
        version_padding = f.read(3)
        padding_hex = version_padding.hex()
        print(f"[{pos}-{pos+3}] 版本填充: 0x{padding_hex}")
        pos += 3
        
        # 4. 张量数量 (8 bytes)
        tensor_count_bytes = f.read(8)
        tensor_count = struct.unpack('<Q', tensor_count_bytes)[0]
        print(f"[{pos}-{pos+8}] 张量数量: 0x{tensor_count:016x} = {tensor_count}")
        pos += 8
        
        # 5. 元数据数量 (8 bytes)
        metadata_count_bytes = f.read(8)
        metadata_count = struct.unpack('<Q', metadata_count_bytes)[0]
        print(f"[{pos}-{pos+8}] 元数据数量: 0x{metadata_count:016x} = {metadata_count}")
        pos += 8
        
        print(f"\n文件头总大小: {pos} 字节")
        print(f"预计元数据区域开始位置: {pos} 字节")
        
        # 检查张量数量和元数据数量是否合理
        if tensor_count > 1000000:
            print("\n警告: 张量数量异常大，可能是解析错误！")
        
        if metadata_count > 1000:
            print("警告: 元数据数量异常大，可能是解析错误！")
        else:
            print(f"\n元数据数量看起来合理: {metadata_count}")
        
        # 读取元数据区域的开始部分
        print("\n元数据区域开始:")
        
        for i in range(min(3, metadata_count)):  # 解析前3个元数据条目
            print(f"\n元数据条目 #{i}:")
            
            # 键长度 (4 bytes)
            key_len_bytes = f.read(4)
            key_len = struct.unpack('<I', key_len_bytes)[0]
            print(f"  [{pos}-{pos+4}] 键长度: 0x{key_len:08x} = {key_len}")
            pos += 4
            
            # 键名
            key = f.read(key_len)
            key_str = key.decode('utf-8', errors='replace')
            print(f"  [{pos}-{pos+key_len}] 键名: '{key_str}'")
            pos += key_len
            
            # 值类型 (4 bytes)
            value_type_bytes = f.read(4)
            value_type = struct.unpack('<I', value_type_bytes)[0]
            print(f"  [{pos}-{pos+4}] 值类型: 0x{value_type:08x} = {value_type}")
            pos += 4
            
            # 检查值类型是否有效
            if 0 <= value_type <= 11:
                type_names = [
                    "UINT8", "INT8", "UINT16", "INT16", "UINT32", "INT32",
                    "UINT64", "INT64", "FLOAT32", "FLOAT64", "BOOL", "STRING", "ARRAY"
                ]
                print(f"  ✅ 有效值类型: {type_names[value_type]}")
            else:
                print(f"  ❌ 无效值类型！")
                print(f"     十六进制: {value_type_bytes.hex()}")
                print(f"     ASCII: '{value_type_bytes.decode('utf-8', errors='replace')}'")

if __name__ == "__main__":
    main()