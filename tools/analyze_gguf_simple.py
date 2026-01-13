#!/usr/bin/env python3
"""
简单分析GGUF文件的元数据结构
"""

import struct
import sys

def main():
    if len(sys.argv) != 2:
        print(f"用法: {sys.argv[0]} <GGUF文件路径>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    with open(file_path, 'rb') as f:
        # 读取文件头
        magic = f.read(4)
        if magic != b'GGUF':
            print("错误: 不是有效的GGUF文件")
            sys.exit(1)
        
        version = struct.unpack('<I', f.read(4))[0]  # uint32_t
        tensor_count = struct.unpack('<Q', f.read(8))[0]  # uint64_t
        metadata_count = struct.unpack('<Q', f.read(8))[0]  # uint64_t
        
        print(f"GGUF版本: {version}")
        print(f"张量数量: {tensor_count}")
        print(f"元数据数量: {metadata_count}")
        print(f"文件头结束位置: {f.tell()} 字节")
        print()
        
        # 只分析前2个元数据条目
        print("=== 前2个元数据条目分析 ===")
        
        for i in range(min(2, metadata_count)):
            print(f"\n元数据条目 #{i}:")
            start_pos = f.tell()
            print(f"  开始位置: {start_pos} 字节")
            
            # 读取原始字节以便调试
            raw_data = f.read(20)  # 读取20个字节用于调试
            f.seek(start_pos)  # 回到开始位置
            
            print(f"  原始数据: {raw_data.hex()}")
            
            # 尝试解析
            try:
                # 1. 键长度 (4 bytes)
                key_len_data = f.read(4)
                key_len = struct.unpack('<I', key_len_data)[0]
                print(f"  键长度: {key_len} = 0x{key_len:08x}")
                
                # 2. 键名
                key = f.read(key_len)
                print(f"  键名: {key.hex()} = '{key.decode('utf-8', errors='replace')}'")
                
                # 3. 值类型 (4 bytes)
                value_type_data = f.read(4)
                value_type = struct.unpack('<I', value_type_data)[0]
                print(f"  值类型: {value_type} = 0x{value_type:08x} = '{value_type_data.decode('utf-8', errors='replace')}'")
                
                end_pos = f.tell()
                print(f"  结束位置: {end_pos} 字节")
                print(f"  条目大小: {end_pos - start_pos} 字节")
                
            except Exception as e:
                print(f"  解析错误: {e}")
                break

if __name__ == "__main__":
    main()