#!/usr/bin/env python3
"""
详细分析GGUF文件结构，特别关注文件头和元数据部分
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
        print("=== GGUF文件头分析 ===")
        
        # 1. 魔数 (4 bytes)
        magic = f.read(4)
        print(f"魔数: {magic.hex()} = {magic.decode('utf-8', errors='replace')}")
        
        if magic != b'GGUF':
            print("错误: 不是有效的GGUF文件")
            sys.exit(1)
        
        # 2. 版本号 (1 byte)
        version = struct.unpack('<B', f.read(1))[0]
        print(f"版本: {version}")
        
        # 3. 张量数量 (8 bytes)
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        print(f"张量数量: {tensor_count}")
        
        # 4. 元数据数量 (8 bytes)
        metadata_count = struct.unpack('<Q', f.read(8))[0]
        print(f"元数据数量: {metadata_count}")
        
        print(f"文件头结束位置: {f.tell()} 字节")
        print()
        
        # 读取元数据
        print("=== 元数据分析 ===")
        print(f"预计有 {metadata_count} 个元数据条目")
        
        for i in range(min(10, metadata_count)):  # 只分析前10个
            print(f"\n元数据条目 #{i}:")
            print(f"  开始位置: {f.tell()} 字节")
            
            try:
                # 尝试解析键长度 (4 bytes)
                key_len_data = f.read(4)
                if len(key_len_data) < 4:
                    print(f"  错误: 无法读取键长度 (仅读取到 {len(key_len_data)} 字节)")
                    break
                
                key_len = struct.unpack('<I', key_len_data)[0]
                print(f"  键长度: {key_len} 字节")
                
                # 尝试解析键名
                key_data = f.read(key_len)
                if len(key_data) < key_len:
                    print(f"  错误: 无法读取键名 (仅读取到 {len(key_data)} 字节，预期 {key_len} 字节)")
                    break
                
                key = key_data.decode('utf-8', errors='replace')
                print(f"  键名: '{key}'")
                
                # 尝试解析值类型 (4 bytes)
                value_type_data = f.read(4)
                if len(value_type_data) < 4:
                    print(f"  错误: 无法读取值类型 (仅读取到 {len(value_type_data)} 字节)")
                    break
                
                value_type = struct.unpack('<I', value_type_data)[0]
                print(f"  值类型: 0x{value_type:08x} = '{value_type_data.decode('utf-8', errors='replace')}'")
                
                # 根据值类型解析值
                print(f"  值类型整数: {value_type}")
                
                # 这里我们不完整解析值，只记录位置
                print(f"  结束位置: {f.tell()} 字节")
                
            except Exception as e:
                print(f"  解析错误: {e}")
                print(f"  当前位置: {f.tell()} 字节")
                break
    
    print("\n分析完成")

if __name__ == "__main__":
    main()