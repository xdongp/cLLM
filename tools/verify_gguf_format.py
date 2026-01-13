#!/usr/bin/env python3
"""
使用Python实现的GGUF文件解析器，用于验证文件格式
基于GGUF版本3规范
"""

import struct
import sys

def read_uint32(f):
    """读取32位无符号整数（小端）"""
    return struct.unpack('<I', f.read(4))[0]

def read_uint64(f):
    """读取64位无符号整数（小端）"""
    return struct.unpack('<Q', f.read(8))[0]

def read_string(f):
    """读取字符串（带32位长度前缀）"""
    length = read_uint32(f)
    return f.read(length).decode('utf-8', errors='replace')

def main():
    if len(sys.argv) != 2:
        print(f"用法: {sys.argv[0]} <GGUF文件路径>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    with open(file_path, 'rb') as f:
        # 读取文件头
        print("=== GGUF文件解析 ===")
        
        # 1. 魔数 (4 bytes)
        magic = f.read(4)
        magic_int = struct.unpack('<I', magic)[0]
        print(f"1. 魔数: {magic.hex()} = 0x{magic_int:08x} = '{magic.decode('utf-8')}'")
        
        if magic != b'GGUF':
            print("错误: 不是有效的GGUF文件")
            sys.exit(1)
        
        # 2. 版本号 (1 byte)
        version = f.read(1)
        version_int = struct.unpack('<B', version)[0]
        print(f"2. 版本号: {version.hex()} = {version_int}")
        
        if version_int != 3:
            print(f"警告: 不是GGUF版本3，当前版本: {version_int}")
        
        # 3. 跳过3个填充字节（版本号是1字节，对齐到4字节）
        padding = f.read(3)
        print(f"3. 版本填充: {padding.hex()}")
        
        # 4. 张量数量 (8 bytes)
        tensor_count = read_uint64(f)
        print(f"4. 张量数量: {tensor_count}")
        
        # 5. 元数据数量 (8 bytes)
        metadata_count = read_uint64(f)
        print(f"5. 元数据数量: {metadata_count}")
        
        print(f"\n文件头解析完成，当前位置: {f.tell()} 字节")
        
        # 解析元数据
        print("\n=== 元数据解析 ===")
        print(f"预计有 {metadata_count} 个元数据条目")
        
        for i in range(min(5, metadata_count)):  # 只解析前5个
            print(f"\n元数据条目 #{i}:")
            
            try:
                # 1. 键名长度 (4 bytes)
                key_len = read_uint32(f)
                print(f"  键长度: {key_len}")
                
                # 2. 键名
                key = f.read(key_len).decode('utf-8', errors='replace')
                print(f"  键名: '{key}'")
                
                # 3. 值类型 (4 bytes)
                value_type = read_uint32(f)
                print(f"  值类型: {value_type} = 0x{value_type:08x}")
                
                # 检查值类型是否有效
                if 0 <= value_type <= 11:
                    print(f"  ✅ 值类型有效")
                else:
                    print(f"  ❌ 值类型无效！")
                    
                    # 读取几个字节看看是什么
                    print(f"  位置: {f.tell()}")
                    data = f.read(16)
                    print(f"  后面的数据: {data.hex()} = '{data.decode('utf-8', errors='replace')}'")
                    return
                
            except Exception as e:
                print(f"  ❌ 解析错误: {e}")
                return

if __name__ == "__main__":
    main()