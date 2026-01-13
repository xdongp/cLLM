#!/usr/bin/env python3
"""
验证 GGUF 文件格式的脚本
"""
import struct
import sys

def verify_gguf(filepath):
    print(f"分析文件: {filepath}")
    print("=" * 60)
    
    with open(filepath, 'rb') as f:
        # 读取文件头
        magic_bytes = f.read(4)
        magic = struct.unpack('<I', magic_bytes)[0]
        
        print(f"魔数: {magic_bytes} = 0x{magic:08x}")
        if magic_bytes != b'GGUF':
            print("❌ 错误: 不是有效的GGUF文件!")
            return False
        print("✅ 魔数正确")
        
        # 读取版本号
        version_bytes = f.read(4)
        version = struct.unpack('<I', version_bytes)[0]
        print(f"版本: {version} (0x{struct.unpack('<I', version_bytes)[0]:08x})")
        
        # 读取张量数量
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        print(f"张量数量: {tensor_count}")
        
        # 读取元数据数量
        metadata_count = struct.unpack('<Q', f.read(8))[0]
        print(f"元数据数量: {metadata_count}")
        
        header_end = f.tell()
        print(f"文件头结束位置: {header_end} 字节")
        print()
        
        # 分析前几个元数据条目
        print("=" * 60)
        print("分析前3个元数据条目:")
        print("=" * 60)
        
        for i in range(min(3, metadata_count)):
            print(f"\n--- 元数据条目 #{i} ---")
            start_pos = f.tell()
            print(f"开始位置: {start_pos} 字节")
            
            # 读取原始字节用于调试
            raw_peek = f.read(32)
            f.seek(start_pos)
            print(f"原始数据 (前32字节): {raw_peek.hex()}")
            
            try:
                # 1. 读取键长度
                key_len_bytes = f.read(4)
                if len(key_len_bytes) < 4:
                    print("❌ 无法读取键长度")
                    break
                key_len = struct.unpack('<I', key_len_bytes)[0]
                print(f"键长度: {key_len} (0x{key_len:08x})")
                
                if key_len > 1000:
                    print(f"❌ 键长度异常大: {key_len}")
                    break
                
                # 2. 读取键名
                key_bytes = f.read(key_len)
                if len(key_bytes) < key_len:
                    print(f"❌ 无法读取完整键名 (需要 {key_len}, 实际 {len(key_bytes)})")
                    break
                key = key_bytes.decode('utf-8', errors='replace')
                print(f"键名: '{key}'")
                
                # 3. 读取值类型
                value_type_bytes = f.read(4)
                if len(value_type_bytes) < 4:
                    print("❌ 无法读取值类型")
                    break
                value_type = struct.unpack('<I', value_type_bytes)[0]
                print(f"值类型: {value_type} (0x{value_type:08x})")
                
                # 检查值类型是否有效
                if value_type > 11:
                    print(f"❌ 无效的值类型! (最大有效值: 11)")
                    print(f"   值类型字节: {value_type_bytes.hex()}")
                    print(f"   值类型ASCII解释: '{value_type_bytes.decode('ascii', errors='replace')}'")
                    print(f"   这可能是文件格式错误或解析位置错误")
                    return False
                
                # 根据值类型读取值
                if value_type == 0:  # UINT8
                    value = struct.unpack('<B', f.read(1))[0]
                    print(f"值 (UINT8): {value}")
                elif value_type == 1:  # INT8
                    value = struct.unpack('<b', f.read(1))[0]
                    print(f"值 (INT8): {value}")
                elif value_type == 2:  # UINT16
                    value = struct.unpack('<H', f.read(2))[0]
                    print(f"值 (UINT16): {value}")
                elif value_type == 3:  # INT16
                    value = struct.unpack('<h', f.read(2))[0]
                    print(f"值 (INT16): {value}")
                elif value_type == 4:  # UINT32
                    value = struct.unpack('<I', f.read(4))[0]
                    print(f"值 (UINT32): {value}")
                elif value_type == 5:  # INT32
                    value = struct.unpack('<i', f.read(4))[0]
                    print(f"值 (INT32): {value}")
                elif value_type == 6:  # UINT64
                    value = struct.unpack('<Q', f.read(8))[0]
                    print(f"值 (UINT64): {value}")
                elif value_type == 7:  # INT64
                    value = struct.unpack('<q', f.read(8))[0]
                    print(f"值 (INT64): {value}")
                elif value_type == 8:  # FLOAT32
                    value = struct.unpack('<f', f.read(4))[0]
                    print(f"值 (FLOAT32): {value}")
                elif value_type == 9:  # FLOAT64
                    value = struct.unpack('<d', f.read(8))[0]
                    print(f"值 (FLOAT64): {value}")
                elif value_type == 10:  # BOOL
                    value = struct.unpack('<B', f.read(1))[0] != 0
                    print(f"值 (BOOL): {value}")
                elif value_type == 11:  # STRING
                    str_length = struct.unpack('<I', f.read(4))[0]
                    str_value = f.read(str_length).decode('utf-8', errors='replace')
                    print(f"值 (STRING): '{str_value}'")
                
                end_pos = f.tell()
                print(f"结束位置: {end_pos} 字节")
                print(f"条目大小: {end_pos - start_pos} 字节")
                
            except Exception as e:
                print(f"❌ 解析错误: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        print("\n" + "=" * 60)
        print("✅ 文件格式验证完成")
        return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"用法: {sys.argv[0]} <GGUF文件路径>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    success = verify_gguf(filepath)
    sys.exit(0 if success else 1)
