#!/usr/bin/env python3
import struct

def analyze_gguf_metadata(filepath):
    with open(filepath, 'rb') as f:
        # 读取文件头
        magic = struct.unpack('<I', f.read(4))[0]
        version = struct.unpack('<I', f.read(4))[0]
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        metadata_count = struct.unpack('<Q', f.read(8))[0]
        
        print(f"魔数: 0x{magic:08x}")
        print(f"版本: {version}")
        print(f"张量数量: {tensor_count}")
        print(f"元数据数量: {metadata_count}")
        print(f"文件头结束位置: {f.tell()}")
        print()
        
        # 读取元数据
        for i in range(metadata_count):
            print(f"=== 元数据条目 #{i} ===")
            print(f"当前位置: {f.tell()}")
            
            # 读取键长度
            key_length_bytes = f.read(4)
            if len(key_length_bytes) < 4:
                print(f"无法读取键长度（文件结束）")
                break
            key_length = struct.unpack('<I', key_length_bytes)[0]
            print(f"键长度: {key_length} (0x{key_length:08x})")
            
            # 读取键内容
            if key_length > 1000:  # 防止读取过大的键
                print(f"键长度过大: {key_length}")
                break
            
            key_bytes = f.read(key_length)
            if len(key_bytes) < key_length:
                print(f"无法读取完整键内容（需要 {key_length} 字节，实际读取 {len(key_bytes)} 字节）")
                break
            key = key_bytes.decode('utf-8', errors='replace')
            print(f"键名: '{key}'")
            print(f"键名十六进制: {key_bytes.hex()}")
            
            # 读取值类型
            value_type_bytes = f.read(4)
            if len(value_type_bytes) < 4:
                print(f"无法读取值类型（文件结束）")
                break
            value_type = struct.unpack('<I', value_type_bytes)[0]
            print(f"值类型: {value_type} (0x{value_type:08x})")
            
            if value_type > 11:
                print(f"  ❌ 无效的值类型！")
                print(f"  值类型字节: {value_type_bytes.hex()}")
                # 尝试解释为ASCII
                try:
                    ascii_str = value_type_bytes.decode('ascii', errors='replace')
                    print(f"  值类型ASCII: '{ascii_str}'")
                except:
                    pass
                break
            
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
                print(f"值 (STRING): '{str_value}' (长度: {str_length})")
            else:
                print(f"未知的值类型: {value_type}")
                break
            
            print()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = "/Users/dannypan/PycharmProjects/xllm/cpp/cLLM/model/Qwen/qwen3-0.6b-q8_0.gguf"
    
    print("=" * 70)
    print("GGUF 文件格式验证")
    print("=" * 70)
    print(f"文件: {filepath}")
    print()
    
    try:
        analyze_gguf_metadata(filepath)
        print("=" * 70)
        print("✅ 验证完成")
    except Exception as e:
        print("=" * 70)
        print(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
