#!/usr/bin/env python3
"""验证 GGUF 文件格式并输出到文件"""
import struct
import sys

filepath = sys.argv[1] if len(sys.argv) > 1 else "/Users/dannypan/PycharmProjects/xllm/cpp/cLLM/model/Qwen/qwen3-0.6b-q8_0.gguf"
output_file = "/tmp/gguf_verify_result.txt"

with open(output_file, 'w') as out:
    out.write(f"验证文件: {filepath}\n")
    out.write("=" * 70 + "\n\n")
    
    with open(filepath, 'rb') as f:
        # 读取文件头
        magic = f.read(4)
        out.write(f"魔数: {magic} = {magic.hex()}\n")
        
        if magic != b'GGUF':
            out.write("❌ 不是有效的GGUF文件!\n")
            sys.exit(1)
        
        version = struct.unpack('<I', f.read(4))[0]
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        metadata_count = struct.unpack('<Q', f.read(8))[0]
        
        out.write(f"版本: {version}\n")
        out.write(f"张量数量: {tensor_count}\n")
        out.write(f"元数据数量: {metadata_count}\n")
        out.write(f"文件头结束位置: {f.tell()} 字节\n\n")
        
        # 分析前3个元数据条目
        out.write("=" * 70 + "\n")
        out.write("分析前3个元数据条目:\n")
        out.write("=" * 70 + "\n\n")
        
        for i in range(min(3, metadata_count)):
            out.write(f"--- 元数据条目 #{i} ---\n")
            start_pos = f.tell()
            out.write(f"开始位置: {start_pos} 字节\n")
            
            # 读取原始数据
            raw_peek = f.read(40)
            f.seek(start_pos)
            out.write(f"原始数据 (前40字节): {raw_peek.hex()}\n")
            
            try:
                # 读取键长度
                key_len_bytes = f.read(4)
                key_len = struct.unpack('<I', key_len_bytes)[0]
                out.write(f"键长度: {key_len} (0x{key_len:08x}) = {key_len_bytes.hex()}\n")
                
                if key_len > 1000:
                    out.write(f"❌ 键长度异常大!\n")
                    break
                
                # 读取键名
                key_bytes = f.read(key_len)
                key = key_bytes.decode('utf-8', errors='replace')
                out.write(f"键名: '{key}'\n")
                out.write(f"键名十六进制: {key_bytes.hex()}\n")
                
                # 读取值类型
                value_type_pos = f.tell()
                value_type_bytes = f.read(4)
                value_type = struct.unpack('<I', value_type_bytes)[0]
                out.write(f"值类型位置: {value_type_pos}\n")
                out.write(f"值类型: {value_type} (0x{value_type:08x})\n")
                out.write(f"值类型字节: {value_type_bytes.hex()} = '{value_type_bytes.decode('ascii', errors='replace')}'\n")
                
                if value_type > 11:
                    out.write(f"❌ 错误: 值类型 {value_type} 无效 (最大有效值: 11)\n")
                else:
                    out.write(f"✅ 值类型有效\n")
                    
                    # 根据类型读取值
                    if value_type == 11:  # STRING
                        str_length = struct.unpack('<I', f.read(4))[0]
                        str_value = f.read(str_length).decode('utf-8', errors='replace')
                        out.write(f"值 (STRING): '{str_value}'\n")
                    elif value_type == 10:  # BOOL
                        value = struct.unpack('<B', f.read(1))[0] != 0
                        out.write(f"值 (BOOL): {value}\n")
                    elif value_type == 5:  # INT32
                        value = struct.unpack('<i', f.read(4))[0]
                        out.write(f"值 (INT32): {value}\n")
                    elif value_type == 4:  # UINT32
                        value = struct.unpack('<I', f.read(4))[0]
                        out.write(f"值 (UINT32): {value}\n")
                
                end_pos = f.tell()
                out.write(f"结束位置: {end_pos} 字节\n")
                out.write(f"条目大小: {end_pos - start_pos} 字节\n\n")
                
            except Exception as e:
                out.write(f"❌ 解析错误: {e}\n")
                import traceback
                out.write(traceback.format_exc())
                break

print(f"验证结果已保存到: {output_file}")
