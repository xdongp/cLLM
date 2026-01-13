#!/usr/bin/env python3
import struct

def main():
    filepath = "/Users/dannypan/PycharmProjects/xllm/cpp/cLLM/model/Qwen/qwen3-0.6b-q8_0.gguf"
    
    with open(filepath, 'rb') as f:
        # Read file header
        magic = struct.unpack('<I', f.read(4))[0]
        version = struct.unpack('<I', f.read(4))[0]
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        metadata_count = struct.unpack('<Q', f.read(8))[0]
        
        print(f"File header:")
        print(f"  Magic: 0x{magic:08x}")
        print(f"  Version: {version}")
        print(f"  Tensor count: {tensor_count}")
        print(f"  Metadata count: {metadata_count}")
        print(f"  Header end offset: {f.tell()}")
        print()
        
        # Now read metadata entries with proper GGUF v3 format
        for i in range(metadata_count):
            print(f"=== Metadata entry #{i} ===")
            print(f"  Current offset: {f.tell()}")
            
            try:
                # Read key length (8 bytes for GGUF v3)
                key_len_bytes = f.read(8)
                if len(key_len_bytes) < 8:
                    print(f"  ❌ Failed to read key length")
                    break
                
                key_len = struct.unpack('<Q', key_len_bytes)[0]
                print(f"  Key length: {key_len} (0x{key_len:016x})")
                print(f"  Key length bytes: {key_len_bytes.hex()}")
                print(f"  Offset after key length: {f.tell()}")
                
                # Read key
                key_bytes = f.read(key_len)
                if len(key_bytes) < key_len:
                    print(f"  ❌ Failed to read key (need {key_len} bytes, got {len(key_bytes)})")
                    break
                
                key = key_bytes.decode('utf-8', errors='replace')
                print(f"  Key: '{key}'")
                print(f"  Key bytes: {key_bytes.hex()}")
                print(f"  Offset after key: {f.tell()}")
                
                # Read value type (4 bytes)
                value_type_bytes = f.read(4)
                if len(value_type_bytes) < 4:
                    print(f"  ❌ Failed to read value type")
                    break
                
                value_type = struct.unpack('<I', value_type_bytes)[0]
                print(f"  Value type: {value_type} (0x{value_type:08x})")
                print(f"  Value type bytes: {value_type_bytes.hex()}")
                print(f"  Offset after value type: {f.tell()}")
                
                # Read value based on type
                if value_type <= 11:
                    if value_type in [0, 1, 10]:  # UINT8, INT8, BOOL
                        value_bytes = f.read(1)
                        value = struct.unpack('<B' if value_type != 1 else '<b', value_bytes)[0]
                        value = (value != 0) if value_type == 10 else value
                    elif value_type in [2, 3]:  # UINT16, INT16
                        value_bytes = f.read(2)
                        value = struct.unpack('<H' if value_type == 2 else '<h', value_bytes)[0]
                    elif value_type in [4, 5, 8]:  # UINT32, INT32, FLOAT32
                        value_bytes = f.read(4)
                        if value_type == 8:
                            value = struct.unpack('<f', value_bytes)[0]
                        else:
                            value = struct.unpack('<I' if value_type == 4 else '<i', value_bytes)[0]
                    elif value_type in [6, 7, 9]:  # UINT64, INT64, FLOAT64
                        value_bytes = f.read(8)
                        if value_type == 9:
                            value = struct.unpack('<d', value_bytes)[0]
                        else:
                            value = struct.unpack('<Q' if value_type == 6 else '<q', value_bytes)[0]
                    elif value_type == 11:  # STRING
                        str_len_bytes = f.read(4)
                        str_len = struct.unpack('<I', str_len_bytes)[0]
                        value_bytes = f.read(str_len)
                        value = value_bytes.decode('utf-8', errors='replace')
                        print(f"  String length: {str_len} (0x{str_len:08x})")
                    elif value_type == 12:  # ARRAY - need special handling
                        print(f"  Array type, need to skip manually")
                        continue
                    
                    print(f"  Value: {value}")
                    print(f"  Offset after value: {f.tell()}")
                else:
                    print(f"  ❌ Invalid value type")
                    
                print()
                
                # Stop after first 5 entries to avoid too much output
                if i >= 4:
                    print("... skipping remaining entries ...")
                    break
                    
            except Exception as e:
                print(f"  ❌ Error: {e}")
                break

if __name__ == "__main__":
    main()
