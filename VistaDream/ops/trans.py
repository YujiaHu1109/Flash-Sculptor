import struct

# 假设外参从第 24 字节开始（跳过头部和预留字段）
extrinsic_data = b'\nDy\xff\xff\x00\x00\x00\x00\x00\x00\x00\x00\x04\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00?\x7f\xff\xf5\x00\x00\x00\x00\x00\x00\x00\x00?rGw?\x80\x00\x00<\x13t\xbcD\x89\x80\x00'

# 解析为 float32 数组
floats = []
for i in range(0, len(extrinsic_data), 4):
    chunk = extrinsic_data[i:i+4]
    if len(chunk) < 4:
        break
    val = struct.unpack('<f', chunk)[0]  # 小端序 float32
    floats.append(val)

print("解析的浮点数:", floats)