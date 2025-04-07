with open("x", "rb") as f:
    content = f.read(1000)  # 读取前 100 字节
    print(content)
