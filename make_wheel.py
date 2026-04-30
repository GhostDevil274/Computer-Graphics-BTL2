import math

def generate_wheel_obj(filename="assets/models/wheel.obj", radius=0.3, width=0.2, segments=8):
    with open(filename, 'w') as f:
        f.write("# Banh xe tu dong tao boi Gemini cho Dai ca\n")
        f.write("o Wheel\n\n")

        # 1. TẠO CÁC ĐỈNH (Vertices)
        # Tâm bên trái (ID = 1) và Tâm bên phải (ID = 2)
        f.write(f"v {-width/2} 0.0 0.0\n")
        f.write(f"v {width/2} 0.0 0.0\n")

        # Các đỉnh viền bánh xe
        for i in range(segments):
            angle = 2.0 * math.pi * i / segments
            y = radius * math.cos(angle)
            z = radius * math.sin(angle)
            f.write(f"v {-width/2} {y} {z}\n") # Viền trái (Số lẻ: 3, 5, 7...)
            f.write(f"v {width/2} {y} {z}\n")  # Viền phải (Số chẵn: 4, 6, 8...)

        f.write("\n")
        
        # 2. TẠO TỌA ĐỘ DÁN ẢNH (UV Textures - Dummy UV)
        # File loader của ông bắt buộc phải có /vt nên phải tạo ra
        f.write("vt 0.5 0.5\n") # 1 (Tâm)
        f.write("vt 0.0 0.0\n") # 2 (Góc 1)
        f.write("vt 1.0 1.0\n") # 3 (Góc 2)
        f.write("\n")

        # 3. TẠO CÁC MẶT TAM GIÁC (Faces)
        # Mặt mâm xe bên Trái
        for i in range(segments):
            v1 = 3 + i * 2
            v2 = 3 + ((i + 1) % segments) * 2
            f.write(f"f 1/1 {v2}/2 {v1}/3\n")

        # Mặt mâm xe bên Phải
        for i in range(segments):
            v1 = 4 + i * 2
            v2 = 4 + ((i + 1) % segments) * 2
            f.write(f"f 2/1 {v1}/2 {v2}/3\n")

        # Mặt lốp xe (Tread) - Nối viền trái và viền phải
        for i in range(segments):
            l1 = 3 + i * 2
            r1 = 4 + i * 2
            l2 = 3 + ((i + 1) % segments) * 2
            r2 = 4 + ((i + 1) % segments) * 2
            # 1 mặt chữ nhật chia làm 2 tam giác
            f.write(f"f {l1}/2 {r1}/3 {l2}/2\n")
            f.write(f"f {r1}/3 {r2}/3 {l2}/2\n")
            
    print(f"[+] Da tao thanh cong file {filename} !!")

if __name__ == "__main__":
    generate_wheel_obj()