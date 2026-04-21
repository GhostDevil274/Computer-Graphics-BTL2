import numpy as np
import math
import os
from shapes.base_shape import BaseShape

def generate_rainbow_colors(vertices):
    coords = np.array(vertices, dtype=np.float32)
    max_val = np.max(np.abs(coords)) if np.max(np.abs(coords)) != 0 else 1.0
    return ((coords / max_val + 1.0) / 2.0).astype(np.float32)

class Cylinder(BaseShape):
    def __init__(self, radius=0.5, height=1.0, segments=36, top_r=None):
        if top_r is None: top_r = radius 
        vertices, indices = [], []
        half_h = height / 2.0
        vertices.extend([[0.0, -half_h, 0.0], [0.0, half_h, 0.0]]) 
        
        angle_step = (2 * math.pi) / segments
        for i in range(segments):
            angle = i * angle_step
            vertices.append([radius * math.cos(angle), -half_h, radius * math.sin(angle)])
            vertices.append([top_r * math.cos(angle),   half_h, top_r * math.sin(angle)])

        for i in range(segments):
            b_cur, t_cur = 2 + i * 2, 3 + i * 2
            b_nxt, t_nxt = 2 + ((i + 1) % segments) * 2, 3 + ((i + 1) % segments) * 2

            indices.extend([0, b_nxt, b_cur]) 
            indices.extend([1, t_cur, t_nxt]) 
            indices.extend([b_cur, t_cur, b_nxt, b_nxt, t_cur, t_nxt]) 

        super().__init__(np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32), generate_rainbow_colors(vertices))

class Cone(Cylinder):
    def __init__(self):
        super().__init__(radius=0.5, height=1.0, segments=36, top_r=0.0)

class TruncatedCone(Cylinder):
    def __init__(self):
        super().__init__(radius=0.6, height=1.0, segments=36, top_r=0.3)

class Torus(BaseShape):
    def __init__(self, R=0.6, r=0.2, rings=36, sectors=36):
        vertices, indices = [], []
        for i in range(rings):
            u = i * (2 * math.pi) / rings
            for j in range(sectors):
                v = j * (2 * math.pi) / sectors
                x = (R + r * math.cos(v)) * math.cos(u)
                y = r * math.sin(v)
                z = (R + r * math.cos(v)) * math.sin(u)
                vertices.append([x, y, z])
                
                nxt_i, nxt_j = (i + 1) % rings, (j + 1) % sectors
                a, b = i * sectors + j, nxt_i * sectors + j
                c, d = nxt_i * sectors + nxt_j, i * sectors + nxt_j
                indices.extend([a, b, c, a, c, d])
                
        super().__init__(np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32), generate_rainbow_colors(vertices))

class SphereLatLong(BaseShape):
    def __init__(self, radius=0.6, sectors=36, stacks=18):
        vertices, indices = [], []
        for i in range(stacks + 1):
            phi = math.pi / 2 - i * math.pi / stacks 
            xy, z = radius * math.cos(phi), radius * math.sin(phi)
            for j in range(sectors + 1):
                theta = j * 2 * math.pi / sectors
                vertices.append([xy * math.cos(theta), xy * math.sin(theta), z])
                
        for i in range(stacks):
            k1 = i * (sectors + 1)
            k2 = k1 + sectors + 1
            for j in range(sectors):
                if i != 0: indices.extend([k1, k2, k1+1])
                if i != (stacks - 1): indices.extend([k1+1, k2, k2+1])
                k1 += 1; k2 += 1
                
        super().__init__(np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32), generate_rainbow_colors(vertices))


class SphereSubdivision(BaseShape):
    def __init__(self, radius=0.8, subdivisions=4):
        verts = [
            np.array([1, 1, 1], dtype=np.float32), 
            np.array([-1, -1, 1], dtype=np.float32),
            np.array([-1, 1, -1], dtype=np.float32), 
            np.array([1, -1, -1], dtype=np.float32)
        ]
        inds = [[0,1,2], [0,3,1], [0,2,3], [1,3,2]]

        def normalize(v):
            return v / np.linalg.norm(v)

        for _ in range(subdivisions):
            new_inds = []
            for tri in inds:
                v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
                
                m01 = normalize(v0 + v1)
                m12 = normalize(v1 + v2)
                m20 = normalize(v2 + v0)

                idx = len(verts)
                verts.extend([m01, m12, m20])
                i01, i12, i20 = idx, idx+1, idx+2

                new_inds.extend([
                    [tri[0], i01, i20],
                    [tri[1], i12, i01],
                    [tri[2], i20, i12],
                    [i01, i12, i20]
                ])
            inds = new_inds

        final_verts = [(v / np.linalg.norm(v) * radius).tolist() for v in verts]
        flat_inds = [i for tri in inds for i in tri]
        
        super().__init__(np.array(final_verts, dtype=np.float32), 
                         np.array(flat_inds, dtype=np.uint32), 
                         generate_rainbow_colors(final_verts))

class SphereCube(BaseShape):
    def __init__(self, radius=0.8, resolution=16):
        vertices = []
        indices = []
        
        faces = [
            ( (1,0,0), (0,1,0), (0,0,1) ),   # Phải
            ( (-1,0,0), (0,1,0), (0,0,-1) ), # Trái
            ( (0,1,0), (0,0,-1), (1,0,0) ),  # Trên
            ( (0,-1,0), (0,0,1), (1,0,0) ),  # Dưới
            ( (0,0,1), (0,1,0), (-1,0,0) ),  # Trước
            ( (0,0,-1), (0,1,0), (1,0,0) )   # Sau
        ]
        
        offset = 0
        for normal, up, right in faces:
            n, u, r = np.array(normal), np.array(up), np.array(right)
            
            for i in range(resolution + 1):
                for j in range(resolution + 1):
                    x = (j / resolution - 0.5) * 2.0
                    y = (i / resolution - 0.5) * 2.0
                    
                    point = n + u * y + r * x
                    point = point / np.linalg.norm(point) * radius
                    vertices.append(point.tolist())
                    
            for i in range(resolution):
                for j in range(resolution):
                    idx = offset + i * (resolution + 1) + j
                    idx_r = idx + 1
                    idx_d = idx + (resolution + 1)
                    idx_dr = idx_d + 1
                    
                    indices.extend([idx, idx_d, idx_r, idx_r, idx_d, idx_dr])
            
            offset += (resolution + 1) ** 2

        super().__init__(np.array(vertices, dtype=np.float32), 
                         np.array(indices, dtype=np.uint32), 
                         generate_rainbow_colors(vertices))
        

class Cube(BaseShape):
    def __init__(self, side=1.0):
        h = side / 2.0
        vertices = [
            [-h,-h, h], [ h,-h, h], [ h, h, h], [-h, h, h], # Trước
            [ h,-h,-h], [-h,-h,-h], [-h, h,-h], [ h, h,-h], # Sau
            [-h,-h,-h], [-h,-h, h], [-h, h, h], [-h, h,-h], # Trái
            [ h,-h, h], [ h,-h,-h], [ h, h,-h], [ h, h, h], # Phải
            [-h, h, h], [ h, h, h], [ h, h,-h], [-h, h,-h], # Trên
            [-h,-h,-h], [ h,-h,-h], [ h,-h, h], [-h,-h, h]  # Dưới
        ]
        
        cw, rh = 1.0 / 4.0, 1.0 / 3.0
        def get_uv(col, row):
            return [
                [col * cw, row * rh], [(col + 1) * cw, row * rh], 
                [(col + 1) * cw, (row + 1) * rh], [col * cw, (row + 1) * rh]
            ]
            
        uvs = []
        uvs.extend(get_uv(1, 1)) # Mặt Trước
        uvs.extend(get_uv(3, 1)) # Mặt Sau
        uvs.extend(get_uv(0, 1)) # Mặt Trái
        uvs.extend(get_uv(2, 1)) # Mặt Phải
        uvs.extend(get_uv(2, 2)) # Mặt Trên
        uvs.extend(get_uv(2, 0)) # Mặt Dưới
        
        indices = [
            0,1,2, 0,2,3,       4,5,6, 4,6,7,
            8,9,10, 8,10,11,    12,13,14, 12,14,15,
            16,17,18, 16,18,19, 20,21,22, 20,22,23
        ]
        super().__init__(np.array(vertices, dtype=np.float32), 
                         np.array(indices, dtype=np.uint32), 
                         generate_rainbow_colors(vertices),
                         uvs=uvs)

class Tetrahedron(BaseShape):
    def __init__(self, size=0.8):
        vertices = [
            [ size,  size,  size],
            [-size, -size,  size],
            [-size,  size, -size],
            [ size, -size, -size]
        ]
        indices = [
            0, 1, 2, # Mặt 1
            0, 3, 1, # Mặt 2
            0, 2, 3, # Mặt 3
            1, 3, 2  # Mặt đáy
        ]
        super().__init__(np.array(vertices, dtype=np.float32), 
                         np.array(indices, dtype=np.uint32), 
                         generate_rainbow_colors(vertices))


"Mặt toán học được định nghĩa bởi hàm do người dùng cung cấp z = f(x, y)"
class MathSurface(BaseShape):
    def __init__(self, func_str="sin(x) + cos(y)", domain=5.0, resolution=50):
        """ Nặn mặt toán học bằng cách giăng lưới tọa độ """
        # 1. Tạo lưới tọa độ X và Y (dùng np.linspace và meshgrid để tạo lưới caro)
        x_vals = np.linspace(-domain, domain, resolution)
        y_vals = np.linspace(-domain, domain, resolution)
        X, Y = np.meshgrid(x_vals, y_vals)
        
        # 2. Tạo môi trường an toàn chứa các hàm Toán học để dịch chuỗi của người dùng
        safe_env = {
            "sin": np.sin, "cos": np.cos, "tan": np.tan,
            "exp": np.exp, "sqrt": lambda v: np.sqrt(np.abs(v)), # abs để không bị lỗi số âm
            "abs": np.abs, "pi": np.pi, "e": np.e,
            "x": X, "y": Y
        }
        
        # 3. Tính toán độ cao Z
        try:
            # Hàm eval sẽ biến cái chữ "sin(x) + cos(y)" thành lệnh Python chạy thật
            Z = eval(func_str, {"__builtins__": None}, safe_env)
            if isinstance(Z, (int, float)): # Lỡ user nhập hàm hằng z = 2
                Z = np.full_like(X, Z)
        except Exception as e:
            print(f"Lỗi cú pháp Toán học: {e}")
            Z = np.zeros_like(X) # Lỗi thì trả về mặt phẳng dẹt z=0

        # 4. Gom tọa độ X, Y, Z thành danh sách đỉnh [x, y, z]
        vertices = np.column_stack((X.flatten(), Y.flatten(), Z.flatten())).astype(np.float32)
        
        # Scale lại cho vừa khung hình (nếu Z quá cao)
        max_val = np.max(np.abs(vertices)) if np.max(np.abs(vertices)) > 0 else 1.0
        vertices = vertices / max_val * 1.5 

        # 5. Dệt lưới tam giác (Indices)
        indices = []
        for i in range(resolution - 1):
            for j in range(resolution - 1):
                # Lấy 4 góc của 1 ô vuông nhỏ trên lưới
                tl = i * resolution + j         # Top-Left
                tr = tl + 1                     # Top-Right
                bl = (i + 1) * resolution + j   # Bottom-Left
                br = bl + 1                     # Bottom-Right
                
                # Cắt ô vuông thành 2 tam giác
                indices.extend([tl, bl, tr, tr, bl, br])

        super().__init__(np.array(vertices, dtype=np.float32), 
                         np.array(indices, dtype=np.uint32), 
                         generate_rainbow_colors(vertices))
        
        
"Mô hình 3D được nhập từ file .obj hoặc .ply."
class ObjModel(BaseShape):
    def __init__(self, filepath):
        vertices, indices, colors, uvs = [], [], [], []
        materials = {"default": [0.8, 0.8, 0.8]} 
        current_mtl = "default"
        
        # 1. ĐỌC FILE .MTL (Nếu có)
        mtl_filepath = filepath.replace('.obj', '.mtl')
        if os.path.exists(mtl_filepath):
            with open(mtl_filepath, 'r', encoding='utf-8', errors='ignore') as f:
                active_mat = None
                for line in f:
                    parts = line.strip().split()
                    if not parts or line.startswith('#'): continue
                    if parts[0] == 'newmtl':
                        active_mat = parts[1]
                    elif parts[0] == 'Kd' and active_mat:
                        materials[active_mat] = [float(parts[1]), float(parts[2]), float(parts[3])]
        else:
            print(f"[CẢNH BÁO] Không tìm thấy file {os.path.basename(mtl_filepath)}")

        v_temp, vt_temp = [], []
        idx_map = {}
        next_idx = 0
        
        if not os.path.exists(filepath):
            print(f"[LỖI] Không tìm thấy file model: {filepath}")
            super().__init__(np.array([[0,0,0]], 'f'), np.array([0], 'u4'), np.array([[1,0,0]], 'f'))
            return

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.strip().split()
                if not parts or line.startswith('#'): continue
                
                if parts[0] == 'usemtl':
                    current_mtl = parts[1] if parts[1] in materials else "default"
                elif parts[0] == 'v':
                    v_temp.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif parts[0] == 'vt':
                    vt_temp.append([float(parts[1]), float(parts[2])])
                elif parts[0] == 'f':
                    face_verts = parts[1:]
                    for i in range(1, len(face_verts) - 1):
                        for vertex_def in [face_verts[0], face_verts[i], face_verts[i+1]]:
                            v_data = vertex_def.split('/')
                            v_idx = int(v_data[0]) - 1
                            vt_idx = int(v_data[1]) - 1 if len(v_data) > 1 and v_data[1] else -1
                            
                            unique_key = f"{vertex_def}_{current_mtl}"
                            if unique_key not in idx_map:
                                vertices.append(v_temp[v_idx])
                                uvs.append(vt_temp[vt_idx] if vt_idx >= 0 else [0.0, 0.0])
                                colors.append(materials[current_mtl])
                                
                                idx_map[unique_key] = next_idx
                                next_idx += 1
                            indices.append(idx_map[unique_key])
                            
        super().__init__(
            np.array(vertices, dtype=np.float32), 
            np.array(indices, dtype=np.uint32), 
            np.array(colors, dtype=np.float32), 
            uvs=np.array(uvs, dtype=np.float32) if uvs else None
        )



