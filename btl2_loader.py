import OpenGL.GL as GL
import numpy as np

class KenneyModel:
    def __init__(self, filepath):
        self.vao = 0
        self.vbo = 0
        self.vertex_count = 0
        self.mesh_data = None
        
        self._load_obj(filepath)
        self._setup_gl()

    def _load_obj(self, filepath):
        temp_vertices = []
        temp_normals = []
        temp_texcoords = []
        
        # Mảng dữ liệu cuối cùng nạp vào card đồ họa: [x,y,z, nx,ny,nz, u,v]
        final_data = []

        try:
            with open(filepath, 'r') as f:
                for line in f:
                    if line.startswith('#'): continue
                    values = line.split()
                    if not values: continue
                    
                    if values[0] == 'v':
                        temp_vertices.append([float(values[1]), float(values[2]), float(values[3])])
                    elif values[0] == 'vt':
                        # Lật trục V (Y) lại vì tọa độ ảnh của OpenGL ngược với OBJ
                        temp_texcoords.append([float(values[1]), 1.0 - float(values[2])])
                    elif values[0] == 'vn':
                        temp_normals.append([float(values[1]), float(values[2]), float(values[3])])
                    elif values[0] == 'f':
                        # Kenney Models luôn dùng mặt tam giác (3 đỉnh)
                        for v in values[1:4]:
                            w = v.split('/')
                            
                            # Tọa độ không gian (v)
                            idx = int(w[0]) - 1
                            vx, vy, vz = temp_vertices[idx]
                            
                            # Tọa độ vân ảnh (vt - UV)
                            if len(w) > 1 and w[1]:
                                t_idx = int(w[1]) - 1
                                tu, tv = temp_texcoords[t_idx]
                            else:
                                tu, tv = 0.0, 0.0
                                
                            # Pháp tuyến ánh sáng (vn - Normal)
                            if len(w) > 2 and w[2]:
                                n_idx = int(w[2]) - 1
                                nx, ny, nz = temp_normals[n_idx]
                            else:
                                nx, ny, nz = 0.0, 1.0, 0.0 
                            
                            final_data.extend([vx, vy, vz, nx, ny, nz, tu, tv])
                            self.vertex_count += 1
                            
            self.mesh_data = np.array(final_data, dtype=np.float32)
            print(f"[OK] Đã load thành công: {filepath} ({self.vertex_count // 3} tam giác)")
            
        except Exception as e:
            print(f"[LỖI] Không thể đọc file {filepath}: {e}")
            self.mesh_data = np.array([], dtype=np.float32)

    def _setup_gl(self):
        if len(self.mesh_data) == 0: return
        
        self.vao = GL.glGenVertexArrays(1)
        self.vbo = GL.glGenBuffers(1)
        
        GL.glBindVertexArray(self.vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, self.mesh_data.nbytes, self.mesh_data, GL.GL_STATIC_DRAW)
        
        # 1 đỉnh có 8 số (3 pos + 3 norm + 2 tex), mỗi số float 4 bytes -> Stride = 32 bytes
        stride = 8 * 4 
        
        # Position attribute (layout location = 0 trong main.vert)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, GL.ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(0)
        
        # Normal attribute (layout location = 1 trong main.vert)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, GL.ctypes.c_void_p(3 * 4))
        GL.glEnableVertexAttribArray(1)
        
        # TexCoord attribute (layout location = 2 trong main.vert)
        GL.glVertexAttribPointer(2, 2, GL.GL_FLOAT, GL.GL_FALSE, stride, GL.ctypes.c_void_p(6 * 4))
        GL.glEnableVertexAttribArray(2)
        
        GL.glBindVertexArray(0)

    def draw(self):
        if self.vao == 0: return
        GL.glBindVertexArray(self.vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.vertex_count)
        GL.glBindVertexArray(0)