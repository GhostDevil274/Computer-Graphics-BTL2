import OpenGL.GL as GL
from PIL import Image
import numpy as np
import os
import glfw

def save_frame(window, folder_path, filename, mode="RGB"):
    """ Chụp màn hình (RGB hoặc Grayscale cho Depth/Mask) """
    width, height = glfw.get_framebuffer_size(window)
    GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
    
    if mode == "RGB":
        data = GL.glReadPixels(0, 0, width, height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
        image = Image.frombytes("RGB", (width, height), data)
    else: # Dùng cho Depth Map hoặc Mask
        data = GL.glReadPixels(0, 0, width, height, GL.GL_LUMINANCE, GL.GL_UNSIGNED_BYTE)
        image = Image.frombytes("L", (width, height), data)
    
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    if not os.path.exists(folder_path): os.makedirs(folder_path)
    image.save(os.path.join(folder_path, filename))

def get_2d_bbox(obj, view_matrix, projection_matrix, win_size):
    """
    Tính toán Bounding Box 2D chuẩn YOLO [class_id, x_center, y_center, width, height]
    bằng cách chiếu 8 đỉnh của khối bao (AABB) từ 3D lên 2D.
    """
    # 1. Lấy tọa độ Min/Max của model 3D (giả sử từ -1 đến 1 nếu là cube)
    # Với ObjModel, ông nên lấy từ thuộc tính bbox của nó nếu có
    v_min, v_max = -1.0, 1.0 
    corners_local = [
        np.array([x, y, z, 1.0]) for x in [v_min, v_max] for y in [v_min, v_max] for z in [v_min, v_max]
    ]

    # 2. Tạo ma trận Model của vật thể
    from libs.transform import translate, scale, rotate_x, rotate_y
    m_trans = translate(obj.pos_x, obj.pos_y, obj.pos_z)
    m_rot = np.matmul(rotate_y(obj.rot_y), rotate_x(obj.rot_x))
    m_scale = scale(obj.scale, obj.scale, obj.scale)
    model_matrix = np.matmul(m_trans, np.matmul(m_rot, m_scale))

    # 3. Chiếu các đỉnh lên tọa độ màn hình
    mvp = np.matmul(projection_matrix, np.matmul(view_matrix, model_matrix))
    px, py = [], []
    
    for corner in corners_local:
        # Nhân ma trận đưa về NDC (Normalized Device Coordinates)
        clip_coords = np.dot(mvp, corner)
        if clip_coords[3] != 0:
            ndc = clip_coords[:3] / clip_coords[3]
            # Chuyển từ [-1, 1] sang [0, win_size]
            screen_x = (ndc[0] + 1.0) * 0.5 * win_size[0]
            screen_y = (1.0 - ndc[1]) * 0.5 * win_size[1] # Lật Y vì màn hình tính từ trên xuống
            px.append(screen_x); py.append(screen_y)

    if not px: return None

    # 4. Tính toán tọa độ YOLO (Chuẩn hóa về dải 0 -> 1)
    x_min, x_max = min(px), max(px)
    y_min, y_max = min(py), max(py)
    
    # Giới hạn trong khung hình
    x_min, x_max = max(0, x_min), min(win_size[0], x_max)
    y_min, y_max = max(0, y_min), min(win_size[1], y_max)

    dw, dh = 1.0 / win_size[0], 1.0 / win_size[1]
    w, h = (x_max - x_min), (y_max - y_min)
    x_center, y_center = x_min + w/2, y_min + h/2

    return f"{obj.class_id} {x_center*dw:.6f} {y_center*dh:.6f} {w*dw:.6f} {h*dh:.6f}"