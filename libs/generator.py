import OpenGL.GL as GL
from PIL import Image
import numpy as np
import os
import glfw

def save_frame(window, folder_path, filename, mode="RGB"):
    width, height = glfw.get_framebuffer_size(window)
    GL.glReadBuffer(GL.GL_BACK)
    GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
    
    if mode == "RGB":
        data = GL.glReadPixels(0, 0, width, height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
        image = Image.frombytes("RGB", (width, height), data)
    else:
        data = GL.glReadPixels(0, 0, width, height, GL.GL_RED, GL.GL_UNSIGNED_BYTE)
        image = Image.frombytes("L", (width, height), data)
    
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    
    if not os.path.exists(folder_path): 
        os.makedirs(folder_path)
    image.save(os.path.join(folder_path, filename))


def get_2d_bbox(obj, view_matrix, projection_matrix, win_size):
    from libs.transform import translate, scale, rotate_x, rotate_y
    
    v_min, v_max = -1.0, 1.0 
    corners_local = [
        np.array([x, y, z, 1.0]) 
        for x in [v_min, v_max] 
        for y in [v_min, v_max] 
        for z in [v_min, v_max]
    ]

    m_scale = scale(obj.scale, obj.scale, obj.scale)
    m_rot_x = rotate_x(obj.rot_x)
    m_rot_y = rotate_y(obj.rot_y)
    m_trans = translate(obj.pos_x, obj.pos_y, obj.pos_z)

    model_matrix = m_trans @ m_rot_y @ m_rot_x @ m_scale

    mvp = projection_matrix @ view_matrix @ model_matrix
    
    px, py = [], []
    
    for corner in corners_local:
        clip_coords = mvp @ corner
        
        if clip_coords[3] <= 0.1:
            return None
            
        ndc = clip_coords[:3] / clip_coords[3]
        
        screen_x = (ndc[0] + 1.0) * 0.5
        screen_y = (1.0 - ndc[1]) * 0.5 
        
        px.append(screen_x)
        py.append(screen_y)

    x_min, x_max = min(px), max(px)
    y_min, y_max = min(py), max(py)

    if x_max < 0.0 or x_min > 1.0 or y_max < 0.0 or y_min > 1.0:
        return None

    x_min = max(0.0, min(1.0, x_min))
    x_max = max(0.0, min(1.0, x_max))
    y_min = max(0.0, min(1.0, y_min))
    y_max = max(0.0, min(1.0, y_max))

    width = x_max - x_min
    height = y_max - y_min
    x_center = x_min + width / 2.0
    y_center = y_min + height / 2.0

    if width < 0.005 or height < 0.005:
        return None

    return f"{obj.class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"