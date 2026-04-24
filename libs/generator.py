import OpenGL.GL as GL
from PIL import Image
from PIL import ImageDraw
import numpy as np
import os
import glfw


def get_2d_bbox(obj, view_matrix, projection_matrix, win_size, depth_map=None):
    import numpy as np
    from libs.transform import translate, scale, rotate_x, rotate_y

    if hasattr(obj.shape, 'bbox'):
        min_x, max_x, min_y, max_y, min_z, max_z = obj.shape.bbox
    else:
        min_x, max_x, min_y, max_y, min_z, max_z = -1.0, 1.0, -1.0, 1.0, -1.0, 1.0

    corners_local = [
        np.array([x, y, z, 1.0]) 
        for x in [min_x, max_x] for y in [min_y, max_y] for z in [min_z, max_z]
    ]
    corners_local.append(np.array([0.0, (min_y+max_y)/2.0, 0.0, 1.0]))

    m_scale = scale(obj.scale, obj.scale, obj.scale)
    m_rot_x = rotate_x(obj.rot_x)
    m_rot_y = rotate_y(obj.rot_y)
    m_trans = translate(obj.pos_x, obj.pos_y, obj.pos_z)

    model_matrix = m_trans @ m_rot_y @ m_rot_x @ m_scale
    mvp = projection_matrix @ view_matrix @ model_matrix
    
    px, py = [], []
    behind_count = 0
    visible_points = 0
    
    for i, corner in enumerate(corners_local):
        clip_coords = mvp @ corner
        if clip_coords[3] <= 0.0:
            behind_count += 1
            continue
        
        w = max(clip_coords[3], 0.0001) 
        ndc = clip_coords[:3] / w
        
        screen_x = (ndc[0] + 1.0) * 0.5
        screen_y_draw = (1.0 - ndc[1]) * 0.5   
        screen_y_depth = (ndc[1] + 1.0) * 0.5  
        
        if i < 8: 
            px.append(screen_x)
            py.append(screen_y_draw)
            
        if depth_map is not None:
            sx = int(screen_x * win_size[0])
            sy = int(screen_y_depth * win_size[1])
            
            if 0 <= sx < win_size[0] and 0 <= sy < win_size[1]:
                buffer_depth = depth_map[sy, sx] 
                obj_depth = (ndc[2] + 1.0) * 0.5 
                
                if obj_depth <= buffer_depth + 0.01:
                    visible_points += 1

    if behind_count >= 8: return None 
    
    if depth_map is not None and visible_points == 0:
        return None

    if not px or not py: return None
    x_min, x_max = min(px), max(px)
    y_min, y_max = min(py), max(py)

    if x_max < 0.0 or x_min > 1.0 or y_max < 0.0 or y_min > 1.0: 
        return None
        
    x_min, x_max = max(0.0, min(1.0, x_min)), max(0.0, min(1.0, x_max))
    y_min, y_max = max(0.0, min(1.0, y_min)), max(0.0, min(1.0, y_max))

    width, height = x_max - x_min, y_max - y_min
    x_center, y_center = x_min + width / 2.0, y_min + height / 2.0

    if width < 0.005 or height < 0.005: return None 
    return f"{obj.class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def save_frame(window, folder_path, filename, mode="RGB", active_bboxes=None):
    width, height = glfw.get_framebuffer_size(window)
    GL.glReadBuffer(GL.GL_BACK)
    GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)

    if mode in ["RGB", "MASK"]:
        data = GL.glReadPixels(0, 0, width, height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
        image = Image.frombytes("RGB", (width, height), data)
    else: 
        data = GL.glReadPixels(0, 0, width, height, GL.GL_RED, GL.GL_UNSIGNED_BYTE)
        image = Image.frombytes("L", (width, height), data)
    
    image = image.transpose(Image.FLIP_TOP_BOTTOM)

    if mode == "MASK" and active_bboxes is not None:
        draw = ImageDraw.Draw(image)
        for b in active_bboxes:
            xc, yc, w_b, h_b = b['xc'], b['yc'], b['w'], b['h']
            x1 = (xc - w_b / 2.0) * width
            y1 = (yc - h_b / 2.0) * height
            x2 = (xc + w_b / 2.0) * width
            y2 = (yc + h_b / 2.0) * height
            
            r, g, b_col = b['obj'].mask_color
            box_color = (int(r*255), int(g*255), int(b_col*255))
            
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=2)
            
            draw.text((x1 + 3, y1 - 14), b['name'], fill=box_color)

    if not os.path.exists(folder_path): os.makedirs(folder_path)
    image.save(os.path.join(folder_path, filename))