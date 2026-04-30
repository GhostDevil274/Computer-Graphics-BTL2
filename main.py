import math
import glfw
import OpenGL.GL as GL
import imgui
from imgui.integrations.glfw import GlfwRenderer
import numpy as np
import os
import time
import random
import json
from PIL import Image

from gui import AppGUI
from libs.shader import Shader
from libs.transform import Trackball, scale, rotate_x, rotate_y, translate
from shapes.basic_3d import ObjModel
from libs.generator import save_frame, get_2d_bbox

class SceneObject:

    _inst_count = 1
    
    def __init__(self, shape, name, class_id=0, class_name="Background"):
        self.shape = shape
        self.name = name
        self.class_id = class_id       
        self.class_name = class_name   
        
        # BẢNG MÀU 
        SEMANTIC_COLORS = {
            0: [0.3, 0.3, 0.3],      # Đường (Xám tối)
            2: [0.5, 0.5, 0.5],      # Building (Xám nhạt hơn)
            3: [0.8, 0.8, 0.8],      # Prop (Xám vừa)
            1: [0.1, 0.1, 0.9],      # Police_Car 
            4: [0.8, 0.1, 0.8],      # SUV 
            5: [0.9, 0.1, 0.1],      # Ambulance 
            6: [0.9, 0.6, 0.1],      # Taxi 
            7: [0.2, 0.8, 0.2],      # Truck 
            8: [0.6, 0.0, 0.0],      # Fire_Truck 
            9: [0.4, 0.2, 0.6],      # Van 
            10: [0.1, 0.8, 0.8],     # Sedan 
            11: [0.5, 0.5, 0.5],     # Garbage_Truck 
            12: [1.0, 0.1, 0.4],     # Hatchback_Sports 
            13: [1.0, 1.0, 0.0],     # Race_Car 
            14: [0.0, 1.0, 0.8],     # Race_Future 
            15: [0.3, 0.3, 0.8],     # Sedan_Sports 
            16: [0.7, 0.4, 0.8],     # SUV_Luxury 
            17: [0.6, 0.3, 0.1],     # Truck_Flat 
            18: [0.9, 0.9, 0.9],     # Delivery 
            19: [0.7, 0.7, 0.2],     # Delivery_Flat 
            20: [0.3, 0.3, 0.3],     # Skyscraper
        }
        self.mask_color = SEMANTIC_COLORS.get(self.class_id, [1.0, 1.0, 1.0])

        self.inst_id = SceneObject._inst_count
        SceneObject._inst_count += 1

        # Dịch bit để tạo màu duy nhất cho từng object trong ảnh mask
        self.instance_color = [
            ((self.inst_id >> 16) & 0xFF) / 255.0,
            ((self.inst_id >> 8) & 0xFF) / 255.0,
            (self.inst_id & 0xFF) / 255.0
        ]

        self.scale = 1.0
        self.rot_x, self.rot_y = 0.0, 0.0
        self.pos_x, self.pos_y, self.pos_z = 0.0, 0.0, 0.0
        
        self.render_mode = 1 
        self.flat_color = [0.8, 0.2, 0.3]
        self.texture_id = 0
        self.texture_filepath = ""

        self.wheel_rotation = 0.0
        self.wheel_radius = 0.3 
        self.velocity = 3.0     

    def update_physics(self, dt, speed_multiplier=1.0, all_objs=None, road_tiles=None):
        if self.class_name in ["Background", "Building", "Prop", "Skyscraper"]: 
            return

        if not hasattr(self, 'velocity'): self.velocity = 3.0
        if not hasattr(self, 'wheel_radius'): self.wheel_radius = 0.3
        if not hasattr(self, 'wheel_rotation'): self.wheel_rotation = 0.0

        # Tránh dt lớn -> xe đi nhanh và xuyên object khác
        safe_dt = min(dt, 0.033) # Tương đương 30 FPS
        actual_speed = self.velocity * speed_multiplier
        
        rad = math.radians(self.rot_y)
        dir_x = round(math.sin(rad)) 
        dir_z = round(math.cos(rad))

        can_move = True
        if all_objs is not None:
            for other in all_objs:
                if other is self or other.class_name in ["Background", "Building", "Prop"]: 
                    continue
                if hasattr(other, 'pos_x'):
                    is_same_lane = False
                    if abs(dir_x) == 1:
                        if abs(other.pos_z - self.pos_z) < 0.5: is_same_lane = True
                    else: 
                        if abs(other.pos_x - self.pos_x) < 0.5: is_same_lane = True
                        
                    if is_same_lane:
                        dx = other.pos_x - self.pos_x
                        dz = other.pos_z - self.pos_z
                        dist = math.sqrt(dx**2 + dz**2)
                        
                        if 0.1 < dist < 3.5:
                            dot = (dx * dir_x) + (dz * dir_z) 
                            if dot > 0.85 * dist: # Tạo ra góc nhìn (-31, 31) độ để tránh collapse
                                can_move = False
                                break
        
        if not can_move:
            actual_speed = 0.0 


        self.pos_x += dir_x * actual_speed * safe_dt
        self.pos_z += dir_z * actual_speed * safe_dt
        self.wheel_rotation += (actual_speed * safe_dt / self.wheel_radius) * (180.0 / math.pi)
        

        # giới hạn thành phố (-40, 40) để tele xe tránh spawn và destroy liên tục
        if self.pos_x > 40.0: self.pos_x -= 80.0 
        elif self.pos_x < -40.0: self.pos_x += 80.0
        
        if self.pos_z > 40.0: self.pos_z -= 80.0
        elif self.pos_z < -40.0: self.pos_z += 80.0


def load_texture(filepath):
    base_path = filepath if filepath.startswith("assets") else os.path.join("assets", "textures", filepath)
    possible_paths = [base_path, base_path + ".png", base_path.replace(".png", "") + ".jpg", base_path.replace(".png", "")]
    actual_path = next((p for p in possible_paths if os.path.exists(p)), None)
    
    if not actual_path: return 0
    try:
        img = Image.open(actual_path).convert("RGBA") # (Red, Green, Blue, Alpha) để nạp lên OpenGL an toàn hơn, ảnh không có A auto = 255
        img_data = np.array(img, np.uint8) 
        tex_id = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex_id)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, img.width, img.height, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, img_data)
        GL.glGenerateMipmap(GL.GL_TEXTURE_2D) # tạo mipmap để tạo bản nhỏ hơn khi ở xa -> giảm nhòe và tăng hiệu suất
        return tex_id
    except Exception as e: 
        return 0

def screen_to_world_ray(xpos, ypos, win_w, win_h, view_matrix, proj_matrix):
    ndc_x = (2.0 * xpos) / win_w - 1.0
    ndc_y = 1.0 - (2.0 * ypos) / win_h # đảo y (OpenGL bắt đầu từ dưới, PIL bắt đầu từ trên)

    ray_clip = np.array([ndc_x, ndc_y, -1.0, 1.0])
    inv_proj = np.linalg.inv(proj_matrix)
    ray_eye = np.dot(inv_proj, ray_clip)
    ray_eye = np.array([ray_eye[0], ray_eye[1], -1.0, 0.0])

    inv_view = np.linalg.inv(view_matrix)
    ray_wor = np.dot(inv_view, ray_eye)[:3]
    return inv_view[:3, 3], ray_wor / np.linalg.norm(ray_wor)
           # tọa độ camera, ray direction


def compute_bbox(filepath):
    try:
        with open(filepath, 'r') as f:
            xs, ys, zs = [], [], []
            for line in f:
                if line.startswith('v '): # chỉ dùng vertex, thay vì vt vn f để có bbox chính xác hơn
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        xs.append(float(parts[1]))
                        ys.append(float(parts[2]))
                        zs.append(float(parts[3]))
        if xs: return (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))
    except Exception: pass
    return (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)


def main():
    if not glfw.init(): return
    
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.SAMPLES, 4) 
    
    monitor = glfw.get_primary_monitor()
    video_mode = glfw.get_video_mode(monitor)
    win_w, win_h = int(video_mode.size.width * 1), int(video_mode.size.height * 1)
    window = glfw.create_window(win_w, win_h, "Viewer", None, None)
    glfw.set_window_pos(window, int((video_mode.size.width - win_w)/2), int((video_mode.size.height - win_h)/2))
    glfw.maximize_window(window)

    glfw.make_context_current(window)
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glEnable(GL.GL_MULTISAMPLE) 
    
    imgui.create_context()
    io = imgui.get_io()
    font_loaded = False
    for path in ["/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf", "/System/Library/Fonts/Monaco.ttf"]:
        if os.path.exists(path):
            io.fonts.add_font_from_file_ttf(path, 18.0)
            font_loaded = True
            break
    if not font_loaded: io.font_global_scale = 1.36
        
    imgui.style_colors_dark() 
    impl = GlfwRenderer(window)
    
    dummy_vao = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(dummy_vao)
    
    my_shader = Shader("shaders/main.vert", "shaders/main.frag")
    scene_objects = []
    gui = AppGUI()

    if not hasattr(SceneObject, '_patched'):
        _old_init = SceneObject.__init__
        def _new_init(self, shape, name, class_id=0, class_name="Background"):
            state = random.getstate() 
            _old_init(self, shape, name, class_id, class_name)
            random.setstate(state)    
        SceneObject.__init__ = _new_init
        SceneObject._patched = True

    car_tex = load_texture("car_tex.png")     
    road_tex = load_texture("road_tex.png")   
    building_tex = load_texture("building_tex.png") 
    
    road_straight = ObjModel(os.path.join("assets", "models", "road-straight.obj"))
    road_cross = ObjModel(os.path.join("assets", "models", "road-intersection.obj"))
    
    car_files = [
        "ambulance.obj", "firetruck.obj", "garbage-truck.obj", 
        "hatchback-sports.obj", "police.obj", "race.obj", 
        "race-future.obj", "sedan.obj", "sedan-sports.obj", 
        "suv.obj", "suv-luxury.obj", "taxi.obj", "truck.obj", 
        "truck-flat.obj", "delivery.obj", "delivery-flat.obj", "van.obj"
    ]
    car_models = []
    for m in car_files:
        path = os.path.join("assets", "models", m)
        if os.path.exists(path):
            mdl = ObjModel(path)
            mdl.filename = m 
            mdl.bbox = compute_bbox(path) 
            car_models.append(mdl)
            
    bldg_files = ["building-a.obj", "building-c.obj", "building-d.obj", "building-b.obj", "building-e.obj", "building-f.obj", "building-g.obj", "building-h.obj", "building-i.obj", "building-j.obj", "building-k.obj", "building-l.obj", "building-m.obj", "building-n.obj", "building-skyscraper-a.obj", "building-skyscraper-b.obj", "building-skyscraper-c.obj", "building-skyscraper-d.obj", "building-skyscraper-e.obj"]
    building_models = []
    for m in bldg_files:
        path = os.path.join("assets", "models", m)
        if os.path.exists(path):
            mdl = ObjModel(path)
            mdl.filename = m
            mdl.bbox = compute_bbox(path) 
            building_models.append(mdl)

    prop_files = ["sign-highway.obj", "sign-highway-detailed.obj"]
    prop_models = [ObjModel(os.path.join("assets", "models", m)) for m in prop_files if os.path.exists(os.path.join("assets", "models", m))]

    base_block = [
        ['3', '2', '2', '2', '2', '2', '2', '3'],
        ['1', '.', '.', '.', '.', '.', '.', '1'],
        ['1', '.', '.', '.', '.', '.', '.', '1'],
        ['1', '.', '.', '.', '.', '.', '.', '1'],
        ['1', '.', '.', '.', '.', '.', '.', '1'],
        ['1', '.', '.', '.', '.', '.', '.', '1'],
        ['3', '2', '2', '2', '2', '2', '2', '3']
    ]

    CITY_LAYOUT = []
    num_blocks_x, num_blocks_z = 4, 4
    for r_b in range(num_blocks_z):
        for r_tile in base_block:
            CITY_LAYOUT.append(r_tile * num_blocks_x)

    TILE_SIZE = 1.3 
    grid_rows, grid_cols = len(CITY_LAYOUT), len(CITY_LAYOUT[0])
    offset_x = grid_cols / 2.0 - 0.5
    offset_z = grid_rows / 2.0 - 0.5
    # Đẩy tâm của bản đồ về gốc tọa độ (0, 0) -> quản lý objects dễ

    open_space_tiles, road_tiles = [], []      
    
    for row_idx, row in enumerate(CITY_LAYOUT):
        for col_idx, tile in enumerate(row):
            x_pos = (col_idx - offset_x) * TILE_SIZE
            z_pos = (row_idx - offset_z) * TILE_SIZE
            
            if tile == '1': 
                road = SceneObject(road_straight, f"Road_V_{row_idx}_{col_idx}", 0, "Background")
                road.pos_x, road.pos_z, road.scale = x_pos, z_pos, 1.3
                road.texture_id = road_tex
                scene_objects.append(road)
                road_tiles.append((x_pos, z_pos, 'V'))
            elif tile == '2': 
                road = SceneObject(road_straight, f"Road_H_{row_idx}_{col_idx}", 0, "Background")
                road.pos_x, road.pos_z, road.rot_y, road.scale = x_pos, z_pos, 90.0, 1.3
                road.texture_id = road_tex
                scene_objects.append(road)
                road_tiles.append((x_pos, z_pos, 'H'))
            elif tile == '3': 
                road = SceneObject(road_cross, f"Cross_{row_idx}_{col_idx}", 0, "Background")
                road.pos_x, road.pos_z, road.scale = x_pos, z_pos, 1.3
                road.texture_id = road_tex
                scene_objects.append(road)
                road_tiles.append((x_pos, z_pos, 'C')) 
            elif tile == '.': 
                open_space_tiles.append((x_pos, z_pos))

    
    def is_overlap(x1, z1, w1, h1, x2, z2, w2, h2):
        # Thuật toán AABB (Axis-Aligned Bounding Box) để kiểm tra va chạm giữa hai hình chữ nhật
        # Nếu khoảng cách giữa tâm hai hình nhỏ hơn tổng nửa chiều rộng và nửa chiều cao -> chồng lên nhau
        return (abs(x1 - x2) < (w1 + w2) / 2.0) and (abs(z1 - z2) < (h1 + h2) / 2.0)

    def get_car_info(filename):
        if not filename: return 0.3, 0.8, 1.6 
        if "firetruck" in filename or "garbage-truck" in filename or "truck" in filename:
            return 0.38, 1.1, 2.8 
        elif "suv" in filename or "van" in filename or "ambulance" in filename:
            return 0.34, 0.95, 2.0 
        else:
            return 0.30, 0.8, 1.6 

    def check_car_collision(new_x, new_z, new_w, new_h, new_rot, placed_cars):
        if new_rot in [90.0, -90.0]: new_w, new_h = new_h, new_w # Đổi chiều nếu xe quay ngang
        for pc in placed_cars:
            pw, ph = pc['w'], pc['h']
            if pc['rot'] in [90.0, -90.0]: pw, ph = ph, pw
            if is_overlap(new_x, new_z, new_w + 0.2, new_h + 0.2, pc['x'], pc['z'], pw, ph):
                return True
        return False

    for rx, rz, rdir in road_tiles:
        if not prop_models: break
        if rdir == 'C': continue
        if random.random() > 0.15: continue 
        
        safe_props = [p for p in prop_models if "sign" not in getattr(p, 'filename', '')]
        if not safe_props: safe_props = prop_models
        
        prop = SceneObject(random.choice(safe_props), "Street_Prop", 3, "Prop")
        prop.texture_id = building_tex 

        edge_offset = TILE_SIZE * 1.9
        
        if rdir == 'V':
            prop.pos_x = rx + edge_offset * random.choice([1, -1])
            prop.pos_z = rz + random.uniform(-0.3, 0.3)
            prop.rot_y = 90.0 if prop.pos_x > rx else -90.0
        elif rdir == 'H':
            prop.pos_x = rx + random.uniform(-0.3, 0.3)
            prop.pos_z = rz + edge_offset * random.choice([1, -1])
            prop.rot_y = 0.0 if prop.pos_z > rz else 180.0
        else:
            prop.pos_x = rx + edge_offset * random.choice([1, -1])
            prop.pos_z = rz + edge_offset * random.choice([1, -1])
            prop.rot_y = random.choice([45.0, 135.0, 225.0, 315.0])
            
        scene_objects.append(prop)

    num_cars_to_place = 70 
    placed_cars_data = [] 
    
    for i in range(num_cars_to_place):
        if not car_models: break
        
        rt_x, rt_z, rt_dir = random.choice(road_tiles)
        if rt_dir == 'C': 
            continue

        class_mapping = {
            'police': ("Police_Car", 1),
            'suv': ("SUV", 4),
            'ambulance': ("Ambulance", 5),
            'taxi': ("Taxi", 6),
            'truck': ("Truck", 7),
            'firetruck': ("Fire_Truck", 8),
            'van': ("Van", 9),
            'sedan': ("Sedan", 10),
            'garbage-truck': ("Garbage_Truck", 11),
            'hatchback-sports': ("Hatchback_Sports", 12),
            'race': ("Race_Car", 13),
            'race-future': ("Race_Future", 14),
            'sedan-sports': ("Sedan_Sports", 15),
            'suv-luxury': ("SUV_Luxury", 16),
            'truck-flat': ("Truck_Flat", 17),
            'delivery': ("Delivery", 18),
            'delivery-flat': ("Delivery_Flat", 19)
        }

        car_model = random.choice(car_models)

        if not hasattr(car_model, 'bbox'): 
            car_model.bbox = compute_bbox(os.path.join("assets", "models", getattr(car_model, 'filename', '')))
        
        name_lower = getattr(car_model, 'filename', '').lower()
        c_name, c_id = "Sedan", 10 
        for k, v in class_mapping.items():
            if k in name_lower:
                c_name, c_id = v
                break

        car = SceneObject(car_model, f"{c_name}_{i}", c_id, c_name)
        
        base_car_scale, base_w, base_h = get_car_info(getattr(car_model, 'filename', ''))
        car.scale = base_car_scale * random.uniform(0.95, 1.05) 
        car_w, car_h = base_w * car.scale, base_h * car.scale
        
        offset = 0.30 * TILE_SIZE 
        direction = random.choice([1, -1]) 
        
        if rt_dir == 'V':
            if direction == 1:
                car.pos_x = rt_x + offset
                car.rot_y = 180.0
            else:
                car.pos_x = rt_x - offset
                car.rot_y = 0.0
            car.pos_z = rt_z + random.uniform(-2.0, 2.0)
        else:
            if direction == 1:
                car.pos_z = rt_z + offset
                car.rot_y = 90.0
            else:
                car.pos_z = rt_z - offset
                car.rot_y = -90.0
            car.pos_x = rt_x + random.uniform(-2.0, 2.0)
            
        if not check_car_collision(car.pos_x, car.pos_z, car_w, car_h, car.rot_y, placed_cars_data):
            car.texture_id = car_tex
            scene_objects.append(car)
            placed_cars_data.append({'x': car.pos_x, 'z': car.pos_z, 'w': car_w, 'h': car_h, 'rot': car.rot_y})

    for i, (tile_x, tile_z) in enumerate(open_space_tiles):
        if not building_models: break
        bldg = SceneObject(random.choice(building_models), f"Bldg_{i}", 2, "Building")
        bldg.pos_x = tile_x + random.uniform(-0.1, 0.1) 
        bldg.pos_z = tile_z + random.uniform(-0.1, 0.1)
        bldg.rot_y = random.choice([0.0, 90.0, 180.0, -90.0])
        
        target_scale = random.uniform(1.8, 2.3)
        b_w = 2.0 * target_scale
        
        overlap = False
        for rx, rz, _ in road_tiles:
            if is_overlap(bldg.pos_x, bldg.pos_z, b_w, b_w, rx, rz, TILE_SIZE, TILE_SIZE):
                overlap = True; break
        
        if not overlap:
            bldg.scale = target_scale
            bldg.texture_id = building_tex
            scene_objects.append(bldg)

    if len(car_models) > 0:
        ego_x = (0 - offset_x) * TILE_SIZE + 0.25 * TILE_SIZE
        ego_z = (5 - offset_z) * TILE_SIZE
        ego_model = car_models[0] 
        ego_car = SceneObject(ego_model, "Ego_Dashcam_Car", 10, "Sedan")
        ego_base_scale, _, _ = get_car_info(getattr(ego_model, 'filename', ''))
        ego_car.scale = ego_base_scale
        ego_car.pos_x, ego_car.pos_z, ego_car.rot_y = ego_x, ego_z, 180.0
        ego_car.texture_id = car_tex
        scene_objects.append(ego_car)

    # Thiết lập camera và callback chuột 
    cameras = []
    cam0 = Trackball(distance=15.0); cam0.azimuth = 0.0; cam0.elevation = 10.0
    cam1 = Trackball(distance=40.0); cam1.azimuth = 0.0; cam1.elevation = 80.0
    cam2 = Trackball(distance=25.0, target=[-10.0, 0.0, -10.0]); cam2.azimuth = 45.0; cam2.elevation = 30.0
    cam3 = Trackball(distance=25.0, target=[10.0, 0.0, 10.0]); cam3.azimuth = -45.0; cam3.elevation = 30.0
    cam4 = Trackball(distance=30.0, target=[-20.0, 0.0, 0.0]); cam4.azimuth=90.0; cam4.elevation=25.0
    cam5 = Trackball(distance=30.0, target=[0.0, 0.0, -20.0]); cam5.azimuth=180.0; cam5.elevation=25.0
    cam6 = Trackball(distance=60.0, target=[0.0, 0.0, 0.0]); cam6.azimuth=135.0; cam6.elevation=45.0
    cam7 = Trackball(distance=8.0, target=[5.0, 0.0, 5.0]); cam7.azimuth=-30.0; cam7.elevation=5.0

    cameras = [cam0, cam1, cam2, cam3, cam4, cam5, cam6, cam7]
    if len(car_models) > 0: 
        cameras[0].target = np.array([ego_x, 0.2, ego_z], dtype=np.float32)

    def on_mouse_click(win, button, action, mods):
        if imgui.get_io().want_capture_mouse: return
        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
            if mods & glfw.MOD_SHIFT or mods & glfw.MOD_CONTROL: return

            xpos, ypos = glfw.get_cursor_pos(win)
            ww, wh = glfw.get_window_size(win)
            cur_cam = cameras[gui.selected_cam_idx]
            v_mat, p_mat = cur_cam.view_matrix(), cur_cam.projection_matrix((ww, wh))
            ray_orig, ray_dir = screen_to_world_ray(xpos, ypos, ww, wh, v_mat, p_mat)
            
            closest_dist, selected_idx = float('inf'), -1 #closest_dist đảm bảo chọn xe gần nếu có 2 xe đè nhau
            for i, obj in enumerate(scene_objects):
                t = np.dot(np.array([obj.pos_x, obj.pos_y, obj.pos_z]) - ray_orig, ray_dir) # tính khoảng cách từ mắt cam đến điểm trên ray gần nhất với tâm obj
                if t > 0: 
                    dist = np.linalg.norm(np.array([obj.pos_x, obj.pos_y, obj.pos_z]) - (ray_orig + t * ray_dir)) # khoảng cách từ điểm trên ray đến tâm obj
                    if dist < 1.0 * obj.scale and t < closest_dist:
                        closest_dist, selected_idx = t, i
            if selected_idx != -1: gui.selected_scene_obj_idx = selected_idx
    glfw.set_mouse_button_callback(window, on_mouse_click)

    def on_mouse_move(win, xpos, ypos):
        if not hasattr(on_mouse_move, "old_pos"): on_mouse_move.old_pos = (xpos, ypos)
        dx, dy = xpos - on_mouse_move.old_pos[0], ypos - on_mouse_move.old_pos[1]
        on_mouse_move.old_pos = (xpos, ypos)
        if imgui.get_io().want_capture_mouse: return
        
        is_shift = glfw.get_key(win, glfw.KEY_LEFT_SHIFT) == glfw.PRESS # biến check để xoay/dời object

        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT): # xoay object
            if is_shift and len(scene_objects) > 0:
                scene_objects[gui.selected_scene_obj_idx].rot_y += dx * 0.2
                scene_objects[gui.selected_scene_obj_idx].rot_x += dy * 0.2
            else: # xoay camera
                cameras[gui.selected_cam_idx].azimuth -= dx * 0.2
                cameras[gui.selected_cam_idx].elevation = max(-89.0, min(89.0, cameras[gui.selected_cam_idx].elevation - dy * 0.2))

        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_RIGHT): # dời object
            if is_shift and len(scene_objects) > 0:
                cam_az = np.radians(cameras[gui.selected_cam_idx].azimuth)
                scene_objects[gui.selected_scene_obj_idx].pos_x += dx * 0.015 * np.cos(cam_az)
                scene_objects[gui.selected_scene_obj_idx].pos_z += dx * 0.015 * np.sin(cam_az)
                scene_objects[gui.selected_scene_obj_idx].pos_y -= dy * 0.015
            else: # dời camera
                cameras[gui.selected_cam_idx].pan((xpos, ypos), (xpos - dx, ypos - dy))
                gui.spawn_pos = list(cameras[gui.selected_cam_idx].target)
    glfw.set_cursor_pos_callback(window, on_mouse_move)

    def on_scroll(win, dx, dy):
        impl.scroll_callback(win, dx, dy)
        if not imgui.get_io().want_capture_mouse: 
            if glfw.get_key(win, glfw.KEY_LEFT_CONTROL) == glfw.PRESS and len(scene_objects) > 0:
                scene_objects[gui.selected_scene_obj_idx].scale = max(0.1, scene_objects[gui.selected_scene_obj_idx].scale + dy * 0.1)
            else: 
                cameras[gui.selected_cam_idx].zoom(dy, glfw.get_window_size(win)[1])
    glfw.set_scroll_callback(window, on_scroll)

    def render_scene(v_mat, p_mat, view_mode, shader):
        shader.use()
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(shader.render_idx, "view"), 1, GL.GL_TRUE, v_mat) # GL_TRUE để truyền column theo OpenGL convention thay vì row-major của numpy
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(shader.render_idx, "projection"), 1, GL.GL_TRUE, p_mat)
        GL.glUniform1i(GL.glGetUniformLocation(shader.render_idx, "is_depth_map"), 1 if view_mode == 1 else 0)
        GL.glUniform1i(GL.glGetUniformLocation(shader.render_idx, "is_mask_map"), 1 if view_mode in [2, 3] else 0)
        
        if shader == my_shader:
            lights_active = gui.lights if view_mode == 0 else [False, False, False]
            GL.glUniform1i(GL.glGetUniformLocation(shader.render_idx, "light1_on"), lights_active[0])
            GL.glUniform1i(GL.glGetUniformLocation(shader.render_idx, "light2_on"), lights_active[1])
            GL.glUniform1i(GL.glGetUniformLocation(shader.render_idx, "light3_on"), lights_active[2])
            GL.glUniform3f(GL.glGetUniformLocation(shader.render_idx, "viewPos"), *np.linalg.inv(v_mat)[:3, 3]) # nghịch đảo view matrix để lấy vị trí camera trong world space

        for obj in scene_objects:
            if obj.class_name not in ["Background", "Building", "Prop", "Skyscraper"]:
                # Kỹ thuật Culling, loại bỏ các objects nằm ngoài phạm vi này để tăng efficiency 
                if abs(obj.pos_x) > 22.0 or abs(obj.pos_z) > 19.5: 
                    continue

            # model = Translation * Rotation * Scale (nhân từ phải sang trái)
            m_trans = translate(obj.pos_x, obj.pos_y, obj.pos_z)
            m_rot = np.matmul(rotate_y(obj.rot_y), rotate_x(obj.rot_x))
            m_scale = scale(obj.scale, obj.scale, obj.scale)
            model_matrix = np.matmul(m_trans, np.matmul(m_rot, m_scale))
            GL.glUniformMatrix4fv(GL.glGetUniformLocation(shader.render_idx, "model"), 1, GL.GL_TRUE, model_matrix)
            
            if view_mode == 2: 
                GL.glUniform1i(GL.glGetUniformLocation(shader.render_idx, "render_mode"), 0)
                GL.glUniform3f(GL.glGetUniformLocation(shader.render_idx, "flat_color"), *obj.mask_color)
            elif view_mode == 3:
                GL.glUniform1i(GL.glGetUniformLocation(shader.render_idx, "render_mode"), 0)
                GL.glUniform3f(GL.glGetUniformLocation(shader.render_idx, "flat_color"), *obj.instance_color)
            else: 
                GL.glUniform1i(GL.glGetUniformLocation(shader.render_idx, "render_mode"), 4 if obj.texture_id > 0 else 3)
                if obj.texture_id > 0 and view_mode == 0:
                    GL.glActiveTexture(GL.GL_TEXTURE0)
                    GL.glBindTexture(GL.GL_TEXTURE_2D, obj.texture_id)

            # vẽ object        
            obj.shape.draw()

    
    last_time = glfw.get_time()

    while not glfw.window_should_close(window):
        current_time = glfw.get_time()
        dt = current_time - last_time
        last_time = current_time

        glfw.poll_events()
        impl.process_inputs()

        speed_mult = getattr(gui, 'car_speed', 1.0)
        for obj in scene_objects:
            if hasattr(obj, 'update_physics'):
                obj.update_physics(dt, speed_mult, scene_objects, road_tiles)

        ww, wh = glfw.get_window_size(window)
        current_cam = cameras[gui.selected_cam_idx]
        v_mat, p_mat = current_cam.view_matrix(), current_cam.projection_matrix((ww, wh))

        if getattr(gui, 'delete_obj_requested', False) and len(scene_objects) > 0:
            scene_objects.pop(gui.selected_scene_obj_idx)
            gui.selected_scene_obj_idx = max(0, gui.selected_scene_obj_idx - 1) 
            gui.delete_obj_requested = False
            
        if getattr(gui, 'duplicate_obj_requested', False) and len(scene_objects) > 0:
            old_obj = scene_objects[gui.selected_scene_obj_idx]
            new_obj = SceneObject(old_obj.shape, old_obj.name + " (Copy)", old_obj.class_id, old_obj.class_name)
            new_obj.scale, new_obj.rot_x, new_obj.rot_y = old_obj.scale, old_obj.rot_x, old_obj.rot_y
            new_obj.pos_x, new_obj.pos_y, new_obj.pos_z = old_obj.pos_x + 2.0, old_obj.pos_y, old_obj.pos_z
            new_obj.texture_id, new_obj.texture_filepath = old_obj.texture_id, old_obj.texture_filepath
            scene_objects.append(new_obj)
            gui.selected_scene_obj_idx = len(scene_objects) - 1
            gui.duplicate_obj_requested = False

        if getattr(gui, 'add_shape_requested', False):
            obj_path = os.path.join("assets", "models", gui.obj_filepath) if not gui.obj_filepath.startswith("assets") else gui.obj_filepath
            new_shape = ObjModel(filepath=obj_path)
            obj = SceneObject(new_shape, f"#{len(scene_objects)+1} {gui.class_name}", gui.class_id, gui.class_name)
            
            kw_vehicles = ["car", "vehicle", "police", "taxi", "truck", "suv", "van", "ambulance", "firetruck", "sedan", "hatchback"]
            kw_buildings = ["build", "bldg", "house", "tower", "skyscraper"]
            kw_props = ["sign", "light"] 
            
            is_vehicle = any(kw in gui.class_name.lower() or kw in gui.obj_filepath.lower() for kw in kw_vehicles)
            is_building = any(kw in gui.class_name.lower() or kw in gui.obj_filepath.lower() for kw in kw_buildings)
            is_prop = any(kw in gui.class_name.lower() or kw in gui.obj_filepath.lower() for kw in kw_props)

            cam_target = current_cam.target
            spawn_x, spawn_y, spawn_z = cam_target[0], cam_target[1], cam_target[2]
            
            best_dist, best_road = float('inf'), None

            # Tìm tile đường gần nhất với vị trí spawn
            for rx, rz, rdir in road_tiles:
                d = (rx - spawn_x)**2 + (rz - spawn_z)**2
                if d < best_dist:
                    best_dist = d; best_road = (rx, rz, rdir)
                    
            if best_road and best_dist < 15.0: 
                rx, rz, rdir = best_road
                offset = 0.25 * TILE_SIZE
                obj.pos_y = 0.0
                
                if is_vehicle:
                    if rdir == 'V':
                        obj.pos_x = rx + offset if spawn_x > rx else rx - offset
                        obj.pos_z = spawn_z
                        obj.rot_y = 0.0 if obj.pos_x > rx else 180.0
                    elif rdir == 'H':
                        obj.pos_x = spawn_x
                        obj.pos_z = rz + offset if spawn_z > rz else rz - offset
                        obj.rot_y = -90.0 if obj.pos_z > rz else 90.0
                    else:
                        obj.pos_x, obj.pos_z = rx, rz
                        obj.rot_y = random.choice([0.0, 90.0, 180.0, -90.0])

                    s_factor, bw, bh = get_car_info(os.path.basename(obj_path))
                    cw, ch = bw * s_factor, bh * s_factor
                    
                    for _ in range(20): 
                        collision = False
                        for other in scene_objects:
                            if other.class_id == 1: 
                                ow, oh = 0.8 * other.scale, 2.0 * other.scale 
                                if other.rot_y in [90.0, -90.0]: ow, oh = oh, ow
                                test_cw, test_ch = cw, ch
                                if obj.rot_y in [90.0, -90.0]: test_cw, test_ch = ch, cw
                                
                                if is_overlap(obj.pos_x, obj.pos_z, test_cw + 0.3, test_ch + 0.3, other.pos_x, other.pos_z, ow, oh):
                                    collision = True
                                    break
                        if not collision: break 
                        
                        if rdir == 'V': obj.pos_z += 1.5
                        elif rdir == 'H': obj.pos_x += 1.5
                        else: obj.pos_z += 1.5
                else:
                    obj.pos_x, obj.pos_z = spawn_x, spawn_z
            else:
                obj.pos_x, obj.pos_y, obj.pos_z = spawn_x, spawn_y, spawn_z

            if is_vehicle:
                s, _, _ = get_car_info(os.path.basename(obj_path))
                obj.scale = s
                obj.texture_id = car_tex
            elif is_building:
                obj.texture_id = building_tex
            elif is_prop:
                obj.texture_id = car_tex 
            
            scene_objects.append(obj)
            gui.selected_scene_obj_idx = len(scene_objects) - 1
            gui.add_shape_requested = False
            
        if getattr(gui, 'clear_scene_requested', False):
            scene_objects = []
            gui.clear_scene_requested = False
            gui.selected_scene_obj_idx = 0

        fb_width, fb_height = glfw.get_framebuffer_size(window)
        GL.glViewport(0, 0, fb_width, fb_height)

        is_generating = getattr(gui, 'generate_requested', False)
        is_showing_bbox = getattr(gui, 'show_bbox', False)
        active_bboxes = [] 

        if is_generating or is_showing_bbox:
            GL.glDisable(GL.GL_MULTISAMPLE) 
            # Tắt khử răng cưa để đảm bảo màu sắc chính xác khi đọc pixel ở mask instance

            GL.glClearColor(0.0, 0.0, 0.0, 1.0) # Ép màu đen tuyệt đối
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            render_scene(v_mat, p_mat, 3, my_shader) 
            
            inst_data = GL.glReadPixels(0, 0, fb_width, fb_height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
            inst_arr = np.frombuffer(inst_data, dtype=np.uint8).reshape((fb_height, fb_width, 3))
            inst_arr = np.flipud(inst_arr)
            GL.glEnable(GL.GL_MULTISAMPLE)

            inst_arr_int = (inst_arr[:,:,0].astype(np.uint32) << 16) | (inst_arr[:,:,1].astype(np.uint32) << 8) | inst_arr[:,:,2].astype(np.uint32)

            for obj in scene_objects:
                if obj.class_name not in ["Background", "Building", "Prop", "Skyscraper"]:
                    if abs(obj.pos_x) > 21.0 or abs(obj.pos_z) > 18.5: continue
                    
                    r_val = (obj.inst_id >> 16) & 0xFF
                    g_val = (obj.inst_id >> 8) & 0xFF
                    b_val = obj.inst_id & 0xFF
                    target_color = (r_val << 16) | (g_val << 8) | b_val
                    
                    y_coords, x_coords = np.nonzero(inst_arr_int == target_color)

                    if y_coords.size > 50: 
                        y_min, x_min = y_coords.min(), x_coords.min()
                        y_max, x_max = y_coords.max(), x_coords.max()
                        w_box = (x_max - x_min) / fb_width
                        h_box = (y_max - y_min) / fb_height
                        xc = (x_min + x_max) / 2.0 / fb_width
                        yc = (y_min + y_max) / 2.0 / fb_height
                        
                        if w_box > 0.005 and h_box > 0.005:
                            active_bboxes.append({
                                "obj": obj, "class_id": obj.class_id, "name": obj.name,
                                "xc": xc, "yc": yc, "w": w_box, "h": h_box
                            })

        if is_generating:
            file_id = f"frame_{int(time.time())}"
            for d in ["dataset/images", "dataset/labels", "dataset/masks", "dataset/depth"]:
                os.makedirs(d, exist_ok=True) 

            bg = gui.bg_color if getattr(gui, 'view_mode', 0) == 0 else [0.0, 0.0, 0.0]
            GL.glClearColor(*bg, 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            render_scene(v_mat, p_mat, 0, my_shader)
            save_frame(window, "dataset/images", f"frame_{file_id}.png", mode="RGB")
            
            GL.glClearColor(0.0, 0.0, 0.0, 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            render_scene(v_mat, p_mat, 1, my_shader)
            save_frame(window, "dataset/depth", f"frame_{file_id}_depth.png", mode="L")
            
            GL.glClearColor(0.0, 0.0, 0.0, 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            render_scene(v_mat, p_mat, 2, my_shader)
            save_frame(window, "dataset/masks", f"frame_{file_id}_mask.png", mode="MASK", active_bboxes=active_bboxes)

            labels_txt = [f"{b['class_id']} {b['xc']:.6f} {b['yc']:.6f} {b['w']:.6f} {b['h']:.6f}" for b in active_bboxes]
            with open(f"dataset/labels/{file_id}.txt", "w") as f:
                f.write("\n".join(labels_txt))

            metadata = {"camera": {"view_matrix": v_mat.tolist(), "projection_matrix": p_mat.tolist()}, "objects": []}
            for b in active_bboxes:
                o = b['obj']
                metadata["objects"].append({"name": o.name, "class": o.class_name, "position_3d": [o.pos_x, o.pos_y, o.pos_z], "rotation_y": o.rot_y})
            with open(f"dataset/labels/{file_id}_meta.json", "w") as f: json.dump(metadata, f, indent=4)

            print(f"✅ Đã xuất xong bộ dữ liệu chuẩn: {file_id}")
            gui.generate_requested = False

        bg = gui.bg_color if getattr(gui, 'view_mode', 0) == 0 else [0.0, 0.0, 0.0]
        GL.glClearColor(*bg, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        
        my_shader.use()
        lights_active = gui.lights if getattr(gui, 'view_mode', 0) == 0 else [False, False, False]
        GL.glUniform1i(GL.glGetUniformLocation(my_shader.render_idx, "light1_on"), lights_active[0])
        GL.glUniform1i(GL.glGetUniformLocation(my_shader.render_idx, "light2_on"), lights_active[1])
        GL.glUniform1i(GL.glGetUniformLocation(my_shader.render_idx, "light3_on"), lights_active[2])
        GL.glUniform3f(GL.glGetUniformLocation(my_shader.render_idx, "viewPos"), *np.linalg.inv(v_mat)[:3, 3])
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE if getattr(gui, 'is_wireframe', False) else GL.GL_FILL)

        render_scene(v_mat, p_mat, getattr(gui, 'view_mode', 0), my_shader)
        
        gui.render(cameras, scene_objects)

        if is_showing_bbox:
            draw_list = imgui.get_background_draw_list()
            ww, wh = glfw.get_window_size(window)
            
            for b in active_bboxes:
                x_c, y_c, w_b, h_b = b['xc'], b['yc'], b['w'], b['h']
                x1 = (x_c - w_b / 2.0) * ww
                y1 = (y_c - h_b / 2.0) * wh
                x2 = (x_c + w_b / 2.0) * ww
                y2 = (y_c + h_b / 2.0) * wh

                r, g, b_col = b['obj'].mask_color
                box_color = imgui.get_color_u32_rgba(r, g, b_col, 1.0)
                
                draw_list.add_rect(x1, y1, x2, y2, box_color, thickness=2.0)
                bg_color = imgui.get_color_u32_rgba(0.0, 0.0, 0.0, 0.6)
                draw_list.add_rect_filled(x1, y1 - 18, x1 + 80, y1, bg_color) 
                draw_list.add_text(x1 + 4, y1 - 17, box_color, f"{b['obj'].class_name}")

        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

    impl.shutdown()
    glfw.terminate()

if __name__ == "__main__":
    main()