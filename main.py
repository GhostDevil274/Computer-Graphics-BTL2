import glfw
import OpenGL.GL as GL
import imgui
from imgui.integrations.glfw import GlfwRenderer
import numpy as np
import os
import random
from PIL import Image

from btl2_loader import KenneyModel
from gui import AppGUI
from libs.shader import Shader
from libs.transform import Trackball, scale, rotate_x, rotate_y, translate
from shapes.basic_3d import ObjModel

class SceneObject:
    def __init__(self, shape, name, class_id=0, class_name="Background"):
        self.shape = shape
        self.name = name
        self.class_id = class_id       
        self.class_name = class_name   
        
        random.seed(self.class_id + 100)
        self.mask_color = [random.uniform(0.1, 1.0), random.uniform(0.1, 1.0), random.uniform(0.1, 1.0)]

        self.scale = 1.0
        self.rot_x, self.rot_y = 0.0, 0.0
        self.pos_x, self.pos_y, self.pos_z = 0.0, 0.0, 0.0
        
        self.texture_id = 0
        self.texture_filepath = "texture.jpg"

def load_texture(filepath):
    base_path = filepath if filepath.startswith("assets") else os.path.join("assets", "textures", filepath)
    possible_paths = [base_path, base_path + ".png", base_path.replace(".png", "") + ".jpg", base_path.replace(".png", "")]
    actual_path = next((p for p in possible_paths if os.path.exists(p)), None)
    
    if not actual_path: 
        print(f"❌ [LỖI] KHÔNG TÌM THẤY ẢNH: {filepath}")
        return 0
        
    try:
        img = Image.open(actual_path).convert("RGBA")
        img_data = np.array(img, np.uint8) 
        tex_id = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex_id)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, img.width, img.height, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, img_data)
        GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
        return tex_id
    except Exception as e: 
        print(f"❌ [LỖI] File ảnh bị hỏng: {actual_path} - Lỗi: {e}")
        return 0

def screen_to_world_ray(xpos, ypos, win_w, win_h, view_matrix, proj_matrix):
    ndc_x = (2.0 * xpos) / win_w - 1.0
    ndc_y = 1.0 - (2.0 * ypos) / win_h
    ray_clip = np.array([ndc_x, ndc_y, -1.0, 1.0])
    inv_proj = np.linalg.inv(proj_matrix)
    ray_eye = np.dot(inv_proj, ray_clip)
    ray_eye = np.array([ray_eye[0], ray_eye[1], -1.0, 0.0])
    inv_view = np.linalg.inv(view_matrix)
    ray_wor = np.dot(inv_view, ray_eye)[:3]
    return inv_view[:3, 3], ray_wor / np.linalg.norm(ray_wor)

def main():
    if not glfw.init(): return
    
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.SAMPLES, 4) 
    
    monitor = glfw.get_primary_monitor()
    video_mode = glfw.get_video_mode(monitor)
    win_w, win_h = int(video_mode.size.width * 0.9), int(video_mode.size.height * 0.9)
    window = glfw.create_window(win_w, win_h, "CALAR Data Generator - Smart City", None, None)
    glfw.set_window_pos(window, int((video_mode.size.width - win_w)/2), int((video_mode.size.height - win_h)/2))

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

    print("Đang quy hoạch Đại Đô Thị MỞ RỘNG QUY MÔ (Bản chuẩn: Giữ nhà, Xe thực tế, Đúng 3 Texture)...")
    car_tex = load_texture("car_tex.png")     
    road_tex = load_texture("road_tex.png")   
    building_tex = load_texture("building_tex.png") 
    
    # Models
    road_straight = KenneyModel(os.path.join("assets", "models", "road-straight.obj"))
    road_cross = KenneyModel(os.path.join("assets", "models", "road-intersection.obj"))
    
    car_files = ["sedan.obj", "police.obj", "taxi.obj", "suv.obj", "van.obj", "ambulance.obj", "firetruck.obj", "garbage-truck.obj", "truck.obj"]
    car_models = []
    for m in car_files:
        path = os.path.join("assets", "models", m)
        if os.path.exists(path):
            mdl = KenneyModel(path)
            mdl.filename = m 
            car_models.append(mdl)
            
    bldg_files = ["building-a.obj", "building-c.obj", "building-d.obj", "building-j.obj", "building-k.obj", "building-m.obj", "building-skyscraper-a.obj", "building-skyscraper-b.obj", "building-skyscraper-d.obj", "building-skyscraper-e.obj"]
    building_models = [KenneyModel(os.path.join("assets", "models", m)) for m in bldg_files if os.path.exists(os.path.join("assets", "models", m))]

    # --- A. ĐỊNH NGHĨA KHỐI CƠ BẢN (Khoảng trống 9x9) ---
    base_block = [
        ['3', '2', '2', '2', '2', '2', '2', '2', '3'],
        ['1', '.', '.', '.', '.', '.', '.', '.', '1'],
        ['1', '.', '.', '.', '.', '.', '.', '.', '1'],
        ['1', '.', '.', '.', '.', '.', '.', '.', '1'],
        ['1', '.', '.', '.', '.', '.', '.', '.', '1'],
        ['1', '.', '.', '.', '.', '.', '.', '.', '1'],
        ['1', '.', '.', '.', '.', '.', '.', '.', '1'],
        ['1', '.', '.', '.', '.', '.', '.', '.', '1'],
        ['3', '2', '2', '2', '2', '2', '2', '2', '3']
    ]

    # --- B. NHÂN QUY MÔ LÊN GẤP 4 X 4 LẦN ---
    CITY_LAYOUT = []
    num_blocks_x, num_blocks_z = 4, 4
    for r_b in range(num_blocks_z):
        for r_tile in base_block:
            CITY_LAYOUT.append(r_tile * num_blocks_x)

    TILE_SIZE = 1.3 
    grid_rows, grid_cols = len(CITY_LAYOUT), len(CITY_LAYOUT[0])
    offset_x = grid_cols / 2.0 - 0.5
    offset_z = grid_rows / 2.0 - 0.5

    open_space_tiles = [] 
    road_tiles = []      
    
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
        return (abs(x1 - x2) < (w1 + w2) / 2.0) and (abs(z1 - z2) < (h1 + h2) / 2.0)

    # 1. Hàm tính thông số thực tế của xe
    def get_car_info(filename):
        if not filename: return 0.3, 0.8, 1.6 
        if "firetruck" in filename or "garbage-truck" in filename or "truck" in filename:
            return 0.38, 1.1, 2.8 
        elif "suv" in filename or "van" in filename or "ambulance" in filename:
            return 0.34, 0.95, 2.0 
        else:
            return 0.30, 0.8, 1.6 

    # 2. Hàm check va chạm giữa xe với xe
    def check_car_collision(new_x, new_z, new_w, new_h, new_rot, placed_cars):
        if new_rot in [90.0, -90.0]: new_w, new_h = new_h, new_w
        for pc in placed_cars:
            pw, ph = pc['w'], pc['h']
            if pc['rot'] in [90.0, -90.0]: pw, ph = ph, pw
            if is_overlap(new_x, new_z, new_w + 0.2, new_h + 0.2, pc['x'], pc['z'], pw, ph):
                return True
        return False

    # 3. Hàm check va chạm giữa tòa nhà và mặt đường
    def check_building_road_collision(bldg_x, bldg_z, bldg_scale, road_tiles_list, t_size):
        b_w, b_h = 2.0 * bldg_scale, 2.0 * bldg_scale
        for rx, rz, rtype in road_tiles_list:
            if is_overlap(bldg_x, bldg_z, b_w, b_h, rx, rz, t_size, t_size):
                return True
        return False

    print("Đang thả xe tấp nập với kích thước chuẩn...")
    num_cars_to_place = 100 
    placed_cars_data = [] 
    
    for i in range(num_cars_to_place):
        if not car_models: break
        rt_x, rt_z, rt_dir = random.choice(road_tiles)
        if rt_dir == 'C': continue 
        
        car_model = random.choice(car_models)
        car = SceneObject(car_model, f"Car_{i}", 1, "Car")
        
        base_car_scale, base_w, base_h = get_car_info(getattr(car_model, 'filename', ''))
        car.scale = base_car_scale * random.uniform(0.95, 1.05) 
        car_w, car_h = base_w * car.scale, base_h * car.scale
        
        offset = 0.25 * TILE_SIZE 
        if rt_dir == 'V':
            car.pos_x = rt_x + offset if random.random() > 0.5 else rt_x - offset
            car.pos_z = rt_z
            car.rot_y = 0.0 if car.pos_x > rt_x else 180.0
        else:
            car.pos_z = rt_z + offset if random.random() > 0.5 else rt_z - offset
            car.pos_x = rt_x
            car.rot_y = -90.0 if car.pos_z > rt_z else 90.0
            
        if not check_car_collision(car.pos_x, car.pos_z, car_w, car_h, car.rot_y, placed_cars_data):
            car.texture_id = car_tex
            scene_objects.append(car)
            placed_cars_data.append({'x': car.pos_x, 'z': car.pos_z, 'w': car_w, 'h': car_h, 'rot': car.rot_y})

    print("Đang xây dựng: Mỗi ô trống 1 tòa nhà (Check lấn đường)...")
    for i, (tile_x, tile_z) in enumerate(open_space_tiles):
        if not building_models: break
        
        bldg = SceneObject(random.choice(building_models), f"Bldg_{i}", 2, "Building")
        bldg.pos_x = tile_x + random.uniform(-0.1, 0.1) 
        bldg.pos_z = tile_z + random.uniform(-0.1, 0.1)
        bldg.rot_y = random.choice([0.0, 90.0, 180.0, -90.0])
        
        target_scale = random.uniform(1.8, 2.3)
        while target_scale > 0.5:
            if not check_building_road_collision(bldg.pos_x, bldg.pos_z, target_scale, road_tiles, TILE_SIZE):
                break 
            target_scale -= 0.15 
            
        bldg.scale = target_scale
        bldg.texture_id = building_tex # TRẢ LẠI TEXTURE CHUẨN CỦA ÔNG
        scene_objects.append(bldg)

    if len(car_models) > 0:
        ego_x = (0 - offset_x) * TILE_SIZE + 0.25 * TILE_SIZE
        ego_z = (5 - offset_z) * TILE_SIZE
        
        ego_model = car_models[0] # Lấy mẫu xe đầu tiên làm xe Dashcam
        ego_car = SceneObject(ego_model, "Ego_Dashcam_Car", 1, "Car")
        
        ego_base_scale, _, _ = get_car_info(getattr(ego_model, 'filename', ''))
        ego_car.scale = ego_base_scale
        
        ego_car.pos_x, ego_car.pos_z, ego_car.rot_y = ego_x, ego_z, 180.0
        ego_car.texture_id = car_tex
        scene_objects.append(ego_car)

    # Cameras (Điều chỉnh khoảng cách xa hơn để nhìn bản đồ khổng lồ)
    cameras = [Trackball(distance=10.0), Trackball(distance=40.0), Trackball(distance=25.0)]
    cameras[0].elevation, cameras[0].azimuth = 10.0, 0.0 
    cameras[1].elevation, cameras[1].azimuth = 80.0, 0.0 
    cameras[2].elevation, cameras[2].azimuth = 30.0, -45.0
    
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
            
            closest_dist, selected_idx = float('inf'), -1
            for i, obj in enumerate(scene_objects):
                t = np.dot(np.array([obj.pos_x, obj.pos_y, obj.pos_z]) - ray_orig, ray_dir)
                if t > 0: 
                    dist = np.linalg.norm(np.array([obj.pos_x, obj.pos_y, obj.pos_z]) - (ray_orig + t * ray_dir))
                    if dist < 1.0 * obj.scale and t < closest_dist:
                        closest_dist, selected_idx = t, i
            if selected_idx != -1: gui.selected_scene_obj_idx = selected_idx
    glfw.set_mouse_button_callback(window, on_mouse_click)

    def on_mouse_move(win, xpos, ypos):
        if not hasattr(on_mouse_move, "old_pos"): on_mouse_move.old_pos = (xpos, ypos)
        dx, dy = xpos - on_mouse_move.old_pos[0], ypos - on_mouse_move.old_pos[1]
        on_mouse_move.old_pos = (xpos, ypos)
        if imgui.get_io().want_capture_mouse: return
        
        is_shift = glfw.get_key(win, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT):
            if is_shift and len(scene_objects) > 0:
                scene_objects[gui.selected_scene_obj_idx].rot_y += dx * 0.2
                scene_objects[gui.selected_scene_obj_idx].rot_x += dy * 0.2
            else:
                cameras[gui.selected_cam_idx].azimuth -= dx * 0.2
                cameras[gui.selected_cam_idx].elevation = max(-89.0, min(89.0, cameras[gui.selected_cam_idx].elevation - dy * 0.2))
        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_RIGHT):
            if is_shift and len(scene_objects) > 0:
                cam_az = np.radians(cameras[gui.selected_cam_idx].azimuth)
                scene_objects[gui.selected_scene_obj_idx].pos_x += dx * 0.015 * np.cos(cam_az)
                scene_objects[gui.selected_scene_obj_idx].pos_z += dx * 0.015 * np.sin(cam_az)
                scene_objects[gui.selected_scene_obj_idx].pos_y -= dy * 0.015
            else:
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

    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()

        if getattr(gui, 'texture_changed', False) and len(scene_objects) > 0:
            target_obj = scene_objects[getattr(gui, 'target_tex_obj_idx', 0)]
            if target_obj.texture_id > 0: GL.glDeleteTextures(1, [target_obj.texture_id])
            target_obj.texture_id = load_texture(target_obj.texture_filepath)
            gui.texture_changed = False

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

        if gui.add_shape_requested:
            obj_path = os.path.join("assets", "models", gui.obj_filepath) if not gui.obj_filepath.startswith("assets") else gui.obj_filepath
            new_shape = ObjModel(filepath=obj_path)
            obj = SceneObject(new_shape, f"#{len(scene_objects)+1} {gui.class_name}", gui.class_id, gui.class_name)
            
            if hasattr(new_shape, 'texture_file') and new_shape.texture_file:
                obj.texture_filepath = new_shape.texture_file
                obj.texture_id = load_texture(new_shape.texture_file)
                obj.render_mode = 4 
            elif hasattr(new_shape, 'diffuse_color') and new_shape.diffuse_color:
                obj.flat_color = new_shape.diffuse_color
                obj.render_mode = 3 
            
            obj.pos_x, obj.pos_y, obj.pos_z = gui.spawn_pos
            scene_objects.append(obj)
            gui.selected_scene_obj_idx = len(scene_objects) - 1
            gui.spawn_pos[0] += 2.5 
            gui.add_shape_requested = False
            
        if gui.clear_scene_requested:
            scene_objects = []
            gui.clear_scene_requested = False
            gui.selected_scene_obj_idx = 0

        gui.render(cameras, scene_objects)
        
        fb_width, fb_height = glfw.get_framebuffer_size(window)
        win_size = glfw.get_window_size(window)
        current_cam = cameras[gui.selected_cam_idx]
        view_matrix, projection_matrix = current_cam.view_matrix(), current_cam.projection_matrix(win_size)
        
        GL.glViewport(0, 0, fb_width, fb_height)
        
        bg = gui.bg_color if gui.view_mode == 0 else [0.0, 0.0, 0.0]
        GL.glClearColor(*bg, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        
        my_shader.use()
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(my_shader.render_idx, "view"), 1, GL.GL_TRUE, view_matrix)
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(my_shader.render_idx, "projection"), 1, GL.GL_TRUE, projection_matrix)
        GL.glUniform1i(GL.glGetUniformLocation(my_shader.render_idx, "is_depth_map"), 1 if gui.view_mode == 1 else 0)
        GL.glUniform3f(GL.glGetUniformLocation(my_shader.render_idx, "bg_color"), *bg)
        
        lights_active = gui.lights if gui.view_mode == 0 else [False, False, False]
        GL.glUniform1i(GL.glGetUniformLocation(my_shader.render_idx, "light1_on"), lights_active[0])
        GL.glUniform1i(GL.glGetUniformLocation(my_shader.render_idx, "light2_on"), lights_active[1])
        GL.glUniform1i(GL.glGetUniformLocation(my_shader.render_idx, "light3_on"), False)
        
        GL.glUniform3f(GL.glGetUniformLocation(my_shader.render_idx, "viewPos"), *np.linalg.inv(view_matrix)[:3, 3])
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE if getattr(gui, 'is_wireframe', False) else GL.GL_FILL)

        for obj in scene_objects:
            m_trans = translate(obj.pos_x, obj.pos_y, obj.pos_z)
            m_scale = scale(obj.scale, obj.scale, obj.scale)
            model_matrix = np.matmul(m_trans, np.matmul(np.matmul(rotate_y(obj.rot_y), rotate_x(obj.rot_x)), m_scale))
            
            GL.glUniformMatrix4fv(GL.glGetUniformLocation(my_shader.render_idx, "model"), 1, GL.GL_TRUE, model_matrix)
            
            if gui.view_mode == 2:
                GL.glUniform1i(GL.glGetUniformLocation(my_shader.render_idx, "render_mode"), 0)
                GL.glUniform3f(GL.glGetUniformLocation(my_shader.render_idx, "flat_color"), *obj.mask_color)
            else: 
                GL.glUniform1i(GL.glGetUniformLocation(my_shader.render_idx, "render_mode"), 4 if obj.texture_id > 0 else 3)
                if obj.texture_id > 0 and gui.view_mode == 0:
                    GL.glActiveTexture(GL.GL_TEXTURE0)
                    GL.glBindTexture(GL.GL_TEXTURE_2D, obj.texture_id)
                    GL.glUniform1i(GL.glGetUniformLocation(my_shader.render_idx, "tex_diffuse"), 0)
            
            obj.shape.draw()
            
        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

    impl.shutdown()
    glfw.terminate()

if __name__ == "__main__":
    main()