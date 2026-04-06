import imgui

class AppGUI:
    def __init__(self):
        self.obj_filepath = "car.obj"
        self.class_name = "Car"
        self.class_id = 1
        
        self.spawn_pos = [0.0, 0.0, 0.0]
        self.add_shape_requested = False 
        self.clear_scene_requested = False
        
        self.delete_obj_requested = False
        self.duplicate_obj_requested = False
        self.selected_scene_obj_idx = 0 
        
        self.texture_changed = False
        self.target_tex_obj_idx = 0
        
        self.is_wireframe = False
        
        self.view_mode = 0 

        self.lights = [True, True, False] 
        self.bg_color = [0.5, 0.7, 0.9] 
        self.selected_cam_idx = 0

    def render(self, cameras, scene_objects):
        imgui.new_frame()
        imgui.set_next_window_size(400, 800, imgui.FIRST_USE_EVER)
        imgui.set_next_window_position(20, 20, imgui.FIRST_USE_EVER)
        
        imgui.begin("CALAR SYNTHETIC DATASET BUILDER", flags=imgui.WINDOW_ALWAYS_VERTICAL_SCROLLBAR)
        
        # --- PHẦN 1: NẠP MODEL VÀ GÁN NHÃN ---
        if imgui.collapsing_header("1. ASSET LOADER & LABELS", flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
            imgui.text_colored("3D Model (.obj / .ply):", 1.0, 0.8, 0.2)
            _, self.obj_filepath = imgui.input_text("File Path", self.obj_filepath, 256)
            
            imgui.spacing()
            imgui.text_colored("Ground Truth Labels (COCO/YOLO):", 0.4, 1.0, 0.4)
            _, self.class_name = imgui.input_text("Class Name", self.class_name, 64)
            _, self.class_id = imgui.input_int("Class ID", self.class_id)
            if self.class_id < 0: self.class_id = 0
            
            imgui.spacing()
            _, self.spawn_pos = imgui.input_float3("Spawn Pos", *self.spawn_pos)
            self.spawn_pos = list(self.spawn_pos)
            
            if imgui.button("Add to Scene", width=180): self.add_shape_requested = True
            imgui.same_line()
            if imgui.button("Clear Scene", width=-1): self.clear_scene_requested = True
        
        # --- PHẦN 2: QUẢN LÝ VÀ CHỈNH SỬA CẢNH ---
        if imgui.collapsing_header("2. SCENE MANAGEMENT", flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
            
            _, self.is_wireframe = imgui.checkbox("Wireframe Mode (Debug Mesh)", self.is_wireframe) # <--- THÊM NÚT NÀY VÀO
            imgui.spacing()
            
            if len(scene_objects) == 0:
                imgui.text_colored("Scene is empty.", 0.5, 0.5, 0.5)
            else:
                obj_names = [f"[{obj.class_id}] {obj.name}" for obj in scene_objects]
                self.selected_scene_obj_idx = max(0, min(self.selected_scene_obj_idx, len(scene_objects) - 1))
                _, self.selected_scene_obj_idx = imgui.combo("Select Object", self.selected_scene_obj_idx, obj_names)
                
                active_obj = scene_objects[self.selected_scene_obj_idx]
                
                imgui.spacing()
                if imgui.button("Delete", width=80): self.delete_obj_requested = True
                imgui.same_line()
                if imgui.button("Duplicate", width=90): self.duplicate_obj_requested = True
                
                imgui.spacing()
                imgui.text_colored("Transformations:", 1.0, 0.8, 0.0)
                _, active_obj.scale = imgui.slider_float("Scale", active_obj.scale, 0.1, 10.0)
                _, active_obj.rot_x = imgui.slider_float("Rot X", active_obj.rot_x, -180.0, 180.0)
                _, active_obj.rot_y = imgui.slider_float("Rot Y", active_obj.rot_y, -180.0, 180.0)
                _, active_obj.pos_x = imgui.slider_float("Pos X", active_obj.pos_x, -50.0, 50.0)
                _, active_obj.pos_y = imgui.slider_float("Pos Y", active_obj.pos_y, -10.0, 50.0)
                _, active_obj.pos_z = imgui.slider_float("Pos Z", active_obj.pos_z, -50.0, 50.0)
                
                imgui.spacing()
                imgui.text_colored("Appearance:", 1.0, 0.8, 0.0)
                _, active_obj.texture_filepath = imgui.input_text("Texture File", active_obj.texture_filepath, 256)
                if imgui.button("Apply Texture", width=-1):
                    self.texture_changed = True
                    self.target_tex_obj_idx = self.selected_scene_obj_idx
        
        # --- PHẦN 3: CAMERA & MÔI TRƯỜNG ---
        if imgui.collapsing_header("3. ENVIRONMENT & DASHCAM", flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
            imgui.text("Lighting:")
            _, self.lights[0] = imgui.checkbox("Sun Light", self.lights[0])
            _, self.lights[1] = imgui.checkbox("Street Light", self.lights[1])
            
            imgui.spacing()
            _, self.selected_cam_idx = imgui.combo("Camera", self.selected_cam_idx, ["1. Dashcam (Ego Vehicle)", "2. Top-Down View", "3. Free Cam"])
            cam = cameras[self.selected_cam_idx]
            _, cam.distance = imgui.slider_float("Zoom", cam.distance, 1.0, 100.0)
            _, cam.azimuth = imgui.slider_float("Azimuth", cam.azimuth, -180.0, 180.0)
            _, cam.elevation = imgui.slider_float("Elevation", cam.elevation, -89.0, 89.0)

        imgui.end()

        # --- BẢNG XUẤT DATASET ---
        io = imgui.get_io()
        imgui.set_next_window_size(350, 200, imgui.FIRST_USE_EVER)
        imgui.set_next_window_position(io.display_size.x - 370, 20, imgui.FIRST_USE_EVER)
        imgui.begin("DATASET PIPELINE", flags=imgui.WINDOW_ALWAYS_VERTICAL_SCROLLBAR)
        
        imgui.text_colored("Preview Render Mode:", 0.4, 0.8, 1.0)
        if imgui.radio_button("RGB Image", self.view_mode == 0): self.view_mode = 0
        if imgui.radio_button("Depth Map", self.view_mode == 1): self.view_mode = 1
        if imgui.radio_button("Segmentation Mask", self.view_mode == 2): self.view_mode = 2
        
        imgui.spacing(); imgui.separator(); imgui.spacing()
        
        if imgui.button("GENERATE GROUND TRUTH", width=-1, height=50):
            pass 
            
        imgui.end() 