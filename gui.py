import imgui

class AppGUI:
    def __init__(self):
        self.obj_filepath = "police.obj"
        self.class_name = "Vehicles"
        self.class_id = 1
        
        self.spawn_pos = [0.0, 0.0, 0.0]
        self.add_shape_requested = False 
        self.clear_scene_requested = False
        
        self.delete_obj_requested = False
        self.duplicate_obj_requested = False
        self.selected_scene_obj_idx = 0 
        
        
        self.is_wireframe = False
        
        self.view_mode = 0 

        self.lights = [True, True, False] 
        self.bg_color = [0.5, 0.7, 0.9] 
        self.selected_cam_idx = 0

    def render(self, cameras, scene_objects):
        imgui.new_frame()
        
        imgui.set_next_window_size(550, 860, imgui.FIRST_USE_EVER)
        imgui.set_next_window_position(20, 20, imgui.FIRST_USE_EVER)
        
        imgui.begin("CALAR SYNTHETIC DATASET BUILDER", flags=imgui.WINDOW_ALWAYS_VERTICAL_SCROLLBAR)
        
        if imgui.collapsing_header("1. ASSET LOADER & LABELS", flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
            imgui.text_colored("3D Model (.obj / .ply):", 1.0, 0.8, 0.2)
            _, self.obj_filepath = imgui.input_text("File Path", self.obj_filepath, 256)
            
            imgui.spacing()
            imgui.text_colored("Ground Truth Labels (COCO/YOLO Format):", 0.4, 1.0, 0.4)
            _, self.class_name = imgui.input_text("Class Name", self.class_name, 64)
            _, self.class_id = imgui.input_int("Class ID", self.class_id)
            if self.class_id < 0: self.class_id = 0
            
            imgui.spacing()
            _, self.spawn_pos = imgui.input_float3("Spawn Pos", *self.spawn_pos)
            self.spawn_pos = list(self.spawn_pos)
            
            imgui.spacing()
            if imgui.button("Add to Scene", width=200): self.add_shape_requested = True
            imgui.same_line()
            if imgui.button("Clear Scene", width=-1): self.clear_scene_requested = True
        
        if imgui.collapsing_header("2. SCENE MANAGEMENT", flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
            
            _, self.is_wireframe = imgui.checkbox("Wireframe Mode (Debug Mesh)", self.is_wireframe)
            imgui.spacing()
            
            if len(scene_objects) == 0:
                imgui.text_colored("Scene is empty. Add a model first.", 0.5, 0.5, 0.5)
            else:
                obj_names = [f"[{obj.class_id}] {obj.name}" for obj in scene_objects]
                self.selected_scene_obj_idx = max(0, min(self.selected_scene_obj_idx, len(scene_objects) - 1))
                _, self.selected_scene_obj_idx = imgui.combo("Select Object", self.selected_scene_obj_idx, obj_names)
                
                active_obj = scene_objects[self.selected_scene_obj_idx]
                
                imgui.spacing()
                if imgui.button("Delete Object", width=120): self.delete_obj_requested = True
                imgui.same_line()
                if imgui.button("Duplicate Object", width=120): self.duplicate_obj_requested = True
                
                imgui.spacing()
                imgui.separator()
                imgui.spacing()
                
                imgui.text_colored("Transformations:", 1.0, 0.8, 0.0)
                _, active_obj.scale = imgui.slider_float("Scale", active_obj.scale, 0.1, 10.0)
                _, active_obj.rot_x = imgui.slider_float("Rot X", active_obj.rot_x, -180.0, 180.0)
                _, active_obj.rot_y = imgui.slider_float("Rot Y", active_obj.rot_y, -180.0, 180.0)
                _, active_obj.pos_x = imgui.slider_float("Pos X", active_obj.pos_x, -50.0, 50.0)
                _, active_obj.pos_y = imgui.slider_float("Pos Y", active_obj.pos_y, -10.0, 50.0)
                _, active_obj.pos_z = imgui.slider_float("Pos Z", active_obj.pos_z, -50.0, 50.0)
                
        
        imgui.end()

        io = imgui.get_io()
        imgui.set_next_window_size(350, 200, imgui.FIRST_USE_EVER)
        imgui.set_next_window_position(io.display_size.x - 370, 20, imgui.FIRST_USE_EVER)
        imgui.begin("DATASET PIPELINE", flags=imgui.WINDOW_ALWAYS_VERTICAL_SCROLLBAR)
        
        imgui.text_colored("Preview Render Mode:", 0.4, 0.8, 1.0)
        if imgui.radio_button("RGB Image (Realistic)", self.view_mode == 0): self.view_mode = 0
        if imgui.radio_button("Depth Map (Z-Buffer)", self.view_mode == 1): self.view_mode = 1
        if imgui.radio_button("Segmentation Mask (Class ID)", self.view_mode == 2): self.view_mode = 2
        
        imgui.spacing(); imgui.separator(); imgui.spacing()
        
        if imgui.button("GENERATE GROUND TRUTH", width=-1, height=50):
            self.generate_requested = True 
            
        imgui.end()