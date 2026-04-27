class Scene:
    """3D scene functions"""
    def __init__(self):
        self.models = []
        self.lights = []
        self.camera = None
        self.background_color = (0, 0, 0) # 預設黑底

    def add_model(self, model):
        self.models.append(model)
        print(f"3D Model added. Total models: {len(self.models)}")

    def remove_model(self, model):
        if model in self.models:
            self.models.remove(model)
            
    def set_camera(self, camera):
    
        self.camera = camera
        
    def add_light(self, light):
        
        self.lights.append(light)

    def clear(self):
        
        self.models.clear()
        self.lights.clear()

