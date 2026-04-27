

import os 
import shutil

class utils:
    @staticmethod
    def load_obj(file_path):
        """載入obj檔案"""
        vertices = []
        faces = []
        if not os.path.exists(file_path):
            return vertices, faces

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.split()
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith('f '):
                    parts = line.split()
                    face = []
                    for p in parts[1:]:
                        # obj format face indices start from 1
                        v_idx = int(p.split('/')[0]) - 1
                        face.append(v_idx)
                    faces.append(face)
        return vertices, faces

    
    @staticmethod
    def save_uploaded_model(real_path):
        """上傳的模型複製到專案內的 input_model 資料夾"""
        
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        input_dir = os.path.join(project_root, "input_model")
        os.makedirs(input_dir, exist_ok=True)
        
        # Always use a safe filename to prevent Gradio 404 URL encoding issues (e.g. spaces, Chinese characters)
        original_filename = "current_model.glb"
            
        target_path = os.path.join(input_dir, original_filename)
        shutil.copy2(real_path, target_path)
        return target_path


    @staticmethod
    def load_glb(file_path): 
        import trimesh
        try:
            # force='mesh' concatenates all meshes in the scene into a single Trimesh object
            mesh = trimesh.load(file_path, force='mesh')
            if isinstance(mesh, trimesh.Trimesh):
                vertices = mesh.vertices.tolist()
                faces = mesh.faces.tolist()
                return vertices, faces
            elif isinstance(mesh, trimesh.Scene):
                # Fallback if force='mesh' didn't perfectly concatenate
                # This usually happens if there are different materials, etc.
                if not mesh.is_empty:
                    concat_mesh = mesh.dump(concatenate=True)
                    return concat_mesh.vertices.tolist(), concat_mesh.faces.tolist()
            return [], []
        except Exception as e:
            print(f"[Utils] Error loading GLB: {e}")
            return [], []