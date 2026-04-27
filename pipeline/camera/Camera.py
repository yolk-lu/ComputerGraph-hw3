import numpy as np

class Camera:
    """相機視角(在scene移動)"""
    def __init__(self):
        self.position = [0.0, 0.0, 0.0]
        self.rotation = [0.0, 0.0, 0.0]
        self.target = [0.0, 0.0, 0.0]
        self.fov = 75.0
        self.near = 0.1
        self.far = 1000.0
        self.aspect = 1.0
        self.mode = "third_person"

    @staticmethod
    def from_threejs(camera_info: dict):
        """從 Three.js 傳回的 camera dict 建立 Camera 物件"""
        cam = Camera()
        
        pos = camera_info.get('position', {})
        cam.position = [pos.get('x', 0), pos.get('y', 0), pos.get('z', 0)]
        
        rot = camera_info.get('rotation', {})
        cam.rotation = [rot.get('x', 0), rot.get('y', 0), rot.get('z', 0)]
        
        target = camera_info.get('target', {})
        cam.target = [target.get('x', 0), target.get('y', 0), target.get('z', 0)]
        
        cam.fov = camera_info.get('fov', 75.0)
        cam.near = camera_info.get('near', 0.1)
        cam.far = camera_info.get('far', 1000.0)
        cam.aspect = camera_info.get('aspect', 1.0)
        cam.mode = camera_info.get('mode', 'third_person')
        
        return cam

    def get_view_matrix(self) -> np.ndarray:
        """
        Calculates the 4x4 View Matrix representing the camera's viewpoint.
        
        If the camera mode is 'third_person', it creates a look-at matrix 
        pointing toward the target (orbit control center).
        If 'first_person', it approximates the forward vector using Euler angles.
        
        Returns:
            np.ndarray: A 4x4 float32 view matrix.
        """
        eye = np.array(self.position, dtype=np.float32)
        
        if self.mode == "third_person" and hasattr(self, 'target') and self.target:
            center = np.array(self.target, dtype=np.float32)
        else:
            # First-person orientation calculation using pitch and yaw
            yaw, pitch = self.rotation[1], self.rotation[0]
            forward = np.array([
                -np.sin(yaw) * np.cos(pitch),
                 np.sin(pitch),
                -np.cos(yaw) * np.cos(pitch)
            ], dtype=np.float32)
            center = eye + forward

        up = np.array([0, 1, 0], dtype=np.float32)

        f = center - eye
        norm_f = np.linalg.norm(f)
        if norm_f < 1e-6:
            f = np.array([0, 0, -1], dtype=np.float32)
        else:
            f = f / norm_f

        s = np.cross(f, up)
        norm_s = np.linalg.norm(s)
        if norm_s < 1e-6:
            s = np.array([1, 0, 0], dtype=np.float32)
        else:
            s = s / norm_s

        u = np.cross(s, f)

        view_matrix = np.eye(4, dtype=np.float32)
        view_matrix[0, :3] = s
        view_matrix[1, :3] = u
        view_matrix[2, :3] = -f
        
        trans_matrix = np.eye(4, dtype=np.float32)
        trans_matrix[:3, 3] = -eye
        
        return view_matrix @ trans_matrix

    def get_projection_matrix(self) -> np.ndarray:
        """
        Calculates the 4x4 Perspective Projection Matrix using FOV, aspect, near, and far.

        Returns:
            np.ndarray: A 4x4 float32 perspective projection matrix.
        """
        f = 1.0 / np.tan(np.radians(self.fov) / 2.0)
        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = f / self.aspect
        proj[1, 1] = f
        proj[2, 2] = (self.far + self.near) / (self.near - self.far)
        proj[2, 3] = (2.0 * self.far * self.near) / (self.near - self.far)
        proj[3, 2] = -1.0
        return proj

    def __repr__(self):
        return (f"Camera(pos={self.position}, rot={self.rotation}, "
                f"target={self.target}, fov={self.fov}, mode={self.mode})")
