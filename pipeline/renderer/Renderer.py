import base64
from io import BytesIO
import PIL.Image
import PIL.ImageOps
import numpy as np
import os
# PyOpenGL/ModernGL
import moderngl
import trimesh

from pipeline.scene.Scene import Scene
from pipeline.model.diffusion_model import Diffusion_Model
from pipeline.camera.Camera import Camera
from pipeline.model.dlss_model import AIUpscaler

from pipeline.shader.shader import SCENE_VERTEX_SHADER, SCENE_FRAGMENT_SHADER

import threading
import queue

class Renderer:
    """render function and flow control"""
    def __init__(self):
        self.scene = Scene()
        self.diffusion_model = None
        self.dlss_upscaler = None
        
        self.ctx = None
        self.prog = None
        self.fbo = None
        self.vbo = None
        self.vao = None
        self.color_tex = None
        self.depth_tex = None

        self.display_res = 1024
        self.scale_factor = 2
        self.render_res = self.display_res // self.scale_factor
        self.obj_num_verts = 0
        
        # Dedicated Threading queue for OpenGL
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def _worker_loop(self):
        # ModernGL context must be created on the thread that uses it
        self.init_gl()
        
        while True:
            task = self.task_queue.get()
            if task is None:
                print("get end signal，close Worker Thread。")
                break
                
            task_type, args, kwargs = task
            print(f"Worker thread start processing: {task_type}")
            try:
                if task_type == 'prepare_scene':
                    res = self._prepare_scene(*args, **kwargs)
                elif task_type == 'render_dlss_comparison':
                    res = self._render_dlss_comparison(*args, **kwargs)
                elif task_type == 'train_dlss':
                    res = self._train_dlss(*args, **kwargs)
                elif task_type == 'render_benchmark':
                    res = self._render_benchmark(*args, **kwargs)
                else:
                    raise ValueError(f"Unknown task: {task_type}")
                print(f"Rendering {task_type} 成功，回傳結果。")
                self.result_queue.put((True, res))
            except Exception as e:
                import traceback
                print(f"Rendering {task_type} 發生例外錯誤！")
                traceback.print_exc()
                self.result_queue.put((False, e))
            
            self.task_queue.task_done()

    def prepare_scene(self, model_file_path: str):
        self.task_queue.put(('prepare_scene', (model_file_path,), {}))
        success, result = self.result_queue.get()
        if not success:
            raise result
        return result

    def render_dlss_comparison(self, camera_info: dict):
        print("排 render_dlss_comparison 到 task_queue...")
        self.task_queue.put(('render_dlss_comparison', (camera_info,), {}))
        success, result = self.result_queue.get()
        if not success:
            print("render_dlss_comparison 失敗，exception。")
            raise result
        print("render_dlss_comparison completed")
        return result

    def render_benchmark(self, camera_info: dict):
        print("排 render_benchmark 到 task_queue...")
        self.task_queue.put(('render_benchmark', (camera_info,), {}))
        success, result = self.result_queue.get()
        if not success:
            raise result
        return result

    def train_dlss(self, camera_info: dict, epochs: int, lr: float, scale: int):
        print(f"發送 train_dlss 到 task_queue... Epochs: {epochs}")
        self.task_queue.put(('train_dlss', (camera_info, epochs, lr, scale), {}))
        success, result = self.result_queue.get()
        if not success:
            raise result
        return result

    def init_gl(self):
        """Initializes the Headless ModernGL Context and Shaders."""
        if self.ctx is None:
            self.ctx = moderngl.create_standalone_context()
            self.prog = self.ctx.program(
                vertex_shader=SCENE_VERTEX_SHADER,
                fragment_shader=SCENE_FRAGMENT_SHADER,
            )
        
        # Recreate FBO if missing or destroyed (e.g. during resolution scaling for DLSS)
        if self.fbo is None:
            self.color_tex = self.ctx.texture((self.render_res, self.render_res), 4, dtype='f1')
            self.depth_tex = self.ctx.depth_texture((self.render_res, self.render_res))
            self.fbo = self.ctx.framebuffer(
                color_attachments=[self.color_tex],
                depth_attachment=self.depth_tex,
            )

    def _prepare_scene(self, model_file_path: str):
        """
        Loads the GLB file and generates ModernGL VBO/VAOs with Textures.
        """
        if not model_file_path:
            return None
            
        self.scene.clear()
        
        import trimesh
        # Use force='scene' to preserve multiple materials and UVs
        t_scene = trimesh.load(model_file_path, force='scene')
        
        if hasattr(self, 'render_batches'):
            for batch in self.render_batches:
                if batch['vao']: batch['vao'].release()
                if batch['vbo']: batch['vbo'].release()
                if batch['texture']: batch['texture'].release()
        self.render_batches = []
        
        # If it loaded as a simple Trimesh object instead of a Scene
        geometries = t_scene.dump() if isinstance(t_scene, trimesh.Scene) else [t_scene]
        
        for geom in geometries:
            verts = np.array(geom.vertices, dtype=np.float32)
            faces = geom.faces
            
            # Check for UVs
            has_uv = hasattr(geom.visual, 'uv') and geom.visual.uv is not None
            uvs = np.array(geom.visual.uv, dtype=np.float32) if has_uv else None
            
            if len(verts) == 0:
                continue
                
            flat_data = []
            for face in faces:
                if len(face) >= 3:
                    for i in range(1, len(face) - 1):
                        v0 = verts[face[0]]
                        v1 = verts[face[i]]
                        v2 = verts[face[i+1]]
                        
                        edge1 = v1 - v0
                        edge2 = v2 - v0
                        normal = np.cross(edge1, edge2)
                        norm_mag = np.linalg.norm(normal)
                        if norm_mag > 0:
                            normal = normal / norm_mag
                        else:
                            normal = np.array([0, 1, 0], dtype=np.float32)
                            
                        uv0 = uvs[face[0]] if has_uv else np.array([0, 0], dtype=np.float32)
                        uv1 = uvs[face[i]] if has_uv else np.array([0, 0], dtype=np.float32)
                        uv2 = uvs[face[i+1]] if has_uv else np.array([0, 0], dtype=np.float32)
                        
                        flat_data.extend([*v0, *normal, *uv0])
                        flat_data.extend([*v1, *normal, *uv1])
                        flat_data.extend([*v2, *normal, *uv2])
                        
            vertex_data = np.array(flat_data, dtype=np.float32)
            obj_num_verts = len(vertex_data) // 8
            
            if obj_num_verts == 0:
                continue
                
            vbo = self.ctx.buffer(vertex_data.tobytes())
            vao = self.ctx.vertex_array(
                self.prog, [(vbo, '3f 3f 2f', 'in_position', 'in_normal', 'in_uv')]
            )
            
            # Texture extraction
            gl_texture = None
            if hasattr(geom.visual, 'material'):
                mat = geom.visual.material
                img = None
                if hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
                    img = mat.baseColorTexture
                elif hasattr(mat, 'image') and mat.image is not None:
                    img = mat.image
                    
                if img is not None:
                    # Convert to RGB, flip Y for OpenGL
                    img = img.convert('RGB').transpose(PIL.Image.FLIP_TOP_BOTTOM)
                    gl_texture = self.ctx.texture(img.size, 3, img.tobytes())
                    gl_texture.build_mipmaps()
                    # Ensure repeating and linear filtering for GLB compatibility
                    gl_texture.repeat_x = True
                    gl_texture.repeat_y = True
                    # Set filter correctly for mipmaps
                    gl_texture.filter = (self.ctx.LINEAR_MIPMAP_LINEAR, self.ctx.LINEAR)
                    
            self.render_batches.append({
                'vao': vao,
                'vbo': vbo,
                'texture': gl_texture,
                'num_verts': obj_num_verts
            })
            
        self.vao = True # Dummy flag so we don't think it's empty
        return len(geometries)

    def _linearize_depth(self, depth_raw: np.ndarray, near: float, far: float) -> np.ndarray:
        """
        Convert non-linear OpenGL depth to linear [0, 1].

        Args:
            depth_raw (np.ndarray): OpenGL perspective depth array [0,1].
            near (float): Camera near clipping plane.
            far (float): Camera far clipping plane.

        Returns:
            np.ndarray: Linearized depth array.
        """
        z_ndc = depth_raw * 2.0 - 1.0
        z_linear = (2.0 * near * far) / (far + near - z_ndc * (far - near))
        return z_linear / far

    def _render_scene_to_fbo(self, camera: Camera):
        """
        Renders the VBO to the low-res FBO and reads back the numpy arrays.

        Args:
            camera (Camera): The python Camera object with projection logic.

        Returns:
            tuple[np.ndarray, np.ndarray]: (color float32 image [H,W,3], linear depth [H,W]).
        """
        self.init_gl()
        if not self.vao:
             # Nothing loaded
             return np.zeros((self.render_res, self.render_res, 3), dtype=np.float32), \
                    np.zeros((self.render_res, self.render_res), dtype=np.float32)

        self.fbo.use()
        self.ctx.viewport = (0, 0, self.render_res, self.render_res)
        self.ctx.clear(0.1, 0.1, 0.15, 1.0) # Dark gray background
        self.ctx.enable(moderngl.DEPTH_TEST)

        # Set uniforms
        view = camera.get_view_matrix()
        proj = camera.get_projection_matrix()
        model_mat = np.eye(4, dtype=np.float32)

        self.prog['u_view'].write(view.T.astype('f4').tobytes())
        self.prog['u_proj'].write(proj.T.astype('f4').tobytes())
        self.prog['u_model'].write(model_mat.T.astype('f4').tobytes())
        self.prog['u_cam_pos'].value = tuple(camera.position)
        self.prog['u_color'].value = (0.7, 0.7, 0.8) # Base material color

        # Draw
        if hasattr(self, 'render_batches'):
            for batch in self.render_batches:
                if batch['texture']:
                    batch['texture'].use(location=0)
                    if 'u_texture' in self.prog:
                        self.prog['u_texture'].value = 0
                    if 'u_use_texture' in self.prog:
                        self.prog['u_use_texture'].value = True
                else:
                    if 'u_use_texture' in self.prog:
                        self.prog['u_use_texture'].value = False
                batch['vao'].render(moderngl.TRIANGLES, vertices=batch['num_verts'])

        # Read back Color
        raw_color = self.color_tex.read()
        color = np.frombuffer(raw_color, dtype=np.uint8).reshape(self.render_res, self.render_res, 4)
        color = color[::-1, :, :3].copy().astype(np.float32) / 255.0  # Strip Alpha and flip Y

        # Read back Depth
        raw_depth = self.depth_tex.read()
        depth = np.frombuffer(raw_depth, dtype=np.float32).reshape(self.render_res, self.render_res)
        depth = depth[::-1].copy()  # flip Y

        # Linearize Depth
        linear_depth = self._linearize_depth(depth, camera.near, camera.far)

        return color, linear_depth


    def render_with_controlnet(self, b64_depth: str, prompt: str, camera_info: dict):
        """
        Executes BOTH the ControlNet pipeline (original Base64 workflow)
        and the DLSS ModernGL rendering flow, returning output for both.

        Args:
            b64_depth (str): Base64 8-bit depth string returned from Three.js MeshDepthMaterial.
            prompt (str): Text prompt for SD ControlNet generation.
            camera_info (dict): Camera JSON packet mapping to properties like position and rotation.

        Returns:
            tuple[PIL.Image.Image, PIL.Image.Image]: (DLSS Resized Output, Diffusion Image)
        """
        # ==========================================
        # 1. Parse Camera
        # ==========================================
        camera = None
        if camera_info:
            camera = Camera.from_threejs(camera_info)
            self.scene.set_camera(camera)
            print(f"Camera loaded: {camera}")
            
        if not camera:
             raise ValueError("Camera data is missing.")

        # ==========================================
        # 2. DLSS Upscale Pipeline (Software 3D->2D)
        # ==========================================
        if self.dlss_upscaler is None:
             self.dlss_upscaler = AIUpscaler(scale_factor=self.scale_factor)
        
        # Render internal low-res OpenGL representation
        low_res_color, low_res_depth = self._render_scene_to_fbo(camera)
        
        # Upscale
        high_res_color = self.dlss_upscaler.upscale(low_res_color, low_res_depth)
        
        # Convert DLSS output to PIL array
        dlss_image_uint8 = (np.clip(high_res_color, 0.0, 1.0) * 255).astype(np.uint8)
        dlss_pil = PIL.Image.fromarray(dlss_image_uint8)

        # ==========================================
        # 3. ControlNet Diffusion Pipeline
        # ==========================================
        if not b64_depth or not b64_depth.startswith('data:image'):
            raise ValueError("Invalid Base64 depth map from Frontend. Did object render completely?")
        
        if not prompt:
            raise ValueError("Please provide a prompt for ControlNet!")
            
        header, encoded = b64_depth.split(",", 1)
        data = base64.b64decode(encoded)
        depth_pil_raw = PIL.Image.open(BytesIO(data)).convert("RGB")
        
        # Invert to map Three.js Depth space to ControlNet expected space
        depth_pil = PIL.ImageOps.invert(depth_pil_raw)
        
        # ControlNet SD1.5 expects 512x512
        depth_pil = depth_pil.resize((512, 512), PIL.Image.LANCZOS)
        
        if self.diffusion_model is None:
            # Lazy load ControlNet (Heavy VRAM operation)
            from pipeline.model.model import Model
            self.diffusion_model = Model()
            
        controlnet_pil = self.diffusion_model.generate(depth_pil, prompt)

        return dlss_pil, controlnet_pil
    def _render_dlss_comparison(self, camera_info: dict):
        """
        Produce a comparison between DLSS OFF (Bilinear) and DLSS ON (AI Upscale).
        Does NOT rely on Base64 depth from Three.js or run ControlNet.
        
        Args:
            camera_info (dict): The camera params from frontend
            
        Returns:
            Tuple[PIL.Image.Image, PIL.Image.Image, str]: (dlss_off_pil, dlss_on_pil, status_message)
        """
        # Parse Camera
        cam = Camera.from_threejs(camera_info)
        self.scene.set_camera(cam)
        
        # Render low-res FBO natively
        color_np, depth_np = self._render_scene_to_fbo(cam)
        
        if self.dlss_upscaler is None:
             self.dlss_upscaler = AIUpscaler(scale_factor=self.scale_factor)
        
        # Check for weights and load (this may change self.dlss_upscaler.scale)
        weight_path = os.path.join(os.path.dirname(__file__), "..", "model", "espcn_weights.pth")
        has_weights = os.path.exists(weight_path)
        
        if has_weights:
            self.dlss_upscaler.load_weights(weight_path)
        
        actual_scale = self.dlss_upscaler.scale
        
        # DLSS OFF (Bilinear) - use same scale as the AI model for fair comparison
        bilinear_np = self.dlss_upscaler.upscale_bilinear(color_np)
        dlss_off_pil = PIL.Image.fromarray((bilinear_np * 255.0).clip(0, 255).astype(np.uint8))
        
        if not has_weights:
            return dlss_off_pil, None, f"state：no pth file, show Bilinear upscaled result"
             
        # DLSS ON (torch.nn)
        ai_np = self.dlss_upscaler.upscale(color_np, depth_np)
        dlss_on_pil = PIL.Image.fromarray((ai_np * 255.0).clip(0, 255).astype(np.uint8))
        
        return dlss_off_pil, dlss_on_pil, f"Render Completed (ESPCN {actual_scale}x Upscaled from .pth)"

    def _train_dlss(self, camera_trajectory: list, epochs: int, lr: float, scale: int):

        hr_colors = []
        hr_depths = []
        lr_colors = []
        lr_depths = []
        
        original_res = self.render_res
        lr_res = original_res  # e.g. 256
        hr_res = lr_res * scale  # e.g. 256 * 2 = 512
        
        def _release_fbo():
            if self.fbo:
                self.fbo.release()
                self.color_tex.release()
                self.depth_tex.release()
                self.fbo = None
                self.color_tex = None
                self.depth_tex = None
            
        try:
            for camera_info in camera_trajectory:
                cam = Camera.from_threejs(camera_info)
                
                # Render LR natively
                self.render_res = lr_res
                _release_fbo()
                lr_color, lr_depth = self._render_scene_to_fbo(cam)
                lr_colors.append(lr_color)
                lr_depths.append(lr_depth)
                
                # Render HR natively
                self.render_res = hr_res
                _release_fbo()
                hr_color, hr_depth = self._render_scene_to_fbo(cam)
                hr_colors.append(hr_color)
                hr_depths.append(hr_depth)
        finally:
            # Restore original res
            self.render_res = original_res
            _release_fbo()
                
        if not hr_colors:
            return "Error: No frames recorded in trajectory."
            
        # Stack into batches: (N, H, W, C)
        hr_colors_batch = np.stack(hr_colors, axis=0)
        hr_depths_batch = np.stack(hr_depths, axis=0)
        lr_colors_batch = np.stack(lr_colors, axis=0)
        lr_depths_batch = np.stack(lr_depths, axis=0)
                
        if self.dlss_upscaler is None:
             self.dlss_upscaler = AIUpscaler(scale_factor=self.scale_factor)
             
        log_output = self.dlss_upscaler.train_step(
            hr_colors_batch, hr_depths_batch,
            epochs, lr, scale,
            lr_colors=lr_colors_batch, lr_depths=lr_depths_batch,
        )
        return log_output

    def _render_benchmark(self, camera_info: dict):
        """
        Render a benchmark comparison grid: Low-Res, Nearest, Bilinear, ESPCN, Ground Truth.
        Computes PSNR/SSIM for each method vs HR ground truth.

        Returns:
            Tuple[PIL.Image.Image, str]: (comparison_grid_pil, status_message)
        """

        from metrics import save_comparison, compute_metrics

        print("[Benchmark] Starting benchmark...")
        cam = Camera.from_threejs(camera_info)
        print(f"[Benchmark] Camera parsed: {cam}")

        # 1. Render Low-Res
        color_lr, depth_lr = self._render_scene_to_fbo(cam)
        print(f"[Benchmark] LR rendered: {color_lr.shape}")

        # 2. Render High-Res Ground Truth (switch FBO resolution)
        original_res = self.render_res
        self.render_res = self.display_res  # HR = display_res (e.g. 512)

        if self.fbo:
            self.fbo.release()
            self.color_tex.release()
            self.depth_tex.release()
            self.fbo = None
            self.color_tex = None
            self.depth_tex = None

        color_hr, _ = self._render_scene_to_fbo(cam)
        print(f"[Benchmark] HR rendered: {color_hr.shape}")

        # Restore LR resolution
        self.render_res = original_res
        if self.fbo:
            self.fbo.release()
            self.color_tex.release()
            self.depth_tex.release()
            self.fbo = None
            self.color_tex = None
            self.depth_tex = None

        # 3. Generate upscaled variants
        if self.dlss_upscaler is None:
            self.dlss_upscaler = AIUpscaler(scale_factor=self.scale_factor)

        bilinear_np = self.dlss_upscaler.upscale_bilinear(color_lr)
        print(f"[Benchmark] Bilinear: {bilinear_np.shape}")

        # ESPCN upscale (load weights if available)
        weight_path = os.path.join(os.path.dirname(__file__), "..", "model", "espcn_weights.pth")
        has_weights = os.path.exists(weight_path)
        print(f"[Benchmark] Weight path: {weight_path}, exists: {has_weights}")

        if has_weights:
            self.dlss_upscaler.load_weights(weight_path)
        espcn_np = self.dlss_upscaler.upscale(color_lr, depth_lr)
        print(f"[Benchmark] ESPCN: {espcn_np.shape}")

        # 4. Compute metrics text
        print(f"[Benchmark] Computing metrics: bilinear {bilinear_np.shape} vs HR {color_hr.shape}")
        m_bilinear = compute_metrics(bilinear_np, color_hr)
        m_espcn = compute_metrics(espcn_np, color_hr)
        print(f"[Benchmark] Bilinear PSNR={m_bilinear['psnr']:.2f}, ESPCN PSNR={m_espcn['psnr']:.2f}")

        status_lines = [
            f"Bilinear  -> PSNR: {m_bilinear['psnr']:.2f} dB, SSIM: {m_bilinear['ssim']:.4f}",
            f"ESPCN     -> PSNR: {m_espcn['psnr']:.2f} dB, SSIM: {m_espcn['ssim']:.4f}",
        ]
        if not has_weights:
            status_lines.append("(No .pth found, ESPCN used random weights)")

        # 5. Generate comparison grid
        print("[Benchmark] Generating comparison grid...")
        grid_pil = save_comparison(
            low_res=color_lr,
            espcn=espcn_np,
            bilinear=bilinear_np,
            ground_truth=color_hr,
        )
        print(f"[Benchmark] Grid generated: {grid_pil.size}")

        return grid_pil, "\n".join(status_lines)