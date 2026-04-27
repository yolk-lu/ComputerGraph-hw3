import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import base64
import urllib.parse
from io import BytesIO
import PIL.Image
import PIL.ImageOps
import html
import json

try:
    import gradio as gr
except ImportError:
    print("pip install gradio")
    exit()
import sys
# app.py is in the root directory, add the root directory to sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

# pipeline 
from pipeline.scene.Scene import Scene
from pipeline.model.model import Model
from pipeline.renderer.Renderer import Renderer
from pipeline.utils.utils import utils



# create global renderer
renderer_engine = Renderer() if 'Renderer' in locals() else None

def run_controlnet_from_base64(b64_depth, prompt, camera_json):
    """
    Execute AI rendering from base64 depth and camera info.
    Args:
        b64_depth (str): Base64 encoded depth image from Three.js
        prompt (str): Prompt for ControlNet
        camera_json (str): JSON string containing camera parameters
    Returns:
        tuple[PIL.Image, PIL.Image, str]: DLSS Image, ControlNet Image, Status Message
    """
    if renderer_engine is None:
        return None, None, "Renderer is not imported correctly."
    try:
        import json
        camera_info = json.loads(camera_json) if camera_json else None
        
        # We will split this inside Renderer.py into DLSS and ControlNet
        dlss_img, controlnet_img = renderer_engine.render_pipeline(b64_depth, prompt, camera_info)
        
        return dlss_img, controlnet_img, "Render complete!"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, f"Rendering failed: {str(e)}"

def render_scene(model_file_paths):
    """
    UI triggered render function (supports multiple files: .obj + .mtl)
    """

    if not model_file_paths:
        return "請先上傳 3D 模型", None
    
    # handle single or multiple file uploads
    if not isinstance(model_file_paths, list):
        model_file_paths = [model_file_paths]
    
    # collect all uploaded file paths by extension
    uploaded_files = {}
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tga', '.gif'}
    uploaded_images = {}  # filename -> file_path
    
    for fp in model_file_paths:
        real = fp.name if hasattr(fp, 'name') else str(fp)
        ext = os.path.splitext(real)[1].lower()
        if ext in image_exts:
            uploaded_images[os.path.basename(real).lower()] = real
        else:
            uploaded_files[ext] = real
    
    # find the GLB file
    glb_path = uploaded_files.get('.glb')
    if not glb_path:
        return "請上傳 .glb 檔案", None
    
    # save GLB to input_model folder
    target_path = os.path.abspath(utils.save_uploaded_model(glb_path))
    
    model_count = renderer_engine.prepare_scene(target_path)
    scene_info = f"場景目前模型數: {model_count or 0}。"
    mode = "free_cam"
    
    # Pass GLB as Base64 to a hidden textbox to avoid OOM in srcdoc and 404 in Gradio file server
    try:
        with open(target_path, 'rb') as f:
            glb_b64 = base64.b64encode(f.read()).decode('ascii')
            glb_data_b64 = f"data:model/gltf-binary;base64,{glb_b64}"
    except Exception as e:
        return f"無法讀取 GLB 檔案: {target_path}\\n{str(e)}", None, None
        
    
    # frontend resources
    base_dir = os.path.join(os.path.dirname(__file__), 'web_viewer', 'web')
    
    with open(os.path.join(base_dir, 'three_style.css'), 'r', encoding='utf-8') as f:
        inline_css = f.read()
        
    with open(os.path.join(base_dir, 'three_script.js'), 'r', encoding='utf-8') as f:
        inline_js = f.read()
    
    template_path = os.path.join(base_dir, "three_viewer.html")
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            html_str = f.read()
    except Exception as e:
        return f"無法讀取 three_viewer.html: {str(e)}", None

    # inject configs as JSON (Skip GLB_DATA to avoid srcdoc OOM)
    html_str = html_str.replace("[[GLB_DATA_PLACEHOLDER]]", "null")
    html_str = html_str.replace("[[CAMERA_MODE_PLACEHOLDER]]", mode)
    html_str = html_str.replace("/* INLINE_CSS_PLACEHOLDER */", inline_css)
    html_str = html_str.replace("/* INLINE_JS_PLACEHOLDER */", inline_js)
    
    escaped_html = html.escape(html_str)
    iframe_html = f'<iframe srcdoc="{escaped_html}" style="width: 100%; height: 75vh; border: 1px solid rgba(255,255,255,0.1); border-radius: 12px;"></iframe>'
    
    return f"{scene_info}\nstate => 3D model loaded : {mode}", iframe_html, glb_data_b64

# point to style.css
css_path = os.path.join(os.path.dirname(__file__), 'web_viewer', 'web', "style.css")

with gr.Blocks(title="CG HW3") as demo:
    gr.Markdown(
        """
        <div style="text-align: center; margin-bottom: 20px;">
            <h1 style="background: linear-gradient(90deg, #c084fc, #60a5fa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.5em; font-weight: 800; margin-bottom: 5px;"> CG HW3 3D Web Viewer</h1>
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("控制面板 (Settings)")
            
            
            model_input = gr.File(label="上傳 3D 模型 (.glb)", file_count="multiple")
            
            with gr.Row():
                render_btn = gr.Button("Generate Scene", elem_classes="primary")
            
            status_out = gr.Textbox(label="系統輸出狀態", interactive=False)
            
            gr.Markdown("---")
            gr.Markdown("Training Pipeline (DLSS ESPCN)")
            with gr.Row():
                train_epochs = gr.Slider(minimum=1, maximum=200, value=50, step=1, label="Training Epochs")
                train_lr = gr.Slider(minimum=0.0001, maximum=0.01, value=0.001, step=0.0001, label="Learning Rate")
                train_scale = gr.Slider(minimum=2, maximum=4, value=2, step=1, label="Upscale 倍率")
            with gr.Row():
                record_start_btn = gr.Button("Start Recording Trajectory", elem_classes="secondary")
                record_stop_btn = gr.Button("Stop Recording & Train DLSS", elem_classes="primary")
            train_logs = gr.Textbox(label="Training Logs", interactive=False, lines=5)
            
            gr.Markdown("---")
            gr.Markdown("Render Pipeline (DLSS)")
            with gr.Row():
                dlss_render_btn = gr.Button("Render (OpenGL & DLSS)", elem_classes="primary")
            with gr.Row():
                dlss_off_image = gr.Image(label="DLSS OFF (Bilinear)", type="pil")
                dlss_on_image = gr.Image(label="DLSS ON (ESPCN Upscale)", type="pil")
                

            gr.Markdown("---")
            gr.Markdown("Stylize Pipeline (ControlNet)")
            prompt_input = gr.Textbox(label="Prompt", placeholder="Abstract crystal geometric, 8k, cinematic lighting")
            stylize_btn = gr.Button("Stylize")
            with gr.Row():
                output_image = gr.Image(label="ControlNet Result", type="pil")
            
            # JS to python (hidden components)
            gr.HTML("<style>#hidden_components { display: none !important; }</style>")
            with gr.Column(elem_id="hidden_components"):
                hidden_depth = gr.Textbox(elem_id="hidden_depth")
                hidden_camera = gr.Textbox(elem_id="hidden_camera")
                hidden_submit_btn = gr.Button("hidden submit", elem_id="hidden_submit_btn")
                hidden_render_camera = gr.Textbox(elem_id="hidden_render_camera")
                hidden_render_submit = gr.Button("hidden render submit", elem_id="hidden_render_submit")
                
                hidden_trajectory = gr.Textbox(elem_id="hidden_trajectory")
                hidden_train_submit = gr.Button("hidden train submit", elem_id="hidden_train_submit")
                
                hidden_glb_b64 = gr.Textbox(elem_id="hidden_glb_b64")
                
            
        with gr.Column(scale=2):
            viewer_html = gr.HTML(label="3D Viewer", value="<div style='padding: 20px;'>請先上傳模型並點擊「產生場景」。</div>")

    render_btn.click(
        fn=render_scene, 
        inputs=[model_input], 
        outputs=[status_out, viewer_html, hidden_glb_b64]
    )

    hidden_glb_b64.change(
        fn=None,
        inputs=[hidden_glb_b64],
        outputs=None,
        js="""
        function(b64) {
            if (!b64 || b64.length < 100) return;
            console.log('透過 postMessage 傳送 GLB Base64 資料...');
            // 等待 iframe 載入
            setTimeout(() => {
                const iframes = document.querySelectorAll('iframe');
                iframes.forEach(iframe => {
                    iframe.contentWindow.postMessage({
                        type: 'load_glb_b64', 
                        data: b64
                    }, '*');
                });
            }, 500);
        }
        """
    )

    # --- JS Event Hook for ControlNet Stylize ---
    stylize_btn.click(
        fn=None, 
        inputs=None, outputs=None,
        js="""
        function() {
            const iframes = document.querySelectorAll('iframe');
            iframes.forEach(iframe => {
                iframe.contentWindow.postMessage({type: 'request_depth'}, '*');
            });
            
            if (!window.depthListenerAdded) {
                window.addEventListener('message', function(e) {
                    if (e.data && e.data.type === 'depth_result') {
                        // depth map
                        const hiddenDepthDiv = document.getElementById('hidden_depth');
                        if (hiddenDepthDiv) {
                            const textarea = hiddenDepthDiv.querySelector('textarea');
                            if (textarea) {
                                textarea.value = e.data.depth;
                                textarea.dispatchEvent(new Event('input', { bubbles: true }));
                            }
                        }
                        // camera info
                        const hiddenCameraDiv = document.getElementById('hidden_camera');
                        if (hiddenCameraDiv) {
                            const textarea = hiddenCameraDiv.querySelector('textarea');
                            if (textarea) {
                                textarea.value = JSON.stringify(e.data.camera || {});
                                textarea.dispatchEvent(new Event('input', { bubbles: true }));
                            }
                        }
                        // trigger backend
                        setTimeout(() => {
                            const hiddenBtn = document.querySelector('#hidden_submit_btn button') || document.getElementById('hidden_submit_btn');
                            if(hiddenBtn) hiddenBtn.click();
                        }, 100);
                    }
                });
                window.depthListenerAdded = true;
            }
        }
        """
    )
    
    hidden_submit_btn.click(
        fn=run_controlnet_from_base64,
        inputs=[hidden_depth, prompt_input, hidden_camera],
        outputs=[dlss_on_image, output_image, status_out] # Reused dlss_on_image to show both
    )

    # --- JS Event Hook for DLSS Render ---
    dlss_render_btn.click(
        fn=None,
        inputs=None, outputs=None,
        js="""
        function() {
            console.log('click Render button， broadcast request_camera');
            window.PENDING_ACTION = 'render';
            const iframes = document.querySelectorAll('iframe');
            iframes.forEach(iframe => {
                iframe.contentWindow.postMessage({type: 'request_camera'}, '*');
            });
            
            if (!window.cameraListenerAdded) {
                console.log('register response receiver (cameraListenerAdded)');
                window.addEventListener('message', function(e) {
                    if (e.data && e.data.type === 'camera_result') {
                        console.log('received camera_result, action:', window.PENDING_ACTION);
                        
                        if (window.PENDING_ACTION === 'benchmark') {
                            const div = document.getElementById('hidden_benchmark_camera');
                            if (div) {
                                const ta = div.querySelector('textarea');
                                if (ta) {
                                    ta.value = JSON.stringify(e.data.camera || {});
                                    ta.dispatchEvent(new Event('input', { bubbles: true }));
                                }
                            }
                            setTimeout(() => {
                                const btn = document.querySelector('#hidden_benchmark_submit button') || document.getElementById('hidden_benchmark_submit');
                                if(btn) btn.click();
                            }, 100);
                        } else {
                            const div = document.getElementById('hidden_render_camera');
                            if (div) {
                                const ta = div.querySelector('textarea');
                                if (ta) {
                                    ta.value = JSON.stringify(e.data.camera || {});
                                    ta.dispatchEvent(new Event('input', { bubbles: true }));
                                }
                            }
                            setTimeout(() => {
                                const btn = document.querySelector('#hidden_render_submit button') || document.getElementById('hidden_render_submit');
                                if(btn) btn.click();
                            }, 100);
                        }
                        window.PENDING_ACTION = null;
                    } else if (e.data && e.data.type === 'trajectory_result') {
                        console.log('收到 trajectory_result', e.data.trajectory.length, 'frames');
                        const hiddenTrajDiv = document.getElementById('hidden_trajectory');
                        if (hiddenTrajDiv) {
                            const textarea = hiddenTrajDiv.querySelector('textarea');
                            if (textarea) {
                                textarea.value = JSON.stringify(e.data.trajectory || []);
                                textarea.dispatchEvent(new Event('input', { bubbles: true }));
                            }
                        }
                        setTimeout(() => {
                            const hiddenBtn = document.querySelector('#hidden_train_submit button') || document.getElementById('hidden_train_submit');
                            if(hiddenBtn) hiddenBtn.click();
                        }, 100);
                    }
                });
                window.cameraListenerAdded = true;
            }
        }
        """
    )
    

    
    # --- JS Event Hooks for DLSS Train Trajectory ---
    record_start_btn.click(
        fn=None, inputs=None, outputs=None,
        js="""
        function() {
            console.log('Click Start Recording，Broadcast start_record');
            const iframes = document.querySelectorAll('iframe');
            iframes.forEach(iframe => {
                iframe.contentWindow.postMessage({type: 'start_record'}, '*');
            });
        }
        """
    )

    record_stop_btn.click(
        fn=None, inputs=None, outputs=None,
        js="""
        function() {
            console.log('click Stop Recording，broadcast stop_record');
            const iframes = document.querySelectorAll('iframe');
            iframes.forEach(iframe => {
                iframe.contentWindow.postMessage({type: 'stop_record'}, '*');
            });
            
            if (!window.cameraListenerAdded) {
                console.log('register listener (cameraListenerAdded)');
                window.addEventListener('message', function(e) {
                    if (e.data && e.data.type === 'camera_result') {
                        console.log('receive camera_result，trigger dlss training step', e.data.camera);
                        const hiddenCameraDiv = document.getElementById('hidden_render_camera');
                        if (hiddenCameraDiv) {
                            const textarea = hiddenCameraDiv.querySelector('textarea');
                            if (textarea) {
                                textarea.value = JSON.stringify(e.data.camera || {});
                                textarea.dispatchEvent(new Event('input', { bubbles: true }));
                            }
                        }
                        setTimeout(() => {
                            const hiddenBtn = document.querySelector('#hidden_render_submit button') || document.getElementById('hidden_render_submit');
                            if(hiddenBtn) hiddenBtn.click();
                        }, 100);
                    } else if (e.data && e.data.type === 'trajectory_result') {
                        console.log('receive trajectory_result, trigger dlss training step', e.data.trajectory.length, 'frames');
                        const hiddenTrajDiv = document.getElementById('hidden_trajectory');
                        if (hiddenTrajDiv) {
                            const textarea = hiddenTrajDiv.querySelector('textarea');
                            if (textarea) {
                                textarea.value = JSON.stringify(e.data.trajectory || []);
                                textarea.dispatchEvent(new Event('input', { bubbles: true }));
                            }
                        }
                        setTimeout(() => {
                            const hiddenBtn = document.querySelector('#hidden_train_submit button') || document.getElementById('hidden_train_submit');
                            if(hiddenBtn) hiddenBtn.click();
                        }, 100);
                    }
                });
                window.cameraListenerAdded = true;
            }
        }
        """
    )

    def trigger_dlss_comparison(camera_json: str):
        print(f"\nloadRender 請求")
        print(f"load Camera JSON 長度: {len(camera_json) if camera_json else 0}")
        if not camera_json:
            print("沒有收到Three.js傳來的相機資料")
            return None, None, "No camera data from frontend!"
        import json
        info = json.loads(camera_json)
        print(f"ready to renderer_engine.render_dlss_comparison()...")
        try:
            dlss_off, dlss_on, status_msg = renderer_engine.render_dlss_comparison(info)
            print(f"Render Completed! Status: {status_msg}")
            return dlss_off, dlss_on, status_msg
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, None, f"Render Failed: {e}"

    hidden_render_submit.click(
        fn=trigger_dlss_comparison,
        inputs=[hidden_render_camera],
        outputs=[dlss_off_image, dlss_on_image, status_out]
    )



    def trigger_dlss_train(trajectory_json: str, epochs: int, lr: float, scale: int):
        print(f"\nreceive Train request！")
        if not trajectory_json:
            return "No trajectory data from frontend!"
        import json
        trajectory = json.loads(trajectory_json)
        try:
            log_output = renderer_engine.train_dlss(trajectory, int(epochs), float(lr), int(scale))
            return log_output
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Training Failed: {e}"

    hidden_train_submit.click(
        fn=trigger_dlss_train,
        inputs=[hidden_trajectory, train_epochs, train_lr, train_scale],
        outputs=[train_logs]
    )

if __name__ == "__main__":

    print(" 啟動 Gradio Web Viewer...")
    print(" 系統將會輸出本機 URL (如 http://127.0.0.1:8000)")
    print(" share=True 產生可分享的公開連結")

    # allowed_paths 沒啥用 share = True 產生公開連結
    input_model_dir = os.path.join(PROJECT_ROOT, "input_model")
    # print(f"allowed_paths = [{input_model_dir}]")
    demo.launch(server_name="127.0.0.1", server_port=8000, share=True, allowed_paths=[input_model_dir], css=css_path)
