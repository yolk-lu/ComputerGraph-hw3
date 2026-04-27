
import os
import torch
from pipeline.utils.utils import utils
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel



class Model:
    """load model"""
    def __init__(self):
        self.pipe = None

    def load(self):
        if self.pipe is not None:
            return # already loaded
        print("Loading Pretrained ControlNet Models...")
        
        # 設定cache目錄為專案下的 .cache 資料夾
        cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".cache"))
        os.makedirs(cache_dir, exist_ok=True)
        
        try:
            
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-depth", 
                torch_dtype=torch.float16,
                local_files_only=True,
                cache_dir=cache_dir
            )
            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5", 
                controlnet=controlnet, 
                torch_dtype=torch.float16,
                local_files_only=True,
                cache_dir=cache_dir
            )
            print("Successfully loaded from local cache.")
        except Exception as e:
            print(f"Local cache not found or incomplete: {e}. Downloading from Hugging Face...")
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-depth", 
                torch_dtype=torch.float16,
                cache_dir=cache_dir
            )
            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5", 
                controlnet=controlnet, 
                torch_dtype=torch.float16,
                cache_dir=cache_dir
            )
        
        self.pipe.to("cuda")

    def generate(self, depth_image, prompt, negative_prompt="low quality, blurry, flat, ugly, text, out of focus, distorted"):
        """generate image"""
        if self.pipe is None:
            self.load()
            
        print(f"Generating image with prompt: {prompt}")
        result = self.pipe(
            prompt, 
            negative_prompt=negative_prompt, 
            image=depth_image,
            num_inference_steps=20,
        ).images[0]
        
        return result