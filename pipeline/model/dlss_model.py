import numpy as np
import os
import gc

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True

    def gaussian(window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(window_size, channel):
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def ssim(img1, img2, window_size=11, window=None):
        channel = img1.size(1)
        if window is None:
            window = create_window(window_size, channel).to(img1.device)
        
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()
        
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    import torch.optim as optim
    class ESPCN(nn.Module):
        """
        Efficient Sub-Pixel Convolutional Neural Network (Shi et al., 2016).
        A lightweight super-resolution model suitable for real-time inference.

        Input:  (B, C_in, H, W) - low-res image, C_in = 4 (RGB + Depth)
        Output: (B, 3, H*scale, W*scale) - high-res RGB
        """

        def __init__(self, scale_factor: int = 2, in_channels: int = 4, num_features: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, num_features, 5, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_features, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 3 * scale_factor ** 2, 3, padding=1),
                nn.PixelShuffle(scale_factor),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass of the ESPCN model.
            Args: 
            x (torch.Tensor): Intput tensor of shape (B, C_in, H, W).
                Values should be normalized between [0, 1].
            Returns:
                torch.Tensor: Output upscaled RGB tensor of shape (B, 3, H*scale, W*scale).
                Values are clamped between [0, 1].
            """
            return torch.clamp(self.net(x), 0.0, 1.0)

    class AIUpscaler:
        """
        Wraps the ESPCN PyTorch model for integration with the rendering loop.
        Handles numpy-to-tensor conversions and device placement.
        """

        def __init__(self, scale_factor: int = 2, device: str = None):
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.scale = scale_factor
            self.model = ESPCN(scale_factor=scale_factor, in_channels=4).to(self.device)
            self.model.eval()
            # print(f"Initialized on {self.device}, Scale: {scale_factor}x")

        def load_weights(self, path: str):
            """
            Load pretrained weights into the ESPCN model.
            Auto-detects the scale factor from the checkpoint and rebuilds the model if needed.
            """
            try:
                state_dict = torch.load(path, map_location=self.device)
                
                # Infer scale from the PixelShuffle layer weight shape
                # net.4 is the Conv2d before PixelShuffle: out_channels = 3 * scale^2
                if 'net.4.weight' in state_dict:
                    out_channels = state_dict['net.4.weight'].shape[0]
                    inferred_scale = int((out_channels / 3) ** 0.5)
                    if inferred_scale != self.scale:
                        print(f"Scale mismatch: model={self.scale}x, weights={inferred_scale}x. Rebuilding model...")
                        self.scale = inferred_scale
                        self.model = ESPCN(scale_factor=inferred_scale, in_channels=4).to(self.device)
                
                self.model.load_state_dict(state_dict)
                self.model.eval()
                print(f"Loaded weights from {path} (scale={self.scale}x)")
            except Exception as e:
                print(f"Error loading weights: {e}")

        def train_step(self, hr_colors: np.ndarray, hr_depths: np.ndarray, epochs: int, lr: float, scale: int, batch_size: int = 4, lr_colors: np.ndarray = None, lr_depths: np.ndarray = None):
            """
            Train the ESPCN model on paired LR/HR images.
            If lr_colors/lr_depths are provided, uses natively-rendered LR inputs.
            Otherwise falls back to bicubic downsampling from HR (legacy mode).
            """
  
            
            if len(hr_colors) == 0:
                return "No data to train on."
                
            # Change scale if needed
            # if self.scale != scale:
            self.scale = scale
            # print(scale)
            # print(self.scale)

            self.model = ESPCN(scale_factor=scale, in_channels=4).to(self.device)
            print(f"Initialized on {self.device}, Scale: {scale}x")    
            self.model.train()
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
            
            n, h, w = hr_colors.shape[:3]
            lr_h, lr_w = h // scale, w // scale
            
            # Convert full dataset to tensors
            hr_color_t = torch.from_numpy(hr_colors).permute(0, 3, 1, 2).to(self.device).float()  # (N, 3, H, W)
            
            # Use natively-rendered LR if provided, otherwise fall back to bicubic downsampling
            if lr_colors is not None and lr_depths is not None:
                lr_color_t = torch.from_numpy(lr_colors).permute(0, 3, 1, 2).to(self.device).float()  # (N, 3, lr_h, lr_w)
                lr_depth_t = torch.from_numpy(lr_depths).unsqueeze(1).to(self.device).float()          # (N, 1, lr_h, lr_w)
                print(f"Using natively-rendered LR inputs: {lr_color_t.shape}")
            else:
                hr_depth_t = torch.from_numpy(hr_depths).unsqueeze(1).to(self.device).float()
                lr_color_t = F.interpolate(hr_color_t, size=(lr_h, lr_w), mode='bicubic', align_corners=False).clamp(0, 1)
                lr_depth_t = F.interpolate(hr_depth_t, size=(lr_h, lr_w), mode='bilinear', align_corners=False).clamp(0, 1)
                print(f"Using bicubic-downsampled LR inputs: {lr_color_t.shape}")
            
            lr_input_t = torch.cat([lr_color_t, lr_depth_t], dim=1) # (N, 4, lr_h, lr_w)
            
            dataset_size = n
            logs = []
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                epoch_mse = 0.0
                epoch_ssim = 0.0
                
                # Shuffle dataset manually
                indices = torch.randperm(dataset_size)
                
                for i in range(0, dataset_size, batch_size):
                    batch_idx = indices[i:i+batch_size]
                    
                    b_lr_input = lr_input_t[batch_idx]
                    b_hr_color = hr_color_t[batch_idx]
                    
                    optimizer.zero_grad()
                    
                    # Forward
                    pred_hr = self.model(b_lr_input)
                    
                    if pred_hr.shape != b_hr_color.shape:
                        pred_hr = F.interpolate(pred_hr, size=(h, w), mode='bicubic', align_corners=False)
                    
                    # Loss
                    mse_loss = F.mse_loss(pred_hr, b_hr_color)
                    ssim_val = ssim(pred_hr, b_hr_color)
                    loss = mse_loss + (1.0 - ssim_val) * 0.1
                    
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item() * len(batch_idx)
                    epoch_mse += mse_loss.item() * len(batch_idx)
                    epoch_ssim += ssim_val.item() * len(batch_idx)
                
                # Average loss for the epoch
                epoch_loss /= dataset_size
                epoch_mse /= dataset_size
                epoch_ssim /= dataset_size
                
                if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == epochs - 1:
                    log_msg = f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f} (MSE: {epoch_mse:.4f}, SSIM: {epoch_ssim:.4f})"
                    logs.append(log_msg)
                    print(log_msg)
            
            # Save weights
            save_path = os.path.join(os.path.dirname(__file__), "espcn_weights.pth")
            torch.save(self.model.state_dict(), save_path)
            logs.append(f"Weights saved to {save_path}")
            
            self.model.eval()
            
            # Free memory
            del hr_color_t, lr_color_t, lr_depth_t, lr_input_t
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
            
            return "\\n".join(logs)


        @torch.no_grad()
        def upscale(self, color_np: np.ndarray, depth_np: np.ndarray) -> np.ndarray:
            """
            Upscales a low-resolution frame using the ESPCN model.

            Args:
                color_np (np.ndarray): The low-res color image. Shape must be (H, W, 3). 
                                       Data type float32, range [0, 1].
                depth_np (np.ndarray): The low-res depth map. Shape must be (H, W). 
                                       Data type float32, range [0, 1] (linearized depth).

            Returns:
                np.ndarray: The upscaled high-res RGB image. 
                            Shape is (H * scale, W * scale, 3), range [0, 1], type float32.
            """
            h, w = color_np.shape[:2]
            
            # Step 1: Concatenate color (H,W,3) and depth (H,W,1)
            depth_expanded = np.expand_dims(depth_np, axis=-1)
            fbo_combined = np.concatenate([color_np, depth_expanded], axis=-1) # (H, W, 4)
            
            # Step 2: Transpose to (C, H, W) and add batch dim -> (1, 4, H, W)
            fbo_tensor = np.expand_dims(fbo_combined.transpose(2, 0, 1), axis=0)
            
            # Step 3: Convert to torch tensor
            input_tensor = torch.from_numpy(fbo_tensor).contiguous().to(self.device).float()
            
            # Step 4: Inference
            output_tensor = self.model(input_tensor)
            
            # Step 5: Convert back to numpy (H_out, W_out, 3)
            # output shape was (1, 3, H_out, W_out)
            output_np = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            return output_np
            
        def upscale_bilinear(self, color_np: np.ndarray) -> np.ndarray:
            """
            Baseline Upscale logic (Bilinear) without depth or AI processing.
            """
            import PIL.Image
            h, w = color_np.shape[:2]
            img = PIL.Image.fromarray((color_np * 255.0).clip(0, 255).astype(np.uint8))
            img = img.resize((w * self.scale, h * self.scale), PIL.Image.BILINEAR)
            return np.array(img).astype(np.float32) / 255.0
else:
    class AIUpscaler:
        """
        Fallback AI Upscaler for environments without PyTorch.
        It uses PIL for simple Bilinear upscaling.
        """
        def __init__(self, scale_factor: int = 2, device: str = None):
            self.scale = scale_factor
            print("PyTorch not found. Using CPU Bilinear Fallback.")

        def upscale(self, color_np: np.ndarray, depth_np: np.ndarray) -> np.ndarray:
            """
            Fallback Upscale logic (Bilinear).

            Args:
                color_np (np.ndarray): Low-res image (H, W, 3) in [0, 1].
                depth_np (np.ndarray): Depth image (H, W).

            Returns:
                np.ndarray: The upscaled High-res RGB image.
            """
            from PIL import Image
            h, w = color_np.shape[:2]
            img = Image.fromarray((color_np * 255.0).clip(0, 255).astype(np.uint8))
            img = img.resize((w * self.scale, h * self.scale), Image.BILINEAR)
            return np.array(img).astype(np.float32) / 255.0

        def upscale_bilinear(self, color_np: np.ndarray) -> np.ndarray:
            """
            Baseline Upscale logic (Bilinear) without depth or AI processing.
            """
            return self.upscale(color_np, np.zeros_like(color_np[:,:,0]))
