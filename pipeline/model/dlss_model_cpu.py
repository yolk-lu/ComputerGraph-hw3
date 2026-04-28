"""
CPU-only version of the DLSS ESPCN model.
Identical architecture and API as dlss_model.py, but forces torch device to 'cpu'.
Use this on systems without CUDA (e.g. macOS, CI environments).

Usage:
    from pipeline.model.dlss_model_cpu import AIUpscaler
    # (drop-in replacement for the GPU version)
"""

import numpy as np
import os
import gc
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ── SSIM Loss (differentiable, torch-based) ──

def gaussian(window_size, sigma):
    gauss = torch.Tensor([np.exp(-(x - window_size//2)**2 / float(2*sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()

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


# ── ESPCN Model ──

class ESPCN(nn.Module):
    """
    Efficient Sub-Pixel Convolutional Neural Network (Shi et al., 2016).
    CPU-only version — identical architecture to the GPU variant.

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
        return torch.clamp(self.net(x), 0.0, 1.0)


# ── AIUpscaler (CPU-only) ──

class AIUpscaler:
    """
    Wraps the ESPCN PyTorch model for CPU-only inference and training.
    Drop-in replacement for the GPU version in dlss_model.py.
    """

    def __init__(self, scale_factor: int = 2, device: str = None):
        self.device = "cpu"  # Always CPU
        self.scale = scale_factor
        self.model = ESPCN(scale_factor=scale_factor, in_channels=4).to(self.device)
        self.model.eval()

    def load_weights(self, path: str):
        """
        Load pretrained weights into the ESPCN model.
        Auto-detects the scale factor from the checkpoint and rebuilds the model if needed.
        """
        try:
            state_dict = torch.load(path, map_location="cpu")

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
            print(f"Loaded weights from {path} (scale={self.scale}x, device=cpu)")
        except Exception as e:
            print(f"Error loading weights: {e}")

    def train_step(self, hr_colors: np.ndarray, hr_depths: np.ndarray, epochs: int, lr: float, scale: int, batch_size: int = 4, lr_colors: np.ndarray = None, lr_depths: np.ndarray = None):
        """
        Train the ESPCN model on paired LR/HR images (CPU-only).
        If lr_colors/lr_depths are provided, uses natively-rendered LR inputs.
        Otherwise falls back to bicubic downsampling from HR (legacy mode).
        """
        if len(hr_colors) == 0:
            return "No data to train on."

        self.scale = scale
        self.model = ESPCN(scale_factor=scale, in_channels=4).to(self.device)
        print(f"Initialized on cpu, Scale: {scale}x")
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        n, h, w = hr_colors.shape[:3]
        lr_h, lr_w = h // scale, w // scale

        # Convert full dataset to tensors (always on CPU)
        hr_color_t = torch.from_numpy(hr_colors).permute(0, 3, 1, 2).float()

        # Use natively-rendered LR if provided, otherwise fall back to bicubic downsampling
        if lr_colors is not None and lr_depths is not None:
            lr_color_t = torch.from_numpy(lr_colors).permute(0, 3, 1, 2).float()
            lr_depth_t = torch.from_numpy(lr_depths).unsqueeze(1).float()
            print(f"Using natively-rendered LR inputs: {lr_color_t.shape}")
        else:
            hr_depth_t = torch.from_numpy(hr_depths).unsqueeze(1).float()
            lr_color_t = F.interpolate(hr_color_t, size=(lr_h, lr_w), mode='bicubic', align_corners=False).clamp(0, 1)
            lr_depth_t = F.interpolate(hr_depth_t, size=(lr_h, lr_w), mode='bilinear', align_corners=False).clamp(0, 1)
            print(f"Using bicubic-downsampled LR inputs: {lr_color_t.shape}")

        lr_input_t = torch.cat([lr_color_t, lr_depth_t], dim=1)  # (N, 4, lr_h, lr_w)

        dataset_size = n
        logs = []

        pbar = tqdm(range(epochs), desc="Training DLSS (CPU)", unit="epoch")

        for epoch in pbar:
            epoch_loss = 0.0
            epoch_mse = 0.0
            epoch_ssim = 0.0

            indices = torch.randperm(dataset_size)

            for i in range(0, dataset_size, batch_size):
                batch_idx = indices[i:i+batch_size]

                b_lr_input = lr_input_t[batch_idx]
                b_hr_color = hr_color_t[batch_idx]

                optimizer.zero_grad()

                pred_hr = self.model(b_lr_input)

                if pred_hr.shape != b_hr_color.shape:
                    pred_hr = F.interpolate(pred_hr, size=(h, w), mode='bicubic', align_corners=False)

                mse_loss = F.mse_loss(pred_hr, b_hr_color)
                ssim_val = ssim(pred_hr, b_hr_color)
                loss = mse_loss + (1.0 - ssim_val) * 0.1

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * len(batch_idx)
                epoch_mse += mse_loss.item() * len(batch_idx)
                epoch_ssim += ssim_val.item() * len(batch_idx)

            epoch_loss /= dataset_size
            epoch_mse /= dataset_size
            epoch_ssim /= dataset_size

            pbar.set_postfix({'Loss': f"{epoch_loss:.4f}", 'MSE': f"{epoch_mse:.4f}", 'SSIM': f"{epoch_ssim:.4f}"})

            if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == epochs - 1:
                log_msg = f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f} (MSE: {epoch_mse:.4f}, SSIM: {epoch_ssim:.4f})"
                logs.append(log_msg)

        # Save weights
        save_path = os.path.join(os.path.dirname(__file__), "espcn_weights.pth")
        torch.save(self.model.state_dict(), save_path)
        logs.append(f"Weights saved to {save_path}")

        self.model.eval()

        # Free memory
        del hr_color_t, lr_color_t, lr_depth_t, lr_input_t
        gc.collect()

        return "\\n".join(logs)

    @torch.no_grad()
    def upscale(self, color_np: np.ndarray, depth_np: np.ndarray) -> np.ndarray:
        """
        Upscales a low-resolution frame using the ESPCN model on CPU.

        Args:
            color_np (np.ndarray): Low-res color image (H, W, 3), float32 [0, 1].
            depth_np (np.ndarray): Low-res depth map (H, W), float32 [0, 1].

        Returns:
            np.ndarray: Upscaled HR RGB image (H*scale, W*scale, 3), float32 [0, 1].
        """
        depth_expanded = np.expand_dims(depth_np, axis=-1)
        fbo_combined = np.concatenate([color_np, depth_expanded], axis=-1)  # (H, W, 4)

        fbo_tensor = np.expand_dims(fbo_combined.transpose(2, 0, 1), axis=0)
        input_tensor = torch.from_numpy(fbo_tensor).contiguous().float()  # CPU tensor

        output_tensor = self.model(input_tensor)

        output_np = output_tensor.squeeze(0).permute(1, 2, 0).numpy()
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
