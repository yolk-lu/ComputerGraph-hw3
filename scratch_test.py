"""
Diagnostic: save rendered LR, bilinear, ESPCN outputs as files for visual inspection.
"""
import sys, os
sys.path.append(os.path.dirname(__file__))

import numpy as np
import PIL.Image

# Check what OpenGL renders vs what ESPCN produces
from pipeline.model.dlss_model import AIUpscaler

# Load a sample rendered LR image for testing
# We'll create a synthetic test image to verify the ESPCN pipeline
print("=== ESPCN Color Pipeline Diagnostic ===")

upscaler = AIUpscaler(scale_factor=2, device='cpu')
weight_path = os.path.join('pipeline', 'model', 'espcn_weights.pth')

# Create a simple test gradient image (LR 256x256)
lr_h, lr_w = 256, 256
# Red-to-blue horizontal gradient
color_lr = np.zeros((lr_h, lr_w, 3), dtype=np.float32)
for x in range(lr_w):
    color_lr[:, x, 0] = x / lr_w  # R
    color_lr[:, x, 2] = 1.0 - x / lr_w  # B
color_lr[:, :, 1] = 0.3  # constant G

depth_lr = np.random.rand(lr_h, lr_w).astype(np.float32) * 0.5

# Bilinear
bilinear_np = upscaler.upscale_bilinear(color_lr)
print(f"Bilinear shape: {bilinear_np.shape}, range: [{bilinear_np.min():.3f}, {bilinear_np.max():.3f}]")

# ESPCN (random weights since no training on this)
espcn_np = upscaler.upscale(color_lr, depth_lr)
print(f"ESPCN (random) shape: {espcn_np.shape}, range: [{espcn_np.min():.3f}, {espcn_np.max():.3f}]")

# Load trained weights
if os.path.exists(weight_path):
    upscaler.load_weights(weight_path)
    espcn_trained_np = upscaler.upscale(color_lr, depth_lr)
    print(f"ESPCN (trained) shape: {espcn_trained_np.shape}, range: [{espcn_trained_np.min():.3f}, {espcn_trained_np.max():.3f}]")
    
    # Check channel order
    print(f"\nPixel [0,0] LR:       R={color_lr[0,0,0]:.3f} G={color_lr[0,0,1]:.3f} B={color_lr[0,0,2]:.3f}")
    print(f"Pixel [0,0] Bilinear: R={bilinear_np[0,0,0]:.3f} G={bilinear_np[0,0,1]:.3f} B={bilinear_np[0,0,2]:.3f}")
    print(f"Pixel [0,0] ESPCN:    R={espcn_trained_np[0,0,0]:.3f} G={espcn_trained_np[0,0,1]:.3f} B={espcn_trained_np[0,0,2]:.3f}")
    
    # Save images
    PIL.Image.fromarray((color_lr * 255).clip(0,255).astype(np.uint8)).save('debug_lr.png')
    PIL.Image.fromarray((bilinear_np * 255).clip(0,255).astype(np.uint8)).save('debug_bilinear.png')
    PIL.Image.fromarray((espcn_trained_np * 255).clip(0,255).astype(np.uint8)).save('debug_espcn.png')
    
    # Mean color difference
    diff = np.abs(espcn_trained_np - bilinear_np)
    print(f"\nMean abs diff (ESPCN vs Bilinear): {diff.mean():.6f}")
    print(f"Per-channel diff: R={diff[:,:,0].mean():.6f} G={diff[:,:,1].mean():.6f} B={diff[:,:,2].mean():.6f}")
else:
    print("No trained weights found.")

print("\nDone. Check debug_lr.png, debug_bilinear.png, debug_espcn.png")
