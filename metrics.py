"""
metrics.py — Image Quality Metrics for DIY-DLSS
================================================
Utility functions to compare upscaled results against ground truth
(high-resolution direct rendering).

Usage:
    from metrics import compute_metrics, save_comparison

    results = compute_metrics(upscaled_img, ground_truth_img)
    print(f"PSNR: {results['psnr']:.2f} dB, SSIM: {results['ssim']:.4f}")
"""

import numpy as np


def psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 1.0) -> float:
    """
    Peak Signal-to-Noise Ratio.
    Higher is better. Typical range: 20-40 dB for upscaling.

    Args:
        img1, img2: (H, W, 3) float32 in [0, 1]
        max_val: maximum pixel value

    Returns:
        PSNR in dB
    """
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(max_val ** 2 / mse)


def ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Structural Similarity Index (simplified single-scale).
    Higher is better. Range: [-1, 1], typically 0.8-1.0 for good results.

    Uses scikit-image if available, otherwise falls back to a simple
    implementation.
    """
    try:
        from skimage.metrics import structural_similarity
        return structural_similarity(img1, img2, channel_axis=2, data_range=1.0)
    except ImportError:
        # Simplified SSIM (mean over channels)
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        results = []
        for c in range(img1.shape[2]):
            a = img1[:, :, c].astype(np.float64)
            b = img2[:, :, c].astype(np.float64)
            mu_a, mu_b = a.mean(), b.mean()
            sig_a, sig_b = a.var(), b.var()
            sig_ab = ((a - mu_a) * (b - mu_b)).mean()
            num = (2 * mu_a * mu_b + C1) * (2 * sig_ab + C2)
            den = (mu_a ** 2 + mu_b ** 2 + C1) * (sig_a + sig_b + C2)
            results.append(num / den)
        return np.mean(results)


def compute_metrics(upscaled: np.ndarray, ground_truth: np.ndarray) -> dict:
    """
    Compute all quality metrics between upscaled result and ground truth.

    Both inputs should be (H, W, 3) float32 in [0, 1].
    They must have the same dimensions.
    """
    assert upscaled.shape == ground_truth.shape, (
        f"Shape mismatch: {upscaled.shape} vs {ground_truth.shape}"
    )
    return {
        "psnr": psnr(upscaled, ground_truth),
        "ssim": ssim(upscaled, ground_truth),
    }


def save_comparison(
    low_res: np.ndarray,
    espcn: np.ndarray,
    bilinear: np.ndarray,
    ground_truth: np.ndarray,
):
    """
    Create a 2x2 comparison grid image with PSNR/SSIM text overlays.

    Layout:
    ┌──────────────┬──────────────┐
    │   Low-Res    │    ESPCN     │
    ├──────────────┼──────────────┤
    │   Bilinear   │ Ground Truth │
    └──────────────┴──────────────┘

    Each cell shows its PSNR/SSIM relative to Ground Truth.

    Returns:
        PIL.Image.Image: The comparison grid as a PIL image.
    """
    from PIL import Image, ImageDraw, ImageFont

    h, w = ground_truth.shape[:2]

    def to_pil(arr):
        return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))

    # Resize low_res to match ground truth dimensions for display
    low_pil = to_pil(low_res).resize((w, h), Image.NEAREST)

    # Compute metrics for each method vs ground truth
    # Resize low_res with nearest for metrics calculation too
    low_res_resized = np.array(low_pil).astype(np.float32) / 255.0
    metrics_lr = compute_metrics(low_res_resized, ground_truth)
    metrics_bilinear = compute_metrics(bilinear, ground_truth)
    metrics_espcn = compute_metrics(espcn, ground_truth)

    # Build 2x2 canvas
    pad = 4  # padding between cells
    canvas_w = w * 2 + pad
    canvas_h = h * 2 + pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), (30, 30, 30))

    canvas.paste(low_pil, (0, 0))
    canvas.paste(to_pil(espcn), (w + pad, 0))
    canvas.paste(to_pil(bilinear), (0, h + pad))
    canvas.paste(to_pil(ground_truth), (w + pad, h + pad))

    draw = ImageDraw.Draw(canvas)

    # Try to load a readable font, fall back to default
    try:
        font = ImageFont.truetype("arial.ttf", max(14, h // 18))
        font_small = ImageFont.truetype("arial.ttf", max(11, h // 24))
    except (IOError, OSError):
        font = ImageFont.load_default()
        font_small = font

    # Label data: (x_offset, y_offset, title, metrics_dict or None)
    labels = [
        (0, 0, "Low-Res", metrics_lr),
        (w + pad, 0, "ESPCN", metrics_espcn),
        (0, h + pad, "Bilinear", metrics_bilinear),
        (w + pad, h + pad, "Ground Truth", None),
    ]

    # Get font height safely
    try:
        font_h = font.size
    except AttributeError:
        font_h = 12  # default font height

    for x_off, y_off, title, m in labels:
        # Dark semi-transparent banner background (efficient method)
        banner_h = max(36, h // 7)
        overlay = Image.new("RGB", (w, banner_h), (0, 0, 0))
        # Blend: crop region, blend with dark overlay
        region = canvas.crop((x_off, y_off, x_off + w, y_off + banner_h))
        blended = Image.blend(region, overlay, alpha=0.65)
        canvas.paste(blended, (x_off, y_off))

        # Title text
        draw = ImageDraw.Draw(canvas)  # refresh draw after paste
        draw.text((x_off + 6, y_off + 4), title, fill=(255, 255, 255), font=font)

        # Metrics text
        if m is not None:
            psnr_val = m['psnr']
            ssim_val = m['ssim']
            psnr_str = f"PSNR: {psnr_val:.2f} dB" if psnr_val != float('inf') else "PSNR: INF"
            ssim_str = f"SSIM: {ssim_val:.4f}"
            metrics_text = f"{psnr_str}  |  {ssim_str}"
            draw.text((x_off + 6, y_off + 4 + font_h + 2), metrics_text,
                      fill=(180, 255, 180), font=font_small)

    return canvas
