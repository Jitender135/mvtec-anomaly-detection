"""
Visualization utilities for images, masks, and (later) anomaly heatmaps.

These functions are intentionally generic: the same overlay logic used
here to visualize ground-truth defect masks (Phase 2/10) will be reused
in Phase 5 to overlay predicted anomaly score maps on images. Building
this once, correctly, avoids duplicating overlay logic later.
"""

import numpy as np
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def denormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    """
    Reverse ImageNet normalization to get a displayable RGB image.

    Args:
        image_tensor: shape (3, H, W), normalized with IMAGENET_MEAN/STD.

    Returns:
        numpy array of shape (H, W, 3), float values clipped to [0, 1].
    """
    image = image_tensor.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))  # (C,H,W) -> (H,W,C)
    image = image * IMAGENET_STD + IMAGENET_MEAN  # undo normalization
    image = np.clip(image, 0.0, 1.0)
    return image


def mask_to_numpy(mask_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a (1, H, W) mask tensor to a plain (H, W) numpy array.
    """
    mask = mask_tensor.detach().cpu().numpy()
    return mask.squeeze(0)  # (1,H,W) -> (H,W)


def overlay_mask_on_image(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[float, float, float] = (1.0, 0.0, 0.0),
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Blend a binary (or continuous [0,1]) mask onto an RGB image as a
    translucent color overlay. Used for ground-truth defect masks now,
    and predicted anomaly score maps later -- same function, same logic.

    Args:
        image_rgb: (H, W, 3) float array in [0, 1].
        mask: (H, W) float array in [0, 1]. Values act as a per-pixel
              blend weight -- 0 means "show original image unchanged",
              1 means "show full overlay color".
        color: RGB color for the overlay, each channel in [0, 1].
        alpha: overall overlay strength multiplier.

    Returns:
        (H, W, 3) float array in [0, 1], the blended result.
    """
    mask = mask[..., None]  # (H,W) -> (H,W,1) for broadcasting against RGB
    color_array = np.array(color).reshape(1, 1, 3)

    blend_weight = mask * alpha
    overlaid = image_rgb * (1 - blend_weight) + color_array * blend_weight
    return np.clip(overlaid, 0.0, 1.0)