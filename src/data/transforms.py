"""
Preprocessing pipelines for MVTec AD images and masks.

Two SEPARATE pipelines are required:
  - Image transform: resize + convert to tensor + ImageNet normalization.
    Normalization MUST match ImageNet stats because our feature extractors
    (ResNet/WideResNet) are pretrained on ImageNet and expect that exact
    input distribution.
  - Mask transform: resize + convert to tensor ONLY. No normalization,
    since masks are label data (0 = normal, 1 = defect), not images.
    Uses nearest-neighbor interpolation to avoid blurring the binary mask.
"""

import torchvision.transforms as T

# Standard ImageNet normalization statistics.
# These are NOT computed from MVTec -- they match what the pretrained
# backbone was originally trained on, which is what matters here.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_image_transform(image_size: int) -> T.Compose:
    """
    Build the preprocessing pipeline applied to RGB input images.

    Args:
        image_size: target square size, e.g. 256 -> images become 256x256.
    """
    return T.Compose([
        T.Resize((image_size, image_size)),  # default bilinear interpolation
        T.ToTensor(),  # PIL [0,255] -> torch float [0,1], shape (C, H, W)
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_mask_transform(image_size: int) -> T.Compose:
    """
    Build the preprocessing pipeline applied to ground-truth masks.

    Uses NEAREST interpolation (not bilinear) to keep mask values exactly
    binary after resizing, and skips normalization entirely since mask
    values are labels (0 or 1), not pixel intensities.
    """
    return T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST),
        T.ToTensor(),  # PIL [0,255] -> torch float [0,1], shape (1, H, W)
    ])