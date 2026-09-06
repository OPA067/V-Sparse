"""Image clip transformations for video frames.

Provides frame-list operations: resize, crop and normalize.
Input is a list of numpy.ndarray images in (H, W, C) channel-last format.
"""
from typing import List

import numpy as np
from PIL import Image


def resize_clip(clip: List[np.ndarray],
                target_size,
                interpolation: str = "bilinear") -> List[np.ndarray]:
    """Resize every frame in clip to target_size (height, width).

    Args:
        clip: list of (H, W, C) numpy images.
        target_size: (height, width) or int.
        interpolation: one of 'nearest', 'bilinear', 'bicubic', 'lanczos'.
    Returns:
        List of resized images as numpy arrays.
    """
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    pil_interp_map = {
        "nearest":  Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic":  Image.Resampling.BICUBIC,
        "lanczos":  Image.Resampling.LANCZOS,
    }
    pil_interp = pil_interp_map.get(interpolation, Image.Resampling.BILINEAR)

    resized = []
    for img in clip:
        is_numpy = isinstance(img, np.ndarray)
        pil_img = Image.fromarray(img) if is_numpy else img
        pil_img = pil_img.resize((target_size[1], target_size[0]),
                                 resample=pil_interp)
        resized.append(np.array(pil_img))
    return resized


def crop_clip(clip: List[np.ndarray],
              top: int,
              left: int,
              height: int,
              width: int) -> List[np.ndarray]:
    """Crop every frame to the given bounding-box region.

    Args:
        clip: list of (H, W, C) numpy images.
        top: top-left y.
        left: top-left x.
        height: crop height.
        width: crop width.
    Returns:
        List of cropped images as numpy arrays.
    """
    cropped = []
    for img in clip:
        is_numpy = isinstance(img, np.ndarray)
        pil_img = Image.fromarray(img) if is_numpy else img
        c_w, c_h = pil_img.size
        right = min(left + width, c_w)
        bottom = min(top + height, c_h)
        pil_img = pil_img.crop((left, top, right, bottom))
        cropped.append(np.array(pil_img))
    return cropped


def normalize(clip: List[np.ndarray],
              mean,
              std,
              dtype=None) -> np.ndarray:
    """Normalize clip in (T, C, H, W) format with mean and std.

    Expects the input ``clip`` to already be a numpy array or tensor
    in the ``torchvision`` standard layout (T, C, H, W).

    Args:
        clip: 4-D array/tensor (T, C, H, W).
        mean: sequence of length C.
        std: sequence of length C.
    Returns:
        Normalized array/tensor in the same type.
    """
    import torch

    if isinstance(clip, np.ndarray):
        mean_arr = np.array(mean, dtype=clip.dtype).reshape(-1, 1, 1)
        std_arr = np.array(std, dtype=clip.dtype).reshape(-1, 1, 1)
        return (clip - mean_arr) / std_arr
    elif isinstance(clip, torch.Tensor):
        mean_t = torch.as_tensor(mean, dtype=clip.dtype, device=clip.device)
        std_t = torch.as_tensor(std, dtype=clip.dtype, device=clip.device)
        return (clip - mean_t.view(-1, 1, 1)) / std_t.view(-1, 1, 1)
    else:
        raise TypeError("clip must be numpy.ndarray or torch.Tensor")
