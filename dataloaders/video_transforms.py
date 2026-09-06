#!/usr/bin/env python3
"""Video spatial transforms and augmentation utilities.

Provides PyTorch-compatible video transforms for spatial sampling, cropping,
flipping, color jitter, and RandAugment composition. These operate on tensors
of shape (T, C, H, W) or lists of PIL Images and are used by the video-text
retrieval datasets during training and evaluation.

Key components:
    - Spatial transforms: random_short_side_scale_jitter, random_crop,
      uniform_crop, horizontal_flip.
    - Color transforms: color_jitter, brightness_jitter, contrast_jitter,
      saturation_jitter, lighting_jitter, color_normalization.
    - RandAugment integration: create_random_augment for training data aug.
    - PIL-based transforms: Compose, RandomHorizontalFlip, Resize, CenterCrop,
      RandomCrop, RandomRotation, ColorJitter, etc.
"""

import math
import numpy as np
import random
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision import transforms

from .rand_augment import rand_augment_transform
from .random_erasing import RandomErasing


import numbers
import PIL
import torchvision

from . import functional as FF

_pil_interpolation_to_str = {
    Image.NEAREST: "PIL.Image.NEAREST",
    Image.BILINEAR: "PIL.Image.BILINEAR",
    Image.BICUBIC: "PIL.Image.BICUBIC",
    Image.LANCZOS: "PIL.Image.LANCZOS",
    Image.HAMMING: "PIL.Image.HAMMING",
    Image.BOX: "PIL.Image.BOX",
}


_RANDOM_INTERPOLATION = (Image.BILINEAR, Image.BICUBIC)


def _pil_interp(method):
    """Map a string interpolation method to PIL enum."""
    if method == "bicubic":
        return Image.BICUBIC
    elif method == "lanczos":
        return Image.LANCZOS
    elif method == "hamming":
        return Image.HAMMING
    else:
        return Image.BILINEAR


def random_short_side_scale_jitter(
    images, min_size, max_size, boxes=None, inverse_uniform_sampling=False
):
    """Randomly scale the short side of video frames to [min_size, max_size].

    Args:
        images: Tensor of shape (T, C, H, W).
        min_size: Minimal size to scale the short side to.
        max_size: Maximal size to scale the short side to.
        boxes: Optional bounding boxes of shape (N, 4).
        inverse_uniform_sampling: If True, sample uniformly in reciprocal space
            and take reciprocal to get the scale.
    Returns:
        scaled_images: Tensor of shape (T, C, new_H, new_W).
        scaled_boxes: Rescaled boxes or None.
    """
    if inverse_uniform_sampling:
        size = int(
            round(1.0 / np.random.uniform(1.0 / max_size, 1.0 / min_size))
        )
    else:
        size = int(round(np.random.uniform(min_size, max_size)))

    height = images.shape[2]
    width = images.shape[3]
    if (width <= height and width == size) or (
        height <= width and height == size
    ):
        return images, boxes
    new_width = size
    new_height = size
    if width < height:
        new_height = int(math.floor((float(height) / width) * size))
        if boxes is not None:
            boxes = boxes * float(new_height) / height
    else:
        new_width = int(math.floor((float(width) / height) * size))
        if boxes is not None:
            boxes = boxes * float(new_width) / width

    return (
        torch.nn.functional.interpolate(
            images,
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        ),
        boxes,
    )


def crop_boxes(boxes, x_offset, y_offset):
    """Shift bounding boxes by the given crop offsets.

    Args:
        boxes: ndarray of shape (N, 4) in [x1, y1, x2, y2] format.
        x_offset: Horizontal crop offset.
        y_offset: Vertical crop offset.
    Returns:
        Cropped boxes array.
    """
    cropped_boxes = boxes.copy()
    cropped_boxes[:, [0, 2]] = boxes[:, [0, 2]] - x_offset
    cropped_boxes[:, [1, 3]] = boxes[:, [1, 3]] - y_offset

    return cropped_boxes


def random_crop(images, size, boxes=None):
    """Randomly crop video frames to a square of the given size.

    Args:
        images: Tensor of shape (T, C, H, W).
        size: Output spatial size.
        boxes: Optional bounding boxes of shape (N, 4).
    Returns:
        cropped: Tensor of shape (T, C, size, size).
        cropped_boxes: Shifted boxes or None.
    """
    if images.shape[2] == size and images.shape[3] == size:
        return images
    height = images.shape[2]
    width = images.shape[3]
    y_offset = 0
    if height > size:
        y_offset = int(np.random.randint(0, height - size))
    x_offset = 0
    if width > size:
        x_offset = int(np.random.randint(0, width - size))
    cropped = images[
        :, :, y_offset : y_offset + size, x_offset : x_offset + size
    ]

    cropped_boxes = (
        crop_boxes(boxes, x_offset, y_offset) if boxes is not None else None
    )

    return cropped, cropped_boxes


def horizontal_flip(prob, images, boxes=None):
    """Randomly flip video frames horizontally with probability prob.

    Args:
        prob: Probability of flipping.
        images: Tensor of shape (T, C, H, W).
        boxes: Optional bounding boxes of shape (N, 4).
    Returns:
        Flipped images tensor and flipped boxes (or None).
    """
    if boxes is None:
        flipped_boxes = None
    else:
        flipped_boxes = boxes.copy()

    if np.random.uniform() < prob:
        images = images.flip((-1))

        if len(images.shape) == 3:
            width = images.shape[2]
        elif len(images.shape) == 4:
            width = images.shape[3]
        else:
            raise NotImplementedError("Dimension does not supported")
        if boxes is not None:
            flipped_boxes[:, [0, 2]] = width - boxes[:, [2, 0]] - 1

    return images, flipped_boxes


def uniform_crop(images, size, spatial_idx, boxes=None, scale_size=None):
    """Deterministically crop video frames for test-time evaluation.

    Three spatial positions (0, 1, 2) correspond to left/center/right
    when width > height, or top/center/bottom when height > width.

    Args:
        images: Tensor of shape (T, C, H, W).
        size: Output crop size.
        spatial_idx: 0, 1, or 2.
        boxes: Optional bounding boxes.
        scale_size: If provided, resize short side to scale_size before cropping.
    Returns:
        Cropped tensor and cropped boxes (or None).
    """
    assert spatial_idx in [0, 1, 2]
    ndim = len(images.shape)
    if ndim == 3:
        images = images.unsqueeze(0)
    height = images.shape[2]
    width = images.shape[3]

    if scale_size is not None:
        if width <= height:
            width, height = scale_size, int(height / width * scale_size)
        else:
            width, height = int(width / height * scale_size), scale_size
        images = torch.nn.functional.interpolate(
            images,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

    y_offset = int(math.ceil((height - size) / 2))
    x_offset = int(math.ceil((width - size) / 2))

    if height > width:
        if spatial_idx == 0:
            y_offset = 0
        elif spatial_idx == 2:
            y_offset = height - size
    else:
        if spatial_idx == 0:
            x_offset = 0
        elif spatial_idx == 2:
            x_offset = width - size
    cropped = images[
        :, :, y_offset : y_offset + size, x_offset : x_offset + size
    ]
    cropped_boxes = (
        crop_boxes(boxes, x_offset, y_offset) if boxes is not None else None
    )
    if ndim == 3:
        cropped = cropped.squeeze(0)
    return cropped, cropped_boxes


def clip_boxes_to_image(boxes, height, width):
    """Clip bounding boxes to lie within image boundaries.

    Args:
        boxes: ndarray of shape (N, 4).
        height: Image height.
        width: Image width.
    Returns:
        Clipped boxes array.
    """
    clipped_boxes = boxes.copy()
    clipped_boxes[:, [0, 2]] = np.minimum(
        width - 1.0, np.maximum(0.0, boxes[:, [0, 2]])
    )
    clipped_boxes[:, [1, 3]] = np.minimum(
        height - 1.0, np.maximum(0.0, boxes[:, [1, 3]])
    )
    return clipped_boxes


def blend(images1, images2, alpha):
    """Blend two image tensors with weight alpha.

    Args:
        images1, images2: Tensors of shape (T, C, H, W).
        alpha: Blending weight.
    Returns:
        Blended tensor.
    """
    return images1 * alpha + images2 * (1 - alpha)


def grayscale(images):
    """Convert RGB images to grayscale using standard luminance coefficients.

    Expects BGR channel order: R=0.299, G=0.587, B=0.114.

    Args:
        images: Tensor of shape (T, C, H, W).
    Returns:
        Grayscale tensor with all channels set to the luminance value.
    """
    # R -> 0.299, G -> 0.587, B -> 0.114.
    img_gray = torch.tensor(images)
    gray_channel = (
        0.299 * images[:, 2] + 0.587 * images[:, 1] + 0.114 * images[:, 0]
    )
    img_gray[:, 0] = gray_channel
    img_gray[:, 1] = gray_channel
    img_gray[:, 2] = gray_channel
    return img_gray


def color_jitter(images, img_brightness=0, img_contrast=0, img_saturation=0):
    """Apply random color jittering to video frames.

    Args:
        images: Tensor of shape (T, C, H, W) in BGR order.
        img_brightness: Jitter ratio for brightness.
        img_contrast: Jitter ratio for contrast.
        img_saturation: Jitter ratio for saturation.
    Returns:
        Jittered tensor.
    """

    jitter = []
    if img_brightness != 0:
        jitter.append("brightness")
    if img_contrast != 0:
        jitter.append("contrast")
    if img_saturation != 0:
        jitter.append("saturation")

    if len(jitter) > 0:
        order = np.random.permutation(np.arange(len(jitter)))
        for idx in range(0, len(jitter)):
            if jitter[order[idx]] == "brightness":
                images = brightness_jitter(img_brightness, images)
            elif jitter[order[idx]] == "contrast":
                images = contrast_jitter(img_contrast, images)
            elif jitter[order[idx]] == "saturation":
                images = saturation_jitter(img_saturation, images)
    return images


def brightness_jitter(var, images):
    """Apply brightness jittering.

    Args:
        var: Jitter ratio.
        images: Tensor of shape (T, C, H, W).
    Returns:
        Jittered tensor.
    """
    alpha = 1.0 + np.random.uniform(-var, var)

    img_bright = torch.zeros(images.shape)
    images = blend(images, img_bright, alpha)
    return images


def contrast_jitter(var, images):
    """Apply contrast jittering.

    Args:
        var: Jitter ratio.
        images: Tensor of shape (T, C, H, W).
    Returns:
        Jittered tensor.
    """
    alpha = 1.0 + np.random.uniform(-var, var)

    img_gray = grayscale(images)
    img_gray[:] = torch.mean(img_gray, dim=(1, 2, 3), keepdim=True)
    images = blend(images, img_gray, alpha)
    return images


def saturation_jitter(var, images):
    """Apply saturation jittering.

    Args:
        var: Jitter ratio.
        images: Tensor of shape (T, C, H, W).
    Returns:
        Jittered tensor.
    """
    alpha = 1.0 + np.random.uniform(-var, var)
    img_gray = grayscale(images)
    images = blend(images, img_gray, alpha)

    return images


def lighting_jitter(images, alphastd, eigval, eigvec):
    """Apply AlexNet-style PCA color jitter (lighting jitter).

    Args:
        images: Tensor of shape (T, C, H, W).
        alphastd: Standard deviation of the random alpha coefficients.
        eigval: Eigenvalues of the RGB covariance matrix (length 3).
        eigvec: Eigenvectors of the RGB covariance matrix (3x3).
    Returns:
        Jittered tensor.
    """
    if alphastd == 0:
        return images
    # generate alpha1, alpha2, alpha3.
    alpha = np.random.normal(0, alphastd, size=(1, 3))
    eig_vec = np.array(eigvec)
    eig_val = np.reshape(eigval, (1, 3))
    rgb = np.sum(
        eig_vec * np.repeat(alpha, 3, axis=0) * np.repeat(eig_val, 3, axis=0),
        axis=1,
    )
    out_images = torch.zeros_like(images)
    if len(images.shape) == 3:
        # C H W
        channel_dim = 0
    elif len(images.shape) == 4:
        # T C H W
        channel_dim = 1
    else:
        raise NotImplementedError(f"Unsupported dimension {len(images.shape)}")

    for idx in range(images.shape[channel_dim]):
        # C H W
        if len(images.shape) == 3:
            out_images[idx] = images[idx] + rgb[2 - idx]
        # T C H W
        elif len(images.shape) == 4:
            out_images[:, idx] = images[:, idx] + rgb[2 - idx]
        else:
            raise NotImplementedError(
                f"Unsupported dimension {len(images.shape)}"
            )

    return out_images


def color_normalization(images, mean, stddev):
    """Normalize images by per-channel mean and std deviation.

    Args:
        images: Tensor of shape (T, C, H, W) or (C, H, W).
        mean: Sequence of C mean values.
        stddev: Sequence of C std values.
    Returns:
        Normalized tensor of the same shape.
    """
    if len(images.shape) == 3:
        assert (
            len(mean) == images.shape[0]
        ), "channel mean not computed properly"
        assert (
            len(stddev) == images.shape[0]
        ), "channel stddev not computed properly"
    elif len(images.shape) == 4:
        assert (
            len(mean) == images.shape[1]
        ), "channel mean not computed properly"
        assert (
            len(stddev) == images.shape[1]
        ), "channel stddev not computed properly"
    else:
        raise NotImplementedError(f"Unsupported dimension {len(images.shape)}")

    out_images = torch.zeros_like(images)
    for idx in range(len(mean)):
        # C H W
        if len(images.shape) == 3:
            out_images[idx] = (images[idx] - mean[idx]) / stddev[idx]
        elif len(images.shape) == 4:
            out_images[:, idx] = (images[:, idx] - mean[idx]) / stddev[idx]
        else:
            raise NotImplementedError(
                f"Unsupported dimension {len(images.shape)}"
            )
    return out_images


def _get_param_spatial_crop(
    scale, ratio, height, width, num_repeat=10, log_scale=True, switch_hw=False
):
    """Sample a random spatial crop box given scale and aspect ratio constraints.

    Args:
        scale: Tuple of (min_area_ratio, max_area_ratio) relative to image area.
        ratio: Tuple of (min_aspect, max_aspect).
        height: Image height.
        width: Image width.
        num_repeat: Max attempts to find a valid crop.
        log_scale: If True, sample aspect ratio in log space.
        switch_hw: If True, randomly swap width and height.
    Returns:
        Tuple (i, j, h, w): top, left, crop height, crop width.
    """
    for _ in range(num_repeat):
        area = height * width
        target_area = random.uniform(*scale) * area
        if log_scale:
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))
        else:
            aspect_ratio = random.uniform(*ratio)

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if np.random.uniform() < 0.5 and switch_hw:
            w, h = h, w

        if 0 < w <= width and 0 < h <= height:
            i = random.randint(0, height - h)
            j = random.randint(0, width - w)
            return i, j, h, w

    # Fallback to central crop
    in_ratio = float(width) / float(height)
    if in_ratio < min(ratio):
        w = width
        h = int(round(w / min(ratio)))
    elif in_ratio > max(ratio):
        h = height
        w = int(round(h * max(ratio)))
    else:  # whole image
        w = width
        h = height
    i = (height - h) // 2
    j = (width - w) // 2
    return i, j, h, w


def random_resized_crop(
    images,
    target_height,
    target_width,
    scale=(0.8, 1.0),
    ratio=(3.0 / 4.0, 4.0 / 3.0),
):
    """Crop video frames to random size/aspect ratio, then resize to target.

    Inception-style random resized crop used for training.

    Args:
        images: Tensor of shape (T, C, H, W).
        target_height: Desired output height.
        target_width: Desired output width.
        scale: Area ratio range relative to original image area.
        ratio: Aspect ratio range.
    Returns:
        Resized tensor of shape (T, C, target_height, target_width).
    """

    height = images.shape[2]
    width = images.shape[3]

    i, j, h, w = _get_param_spatial_crop(scale, ratio, height, width)
    cropped = images[:, :, i : i + h, j : j + w]
    return torch.nn.functional.interpolate(
        cropped,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )


def random_resized_crop_with_shift(
    images,
    target_height,
    target_width,
    scale=(0.8, 1.0),
    ratio=(3.0 / 4.0, 4.0 / 3.0),
):
    """Random resized crop with motion shift (different crop per frame).

    Samples two crop boxes (one for the first frame, one for the last frame)
    and linearly interpolates the box for intermediate frames.

    Args:
        images: Tensor of shape (C, T, H, W).
        target_height: Desired output height.
        target_width: Desired output width.
        scale: Area ratio range.
        ratio: Aspect ratio range.
    Returns:
        Tensor of shape (C, T, target_height, target_width).
    """
    t = images.shape[1]
    height = images.shape[2]
    width = images.shape[3]

    i, j, h, w = _get_param_spatial_crop(scale, ratio, height, width)
    i_, j_, h_, w_ = _get_param_spatial_crop(scale, ratio, height, width)
    i_s = [int(i) for i in torch.linspace(i, i_, steps=t).tolist()]
    j_s = [int(i) for i in torch.linspace(j, j_, steps=t).tolist()]
    h_s = [int(i) for i in torch.linspace(h, h_, steps=t).tolist()]
    w_s = [int(i) for i in torch.linspace(w, w_, steps=t).tolist()]
    out = torch.zeros((3, t, target_height, target_width))
    for ind in range(t):
        out[:, ind : ind + 1, :, :] = torch.nn.functional.interpolate(
            images[
                :,
                ind : ind + 1,
                i_s[ind] : i_s[ind] + h_s[ind],
                j_s[ind] : j_s[ind] + w_s[ind],
            ],
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )
    return out


def create_random_augment(
    input_size,
    auto_augment=None,
    interpolation="bilinear",
):
    """Build a RandAugment transform for video frames.

    Args:
        input_size: Target spatial size (int or tuple).
        auto_augment: RandAugment config string, e.g. "rand-m7-n4-mstd0.5-inc1".
        interpolation: Interpolation method string.
    Returns:
        torchvision.transforms.Compose containing RandAugment.
    Raises:
        NotImplementedError: If auto_augment is not a valid rand* string.
    """
    if isinstance(input_size, tuple):
        img_size = input_size[-2:]
    else:
        img_size = input_size

    if auto_augment:
        assert isinstance(auto_augment, str)
        if isinstance(img_size, tuple):
            img_size_min = min(img_size)
        else:
            img_size_min = img_size
        aa_params = {"translate_const": int(img_size_min * 0.45)}
        if interpolation and interpolation != "random":
            aa_params["interpolation"] = _pil_interp(interpolation)
        if auto_augment.startswith("rand"):
            return transforms.Compose(
                [rand_augment_transform(auto_augment, aa_params)]
            )
    raise NotImplementedError


def random_sized_crop_img(
    im,
    size,
    jitter_scale=(0.08, 1.0),
    jitter_aspect=(3.0 / 4.0, 4.0 / 3.0),
    max_iter=10,
):
    """Inception-style random resized crop for a single image.

    Args:
        im: Tensor of shape (C, H, W).
        size: Output spatial size.
        jitter_scale: Scale range relative to original area.
        jitter_aspect: Aspect ratio range.
        max_iter: Max attempts to find a valid crop.
    Returns:
        Cropped and resized tensor of shape (C, size, size).
    """
    assert (
        len(im.shape) == 3
    ), "Currently only support image for random_sized_crop"
    h, w = im.shape[1:3]
    i, j, h, w = _get_param_spatial_crop(
        scale=jitter_scale,
        ratio=jitter_aspect,
        height=h,
        width=w,
        num_repeat=max_iter,
        log_scale=False,
        switch_hw=True,
    )
    cropped = im[:, i : i + h, j : j + w]
    return torch.nn.functional.interpolate(
        cropped.unsqueeze(0),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


# The following code are modified based on timm lib, we will replace the following
# contents with dependency from PyTorchVideo.
# https://github.com/facebookresearch/pytorchvideo
class RandomResizedCropAndInterpolation:
    """Random crop and resize with random interpolation (timm-style).

    A crop of random size (default: 0.08 to 1.0 of original size) and random
    aspect ratio (default: 3/4 to 4/3) is made, then resized to the target size.
    """

    def __init__(
        self,
        size,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation="bilinear",
    ):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            print("range should be of kind (min, max)")

        if interpolation == "random":
            self.interpolation = _RANDOM_INTERPOLATION
        else:
            self.interpolation = _pil_interp(interpolation)
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for random sized crop on a PIL Image.

        Args:
            img: PIL Image.
            scale: Tuple of (min, max) area ratio.
            ratio: Tuple of (min, max) aspect ratio.
        Returns:
            Tuple (i, j, h, w) for crop.
        """
        area = img.size[0] * img.size[1]

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.size[0] / img.size[1]
        if in_ratio < min(ratio):
            w = img.size[0]
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = img.size[1]
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = img.size[0]
            h = img.size[1]
        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2
        return i, j, h, w

    def __call__(self, img):
        """Apply random resized crop to a PIL Image.

        Args:
            img: PIL Image.
        Returns:
            Randomly cropped and resized PIL Image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        return F.resized_crop(img, i, j, h, w, self.size, interpolation)

    def __repr__(self):
        if isinstance(self.interpolation, (tuple, list)):
            interpolate_str = " ".join(
                [_pil_interpolation_to_str[x] for x in self.interpolation]
            )
        else:
            interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + "(size={0}".format(self.size)
        format_string += ", scale={0}".format(
            tuple(round(s, 4) for s in self.scale)
        )
        format_string += ", ratio={0}".format(
            tuple(round(r, 4) for r in self.ratio)
        )
        format_string += ", interpolation={0})".format(interpolate_str)
        return format_string


def transforms_imagenet_train(
    img_size=224,
    scale=None,
    ratio=None,
    hflip=0.5,
    vflip=0.0,
    color_jitter=0.4,
    auto_augment=None,
    interpolation="random",
    use_prefetcher=False,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    re_prob=0.0,
    re_mode="const",
    re_count=1,
    re_num_splits=0,
    separate=False,
):
    """Build the ImageNet training transform pipeline.

    Args:
        img_size: Target image size.
        scale: Random crop scale range.
        ratio: Random crop aspect ratio range.
        hflip: Horizontal flip probability.
        vflip: Vertical flip probability.
        color_jitter: Color jitter magnitude.
        auto_augment: RandAugment config string (e.g., "rand-m9-n3-mstd0.5").
        interpolation: Interpolation mode.
        use_prefetcher: If True, return tensor without ToTensor/Normalize.
        mean: Normalization mean.
        std: Normalization std.
        re_prob: Random erasing probability.
        re_mode: Random erasing mode ('const', 'rand', 'pixel').
        re_count: Max number of erasing regions.
        re_num_splits: Number of batch splits for clean/augmented samples.
        separate: If True, return 3 separate Compose transforms.
    Returns:
        Compose, or tuple of 3 Compose objects if separate=True.
    """
    if isinstance(img_size, tuple):
        img_size = img_size[-2:]
    else:
        img_size = img_size

    scale = tuple(scale or (0.08, 1.0))  # default imagenet scale range
    ratio = tuple(
        ratio or (3.0 / 4.0, 4.0 / 3.0)
    )  # default imagenet ratio range
    primary_tfl = [
        RandomResizedCropAndInterpolation(
            img_size, scale=scale, ratio=ratio, interpolation=interpolation
        )
    ]
    if hflip > 0.0:
        primary_tfl += [transforms.RandomHorizontalFlip(p=hflip)]
    if vflip > 0.0:
        primary_tfl += [transforms.RandomVerticalFlip(p=vflip)]

    secondary_tfl = []
    if auto_augment:
        assert isinstance(auto_augment, str)
        if isinstance(img_size, tuple):
            img_size_min = min(img_size)
        else:
            img_size_min = img_size
        aa_params = dict(
            translate_const=int(img_size_min * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
        if interpolation and interpolation != "random":
            aa_params["interpolation"] = _pil_interp(interpolation)
        if auto_augment.startswith("rand"):
            secondary_tfl += [rand_augment_transform(auto_augment, aa_params)]
        elif auto_augment.startswith("augmix"):
            raise NotImplementedError("Augmix not implemented")
        else:
            raise NotImplementedError("Auto aug not implemented")
    elif color_jitter is not None:
        # color jitter is enabled when not using AA
        if isinstance(color_jitter, (list, tuple)):
            # color jitter should be a 3-tuple/list if spec brightness/contrast/saturation
            # or 4 if also augmenting hue
            assert len(color_jitter) in (3, 4)
        else:
            # if it's a scalar, duplicate for brightness, contrast, and saturation, no hue
            color_jitter = (float(color_jitter),) * 3
        secondary_tfl += [transforms.ColorJitter(*color_jitter)]

    final_tfl = []
    final_tfl += [
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
    ]
    if re_prob > 0.0:
        final_tfl.append(
            RandomErasing(
                re_prob,
                mode=re_mode,
                max_count=re_count,
                num_splits=re_num_splits,
                device="cpu",
                cube=False,
            )
        )

    if separate:
        return (
            transforms.Compose(primary_tfl),
            transforms.Compose(secondary_tfl),
            transforms.Compose(final_tfl),
        )
    else:
        return transforms.Compose(primary_tfl + secondary_tfl + final_tfl)

############################################################################################################
############################################################################################################

class Compose(object):
    """Compose multiple transforms into a single callable.

    Args:
        transforms: List of Transform objects.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, clip):
        for t in self.transforms:
            clip = t(clip)
        return clip


class RandomHorizontalFlip(object):
    """Randomly flip a list of images horizontally with probability 0.5."""

    def __call__(self, clip):
        """
        Args:
            clip: List of PIL.Image or numpy.ndarray in (h, w, c) format.
        Returns:
            Flipped clip (same type as input).
        """
        if random.random() < 0.5:
            if isinstance(clip[0], np.ndarray):
                return [np.fliplr(img) for img in clip]
            elif isinstance(clip[0], PIL.Image.Image):
                return [
                    img.transpose(PIL.Image.FLIP_LEFT_RIGHT) for img in clip
                ]
            else:
                raise TypeError('Expected numpy.ndarray or PIL.Image' +
                                ' but got list of {0}'.format(type(clip[0])))
        return clip


class RandomResize(object):
    """Randomly resize a list of numpy images by a random scaling factor.

    Args:
        ratio: Tuple of (min_scale, max_scale).
        interpolation: Interpolation mode string.
    """

    def __init__(self, ratio=(3. / 4., 4. / 3.), interpolation='nearest'):
        self.ratio = ratio
        self.interpolation = interpolation

    def __call__(self, clip):
        scaling_factor = random.uniform(self.ratio[0], self.ratio[1])

        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size

        new_w = int(im_w * scaling_factor)
        new_h = int(im_h * scaling_factor)
        new_size = (new_w, new_h)
        resized = FF.resize_clip(
            clip, new_size, interpolation=self.interpolation)
        return resized


class Resize(object):
    """Resize a list of images to a fixed size.

    Args:
        size: Target (width, height) or int.
        interpolation: Interpolation mode string.
    """

    def __init__(self, size, interpolation='nearest'):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, clip):
        resized = FF.resize_clip(
            clip, self.size, interpolation=self.interpolation)
        return resized


class RandomCrop(object):
    """Randomly crop a list of images at the same spatial location.

    Args:
        size: Desired output size (h, w) or int.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            size = (size, size)

        self.size = size

    def __call__(self, clip):
        """
        Args:
            clip: List of PIL.Image or numpy.ndarray in (h, w, c) format.
        Returns:
            Cropped list of images.
        """
        h, w = self.size
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        if w > im_w or h > im_h:
            error_msg = (
                'Initial image size should be larger then '
                'cropped size but got cropped sizes : ({w}, {h}) while '
                'initial image is ({im_w}, {im_h})'.format(
                    im_w=im_w, im_h=im_h, w=w, h=h))
            raise ValueError(error_msg)

        x1 = random.randint(0, im_w - w)
        y1 = random.randint(0, im_h - h)
        cropped = FF.crop_clip(clip, y1, x1, h, w)

        return cropped


class ThreeCrop(object):
    """Extract 3 evenly spaced crops from a list of images.

    Args:
        size: Desired output size (h, w) or int.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            size = (size, size)

        self.size = size

    def __call__(self, clip):
        """
        Args:
            clip: List of PIL.Image or numpy.ndarray in (h, w, c) format.
        Returns:
            List of 3 cropped images.
        """
        h, w = self.size
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        if w != im_w and h != im_h:
            clip = FF.resize_clip(clip, self.size, interpolation="bilinear")
            im_h, im_w, im_c = clip[0].shape

        step = np.max((np.max((im_w, im_h)) - self.size[0]) // 2, 0)
        cropped = []
        for i in range(3):
            if (im_h > self.size[0]):
                x1 = 0
                y1 = i * step
                cropped.extend(FF.crop_clip(clip, y1, x1, h, w))
            else:
                x1 = i * step
                y1 = 0
                cropped.extend(FF.crop_clip(clip, y1, x1, h, w))
        return cropped


class RandomRotation(object):
    """Rotate a list of images by a random angle in the given range.

    Args:
        degrees: If a single number, range is (-degrees, +degrees).
            If a tuple, it is (min, max).
    """

    def __init__(self, degrees):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError('If degrees is a single number,'
                                 'must be positive')
            degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError('If degrees is a sequence,'
                                 'it must be of len 2.')

        self.degrees = degrees

    def __call__(self, clip):
        """
        Args:
            clip: List of PIL.Image or numpy.ndarray in (h, w, c) format.
        Returns:
            Rotated list of images.
        """
        import skimage
        angle = random.uniform(self.degrees[0], self.degrees[1])
        if isinstance(clip[0], np.ndarray):
            rotated = [skimage.transform.rotate(img, angle) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            rotated = [img.rotate(angle) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))

        return rotated


class CenterCrop(object):
    """Center crop a list of images to the given size.

    Args:
        size: Desired output size (h, w) or int.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            size = (size, size)

        self.size = size

    def __call__(self, clip):
        """
        Args:
            clip: List of PIL.Image or numpy.ndarray in (h, w, c) format.
        Returns:
            Center cropped list of images.
        """
        h, w = self.size
        if isinstance(clip[0], np.ndarray):
            im_h, im_w, im_c = clip[0].shape
        elif isinstance(clip[0], PIL.Image.Image):
            im_w, im_h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        if w > im_w or h > im_h:
            error_msg = (
                'Initial image size should be larger then '
                'cropped size but got cropped sizes : ({w}, {h}) while '
                'initial image is ({im_w}, {im_h})'.format(
                    im_w=im_w, im_h=im_h, w=w, h=h))
            raise ValueError(error_msg)

        x1 = int(round((im_w - w) / 2.))
        y1 = int(round((im_h - h) / 2.))
        cropped = FF.crop_clip(clip, y1, x1, h, w)

        return cropped


class ColorJitter(object):
    """Randomly change brightness, contrast, saturation, and hue of images.

    Args:
        brightness: Jitter magnitude for brightness.
        contrast: Jitter magnitude for contrast.
        saturation: Jitter magnitude for saturation.
        hue: Jitter magnitude for hue.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def get_params(self, brightness, contrast, saturation, hue):
        """Sample random jitter factors."""
        if brightness > 0:
            brightness_factor = random.uniform(
                max(0, 1 - brightness), 1 + brightness)
        else:
            brightness_factor = None

        if contrast > 0:
            contrast_factor = random.uniform(
                max(0, 1 - contrast), 1 + contrast)
        else:
            contrast_factor = None

        if saturation > 0:
            saturation_factor = random.uniform(
                max(0, 1 - saturation), 1 + saturation)
        else:
            saturation_factor = None

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
        else:
            hue_factor = None
        return brightness_factor, contrast_factor, saturation_factor, hue_factor

    def __call__(self, clip):
        """
        Args:
            clip: List of PIL.Image.
        Returns:
            Jittered list of PIL.Image.
        """
        if isinstance(clip[0], np.ndarray):
            raise TypeError(
                'Color jitter not yet implemented for numpy arrays')
        elif isinstance(clip[0], PIL.Image.Image):
            brightness, contrast, saturation, hue = self.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)

            # Build transform function sequence
            img_transforms = []
            if brightness is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
            if saturation is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
            if hue is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
            if contrast is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
            random.shuffle(img_transforms)

            # Apply to all images
            jittered_clip = []
            for img in clip:
                for func in img_transforms:
                    jittered_img = func(img)
                jittered_clip.append(jittered_img)

        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        return jittered_clip


class Normalize(object):
    """Normalize a clip with per-channel mean and std.

    Args:
        mean: Sequence of means.
        std: Sequence of standard deviations.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, clip):
        """
        Args:
            clip: Tensor clip of size (T, C, H, W).
        Returns:
            Normalized tensor clip.
        """
        return FF.normalize(clip, self.mean, self.std)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
