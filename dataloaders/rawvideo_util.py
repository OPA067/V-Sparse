"""Video frame extraction utility using OpenCV (cv2).

RawVideoExtractorCV2 reads video files via cv2.VideoCapture, samples frames
at a configurable FPS, and applies image transforms (resize, center crop,
normalization). Training-time RandAugment is applied when subset == "train".

This is the default video backend used by RetrievalDataset and other
dataset loaders in this project.

Typical usage:
    extractor = RawVideoExtractor(framerate=1.0, size=224)
    result = extractor.get_video_data("path/to/video.mp4")
    video_tensor = result['video']  # shape: (T, C, H, W) or (1,) on error
"""

import torch as th
import numpy as np
from PIL import Image
# pytorch=1.7.1
# pip install opencv-python
import cv2
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode, \
    RandomHorizontalFlip, RandomResizedCrop
import dataloaders.video_transforms as video_transforms


class RawVideoExtractorCV2():
    """OpenCV-based video frame extractor with CLIP-style transforms.

    Extracts frames from MP4 videos using cv2.VideoCapture, samples them
    at the target framerate, and applies train/test image transforms.
    """

    def __init__(self, centercrop=False, size=224, framerate=-1, subset="test"):
        """Initialize the extractor.

        Args:
            centercrop: Whether to apply center crop (legacy, currently unused).
            size: Output spatial resolution (square).
            framerate: Target sampling FPS. -1 disables explicit sampling.
            subset: 'train' or 'test'; controls augmentation policy.
        """
        self.centercrop = centercrop
        self.size = size
        self.framerate = framerate
        self.transform = self._transform(self.size)
        self.subset = subset

        # Pre-defined transform pipelines for CLIP-style inference and training
        self.tsfm_dict = {
            'clip_test': Compose([
                Resize(size, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(size),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]),
            'clip_train': Compose([
                RandomResizedCrop(size, scale=(0.5, 1.0)),
                RandomHorizontalFlip(),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        }

        # Training-time random augmentation (RandAugment)
        self.aug_transform = video_transforms.create_random_augment(
            input_size=(size, size),
            auto_augment='rand-m7-n4-mstd0.5-inc1',
            interpolation='bicubic',
        )

    def _transform(self, n_px):
        """Build the default CLIP-style test transform.

        Args:
            n_px: Target spatial resolution (square).
        Returns:
            torchvision.transforms.Compose pipeline.
        """
        return Compose([
            Resize(n_px, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def video_to_tensor(self, video_file, preprocess, sample_fp=0, start_time=None, end_time=None, _no_process=False):
        """Read a video file and convert sampled frames to a tensor.

        Frame sampling logic:
            - Compute total duration from frame count and FPS.
            - If start_time / end_time are provided, clamp to valid range
              and seek to the starting frame.
            - Compute interval = FPS // sample_fp to control sampling density.
            - For each second in [start_sec, end_sec], read frames at the
              computed interval offsets.

        Args:
            video_file: Path to the video file (e.g., .mp4).
            preprocess: torchvision transform to apply to each frame.
            sample_fp: Target sampling FPS. 0 means use original FPS.
            start_time: Start time in seconds (optional).
            end_time: End time in seconds (optional).
            _no_process: If True, return raw PIL Images without transforms.
        Returns:
            Dict with key 'video':
                - Tensor of shape (T, C, H, W) on success.
                - torch.zeros(1) on failure (no frames extracted).
        """
        if start_time is not None or end_time is not None:
            assert isinstance(start_time, int) and isinstance(end_time, int) \
                   and start_time > -1 and end_time > start_time
        assert sample_fp > -1

        # Open video and query metadata
        cap = cv2.VideoCapture(video_file)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        if fps == 0:
            print((video_file + '\n') * 10)
        total_duration = (frameCount + fps - 1) // fps
        start_sec, end_sec = 0, total_duration

        if start_time is not None:
            start_sec, end_sec = start_time, end_time if end_time <= total_duration else total_duration
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * fps))

        # Compute sampling interval based on target FPS
        interval = 1
        if sample_fp > 0:
            interval = fps // sample_fp
        else:
            sample_fp = fps
        if interval == 0: interval = 1

        inds = [ind for ind in np.arange(0, fps, interval)]
        assert len(inds) >= sample_fp
        inds = inds[:sample_fp]

        ret = True
        images, included = [], []

        # Loop over each second and extract frames at the computed offsets
        for sec in np.arange(start_sec, end_sec + 1):
            if not ret: break
            sec_base = int(sec * fps)
            for ind in inds:
                cap.set(cv2.CAP_PROP_POS_FRAMES, sec_base + ind)
                ret, frame = cap.read()
                if not ret: break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if _no_process:
                    images.append(Image.fromarray(frame_rgb).convert("RGB"))
                else:
                    # Store raw PIL images; transforms applied later
                    images.append(Image.fromarray(frame_rgb))

        cap.release()

        if len(images) > 0:
            if _no_process:
                video_data = images
            else:
                # Apply RandAugment during training before normalization
                if self.subset == "train":
                    images = self.aug_transform(images)
                video_data = th.stack([preprocess(img) for img in images])
        else:
            video_data = th.zeros(1)
        return {'video': video_data}

    def get_video_data(self, video_path, start_time=None, end_time=None, _no_process=False):
        """Public entry point: extract video frames with default transform.

        Args:
            video_path: Path to the video file.
            start_time: Start time in seconds (optional).
            end_time: End time in seconds (optional).
            _no_process: If True, skip transform (return raw PIL Images).
        Returns:
            Dict with key 'video' containing the tensor or image list.
        """
        image_input = self.video_to_tensor(video_path, self.transform, sample_fp=self.framerate, start_time=start_time,
                                           end_time=end_time, _no_process=_no_process)
        return image_input

    def process_raw_data(self, raw_video_data):
        """Reshape raw video data to add a singleton channel dimension.

        Converts (L, T, C, H, W) -> (L*T, 1, C, H, W).

        Args:
            raw_video_data: Tensor of shape (L, T, C, H, W).
        Returns:
            Reshaped tensor of shape (L*T, 1, C, H, W).
        """
        tensor_size = raw_video_data.size()
        tensor = raw_video_data.view(-1, 1, tensor_size[-3], tensor_size[-2], tensor_size[-1])
        return tensor

    def process_frame_order(self, raw_video_data, frame_order=0):
        """Apply temporal ordering augmentation to video frames.

        Args:
            raw_video_data: Tensor of shape (T, ...).
            frame_order: Ordering strategy:
                0 = keep original order.
                1 = reverse order.
                2 = random shuffle.
        Returns:
            Reordered tensor.
        """
        if frame_order == 0:
            pass
        elif frame_order == 1:
            reverse_order = np.arange(raw_video_data.size(0) - 1, -1, -1)
            raw_video_data = raw_video_data[reverse_order, ...]
        elif frame_order == 2:
            random_order = np.arange(raw_video_data.size(0))
            np.random.shuffle(random_order)
            raw_video_data = raw_video_data[random_order, ...]

        return raw_video_data


# Default video frame extractor alias used across the project
RawVideoExtractor = RawVideoExtractorCV2
