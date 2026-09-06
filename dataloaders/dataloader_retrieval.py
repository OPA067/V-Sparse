"""Base retrieval dataset class for video-text retrieval tasks.

Provides a generic RetrievalDataset that loads video-caption pairs, tokenizes
text, and extracts video frames. Two video decoding backends are supported:
    1. OpenCV-based (default, via RawVideoExtractor).
    2. decord-based (via VideoReader, used when GPU decoding is preferred).

The dataset supports three modes:
    'all'  - iterate over all (video, caption) pairs.
    'text' - iterate over captions only.
    'video'- iterate over videos only.

Also exports spatial_sampling and various video transform utilities.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from os.path import exists

import random
import numpy as np
from torch.utils.data import Dataset

import torch
from PIL import Image
from decord import VideoReader, cpu
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode, \
    RandomHorizontalFlip, RandomResizedCrop
import dataloaders.video_transforms as video_transforms


class RetrievalDataset(Dataset):
    """Base retrieval dataset for video-text pairs.

    Loads annotations via the subclass-implemented _get_anns(), builds
    video and sentence dictionaries, and provides text tokenization and
    frame extraction. Supports training-time data augmentation via
    RandAugment and spatial transforms.
    """

    def __init__(
            self,
            subset,
            anno_path,
            video_path,
            tokenizer,
            max_words=30,
            max_frames=12,
            video_framerate=1,
            image_resolution=224,
            mode='all',
            config=None
    ):
        """Initialize the retrieval dataset.

        Args:
            subset: 'train' or 'test'.
            anno_path: Root directory containing annotation files.
            video_path: Root directory containing MP4 video files.
            tokenizer: Text tokenizer (e.g., CLIPTokenizer).
            max_words: Maximum token count per text (including CLS/SEP).
            max_frames: Maximum number of frames extracted per video.
            video_framerate: Target sampling FPS.
            image_resolution: Spatial size of extracted frames.
            mode: Iteration mode - 'all' over pairs, 'text' over captions, 'video' over videos.
            config: Optional external config object.
        """
        self.subset = subset
        self.anno_path = anno_path
        self.video_path = video_path
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.max_frames = max_frames
        self.video_framerate = video_framerate
        self.image_resolution = image_resolution
        self.mode = mode
        self.config = config

        self.video_dict, self.sentences_dict = self._get_anns(self.subset)

        self.video_list = list(self.video_dict.keys())
        self.sample_len = 0

        print("Video number: {}".format(len(self.video_dict)))
        print("Total Pairs: {}".format(len(self.sentences_dict)))

        # ---- Video frame extractor (OpenCV backend) ----
        from .rawvideo_util import RawVideoExtractor
        self.rawVideoExtractor = RawVideoExtractor(framerate=video_framerate, size=image_resolution)

        # ---- Default CLIP-style normalization transform ----
        self.transform = Compose([
            Resize(image_resolution, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(image_resolution),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        # ---- Transform variants for clip-based training / testing ----
        self.tsfm_dict = {
            'clip_test': Compose([
                Resize(image_resolution, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(image_resolution),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]),
            'clip_train': Compose([
                RandomResizedCrop(image_resolution, scale=(0.5, 1.0)),
                RandomHorizontalFlip(),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        }

        # Special tokens used by the tokenizer
        # NOTE: UNK/PAD tokens appear to be placeholders
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": " ", "PAD_TOKEN": " "}  
        self.image_resolution = image_resolution

        # Set iteration length based on mode
        if self.mode in ['all', 'text']:
            self.sample_len = len(self.sentences_dict)
        else:
            self.sample_len = len(self.video_list)

        # ---- Training-time random augmentation (RandAugment) ----
        self.aug_transform = video_transforms.create_random_augment(
            input_size=(self.image_resolution, self.image_resolution),
            auto_augment='rand-m7-n4-mstd0.5-inc1',
            interpolation='bicubic',
        )

    def __len__(self):
        """Number of samples (depends on mode)."""
        return self.sample_len

    def _get_anns(self, subset='train'):
        """Load annotations. Must be implemented by subclasses.

        Returns:
            Tuple of (video_dict, sentences_dict).
        """
        raise NotImplementedError

    def _processing_caption(self, caption):
        """Tokenize a single caption and pad to max_words.

        Processing pipeline:
            1. tokenize(caption) -> word list.
            2. Prepend CLS, append SEP.
            3. Truncate if length exceeds max_words - 1.
            4. Pad to max_words with zeros.
            5. Build binary attention mask (1 for real tokens).

        Args:
            caption: Raw text string.
        Returns:
            input_ids: np.array of shape (max_words,) containing token IDs.
            input_mask: np.array of shape (max_words,) binary attention mask.
        """
        words = self.tokenizer.tokenize(caption)

        words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
        total_length_with_CLS = self.max_words - 1
        if len(words) > total_length_with_CLS:
            words = words[:total_length_with_CLS]
        words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

        input_ids = self.tokenizer.convert_tokens_to_ids(words)
        input_mask = [1] * len(input_ids)

        while len(input_ids) < self.max_words:
            input_ids.append(0)
            input_mask.append(0)
        assert len(input_ids) == self.max_words
        assert len(input_mask) == self.max_words

        input_ids = np.array(input_ids)
        input_mask = np.array(input_mask)

        return input_ids, input_mask

    def _get_text(self, caption_dict):
        """Process a caption tuple into tokenized tensors.

        A caption tuple has the form:
            (query, s, e)
        where:
            query: global caption text string.
            s    : temporal start (may be None).
            e    : temporal end (may be None).

        Args:
            caption_dict: Tuple as described above.
        Returns:
            query_ids : np.array shape (max_words,), global caption token IDs.
            query_mask: np.array shape (max_words,), global attention mask.
            s         : temporal start (may be None).
            e         : temporal end (may be None).
        """
        query, s, e = caption_dict

        # Tokenize the global aggregated caption
        query_ids, query_mask = self._processing_caption(query)

        return query_ids, query_mask, s, e

    def _get_rawvideo(self, video_id, s=None, e=None):
        """Extract video frames using the OpenCV backend.

        Args:
            video_id: Video identifier (key in self.video_dict).
            s: Optional start time in seconds (float). Converted to int internally.
            e: Optional end time in seconds (float). Converted to int internally.
        Returns:
            video      : np.ndarray shape (max_frames, 3, H, W).
            video_mask : np.ndarray shape (max_frames,), binary valid-frame mask.
        """
        video_mask = np.zeros(self.max_frames, dtype=np.int64)
        max_video_length = 0

        # Pre-allocate: T x 3 x H x W
        video = np.zeros((self.max_frames, 3, self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=float)

        if s is None:
            start_time, end_time = None, None
        else:
            start_time = int(s)
            end_time = int(e)
            start_time = start_time if start_time >= 0. else 0.
            end_time = end_time if end_time >= 0. else 0.
            if start_time > end_time:
                start_time, end_time = end_time, start_time
            elif start_time == end_time:
                end_time = end_time + 1
        video_path = self.video_dict[video_id]

        raw_video_data = self.rawVideoExtractor.get_video_data(video_path, start_time, end_time)
        raw_video_data = raw_video_data['video']

        if len(raw_video_data.shape) > 3:
            # L x T x 3 x H x W

            if self.max_frames < raw_video_data.shape[0]:
                sample_indx = np.linspace(0, raw_video_data.shape[0] - 1, num=self.max_frames, dtype=int)
                video_slice = raw_video_data[sample_indx, ...]
            else:
                video_slice = raw_video_data

            video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=0)

            slice_len = video_slice.shape[0]
            max_video_length = max_video_length if max_video_length > slice_len else slice_len
            if slice_len < 1:
                pass
            else:
                video[:slice_len, ...] = video_slice
        else:
            print("video path: {} error. video id: {}".format(video_path, video_id))

        video_mask[:max_video_length] = [1] * max_video_length

        return video, video_mask

    def _get_rawvideo_dec(self, video_id, s=None, e=None):
        """Extract video frames using the decord backend.

        This backend uses decord.VideoReader for decoding and applies
        training-time RandAugment before the CLIP-style normalization.

        Args:
            video_id: Video identifier (key in self.video_dict).
            s: Optional start time in seconds (float).
            e: Optional end time in seconds (float).
        Returns:
            video      : np.ndarray shape (max_frames, 3, H, W).
            video_mask : np.ndarray shape (max_frames,), binary valid-frame mask.
        """
        video_mask = np.zeros(self.max_frames, dtype=np.int64)
        max_video_length = 0

        video = np.zeros((self.max_frames, 3, self.image_resolution, self.image_resolution), dtype=float)

        if s is None:
            start_time, end_time = None, None
        else:
            start_time = int(s)
            end_time = int(e)
            start_time = start_time if start_time >= 0. else 0.
            end_time = end_time if end_time >= 0. else 0.
            if start_time > end_time:
                start_time, end_time = end_time, start_time
            elif start_time == end_time:
                end_time = start_time + 1
        video_path = self.video_dict[video_id]

        if exists(video_path):
            vreader = VideoReader(video_path, ctx=cpu(0))
        else:
            print(video_path)
            raise FileNotFoundError

        fps = vreader.get_avg_fps()
        f_start = 0 if start_time is None else int(start_time * fps)
        f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
        num_frames = f_end - f_start + 1
        if num_frames > 0:
            sample_fps = int(self.video_framerate)
            t_stride = int(round(float(fps) / sample_fps))

            all_pos = list(range(f_start, f_end + 1, t_stride))
            if len(all_pos) > self.max_frames:
                sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=self.max_frames, dtype=int)]
            else:
                sample_pos = all_pos

            patch_images = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]
            # Apply RandAugment during training
            if self.subset == "train":
                patch_images = self.aug_transform(patch_images)

            patch_images = torch.stack([self.transform(img) for img in patch_images])
            slice_len = patch_images.shape[0]
            max_video_length = max_video_length if max_video_length > slice_len else slice_len
            if slice_len < 1:
                pass
            else:
                video[:slice_len, ...] = patch_images
        else:
            print("video path: {} error. video id: {}".format(video_path, video_id))

        video_mask[:max_video_length] = [1] * max_video_length

        return video, video_mask

    def __getitem__(self, idx):
        """Return a single (text, video) sample.

        Uses the OpenCV backend for video decoding.
        Returns:
            Tuple of (text_ids, text_mask, video, video_mask, idx,
                      hash(video_id_with_prefix_removed)).
        """
        video_id, caption_dict = self.sentences_dict[idx]

        text_ids, text_mask, s, e = self._get_text(caption_dict)
        # video, video_mask = self._get_rawvideo_dec(video_id, s, e)
        video, video_mask = self._get_rawvideo(video_id, s, e)
        return text_ids, text_mask, video, video_mask, idx, hash(video_id.replace("video", ""))

    def get_text_len(self):
        """Number of text samples (captions)."""
        return len(self.sentences_dict)

    def get_video_len(self):
        """Number of video samples."""
        return len(self.video_list)

    def get_text_content(self, ind):
        """Get the raw caption tuple at index ind."""
        return self.sentences_dict[ind][1]

    def get_data_name(self):
        """Return a string identifier for this dataset instance."""
        return self.__class__.__name__ + "_" + self.subset

    def get_vis_info(self, idx):
        """Get caption and video path for a given index."""
        video_id, caption = self.sentences_dict[idx]
        video_path = self.video_dict[video_id]
        return caption, video_path

def spatial_sampling(
    frames,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
    aspect_ratio=None,
    scale=None,
    motion_shift=False,
):
    """Perform spatial sampling (crop + resize + flip) on video frames.

    Two modes:
        - spatial_idx == -1: random scale, random crop, random flip (training).
        - spatial_idx in {0, 1, 2}: deterministic uniform crop (testing).

    Args:
        frames: Tensor of shape (T, H, W, C) or (T, C, H, W).
        spatial_idx: -1 for random, 0/1/2 for left/center/right or top/center/bottom.
        min_scale: Minimal side length for random scale jitter.
        max_scale: Maximal side length for random scale jitter.
        crop_size: Final crop size (square).
        random_horizontal_flip: Whether to apply random horizontal flip.
        inverse_uniform_sampling: If True, sample scale uniformly in reciprocal space.
        aspect_ratio: Aspect ratio range for random resized crop.
        scale: Scale range for random resized crop.
        motion_shift: If True, use random_resized_crop_with_shift.
    Returns:
        Spatially sampled frames tensor.
    """
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        if aspect_ratio is None and scale is None:
            frames, _ = video_transforms.random_short_side_scale_jitter(
                images=frames,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=inverse_uniform_sampling,
            )
            frames, _ = video_transforms.random_crop(frames, crop_size)
        else:
            transform_func = (
                video_transforms.random_resized_crop_with_shift
                if motion_shift
                else video_transforms.random_resized_crop
            )
            frames = transform_func(
                images=frames,
                target_height=crop_size,
                target_width=crop_size,
                scale=scale,
                ratio=aspect_ratio,
            )
        if random_horizontal_flip:
            frames, _ = video_transforms.horizontal_flip(0.5, frames)
    else:
        # Deterministic testing path: no jitter, uniform crop
        assert len({min_scale, max_scale, crop_size}) == 1
        frames, _ = video_transforms.random_short_side_scale_jitter(
            frames, min_scale, max_scale
        )
        frames, _ = video_transforms.uniform_crop(frames, crop_size, spatial_idx)
    return frames
