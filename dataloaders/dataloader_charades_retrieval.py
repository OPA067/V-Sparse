"""Charades dataset loader for video-text retrieval.

Charades is a dataset of indoor daily-activity videos. Each video is paired
with a single global textual description.

Two dataset classes are provided:
    Charades_DataLoader      - training split
    Charades_TestDataLoader  - test split

Sample return format (training / test):
    (input_ids, input_mask,   # global caption token IDs / attention mask
     video, video_mask,       # video frames (N, L, 1, 3, H, W) / frame mask
     idx, hash(idx))

Expected directory layout:
    anno_path/
        Charades_v1_train.csv   - video IDs and global descriptions (training)
        Charades_v1_test.csv    - video IDs and global descriptions (test)
    video_path/
        {video_id}.mp4
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import csv
import os
from torch.utils.data import Dataset
import numpy as np

from dataloaders.rawvideo_util import RawVideoExtractor


class Charades_DataLoader(Dataset):
    """Charades training dataset loader.

    Loads per-video global descriptions from the train CSV.
    """

    def __init__(
            self,
            subset,
            anno_path,
            video_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
    ):
        """Initialize the Charades training loader.

        Args:
            subset: Dataset split, "train" or "test" (only "train" is used).
            anno_path: Directory containing the CSV annotations.
            video_path: Root directory of MP4 video files.
            tokenizer: Text tokenizer (e.g., CLIPTokenizer).
            max_words: Maximum token count per text sample (including CLS/SEP tokens); extra tokens are truncated.
            feature_framerate: Sampling frame rate (FPS). Default 1.0 means one frame per second.
            max_frames: Maximum number of frames retained per video; exceeding clips are truncated.
            image_resolution: Spatial resolution of extracted frames (default 224x224).
            frame_order: Temporal ordering strategy: 0=original / 1=reverse / 2=random.
            slice_framepos: Frame position sampling strategy:
                0 = keep the first max_frames;
                1 = keep the last max_frames;
                2 = uniformly sample max_frames across the entire clip.
        """
        self.anno_path = anno_path
        self.video_path = video_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2], \
            "frame_order must be 0 (normal), 1 (reverse), or 2 (random)"
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2], \
            "slice_framepos must be 0 (front), 1 (back), or 2 (uniform)"

        self.subset = subset
        assert self.subset in ["train", "test"], "subset must be 'train' or 'test'"

        # ---- Load annotation files ----
        # CSV files contain video IDs and a single global description per video.
        video_id_path_dict = {}
        video_id_path_dict["train"] = os.path.join(self.anno_path, "Charades_v1_train.csv")
        video_id_path_dict["test"] = os.path.join(self.anno_path, "Charades_v1_test.csv")

        # ---- Build the training sample list ----
        self.all_train_pairs = []
        with open(video_id_path_dict["train"]) as f:
            reader = csv.DictReader(f)
            for row in reader:
                id, descriptions = row["id"], row["descriptions"]
                self.all_train_pairs.append([id, descriptions])
        print("train len is", len(self.all_train_pairs))

        self.sample_len = len(self.all_train_pairs)

        # ---- Video frame extractor (OpenCV backend) ----
        self.rawVideoExtractor = RawVideoExtractor(
            framerate=feature_framerate,
            size=image_resolution
        )

        self.SPECIAL_TOKEN = {
            "CLS_TOKEN": "<|startoftext|>",
            "SEP_TOKEN": "<|endoftext|>",
            "MASK_TOKEN": "[MASK]",
            "UNK_TOKEN": "<|unk|>",
            "PAD_TOKEN": "<|pad|>",
        }

    def __len__(self):
        """Dataset length (number of video-caption pairs)."""
        return self.sample_len

    def _get_text(self, video_id, caption):
        """Tokenize and pad a single caption into model inputs.

        Processing pipeline:
            1. tokenizer.tokenize(caption)  => word list.
            2. Prepend CLS and append SEP tokens.
            3. Truncate if the length exceeds max_words-1 (reserving a slot for SEP).
            4. Pad to max_words with zeros (padding index).
            5. Generate the binary attention mask (1 for real tokens, 0 for pads).

        Args:
            video_id: Video identifier (returned but not used for encoding).
            caption: Raw text string.
        Returns:
            pairs_text      : Array of shape (k=1, max_words) containing input token IDs.
            pairs_mask      : Array of shape (k=1, max_words) attention mask.
            choice_video_ids: Single-element list [video_id].
        """

        choice_video_ids = [video_id]
        k = len(choice_video_ids)
        pairs_text = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.int64)

        for i, video_id in enumerate(choice_video_ids):
            words = self.tokenizer.tokenize(caption)

            # Add special tokens: CLS at the beginning, SEP at the end
            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            # Convert to IDs and build the attention mask (1 for real tokens)
            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)

            # Zero-pad to max_words
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)

        return pairs_text, pairs_mask, choice_video_ids

    def _get_rawvideo(self, choice_video_ids):
        """Load and preprocess raw video frames.

        Args:
            choice_video_ids: List of video IDs to load (typically length 1).
        Returns:
            video      : Array of shape (N, max_frames, 1, 3, H, W) video tensor.
            video_mask : 1-D binary mask of length max_frames indicating valid frames.

        Frame sampling strategy (controlled by slice_framepos):
            0 -> Keep the first max_frames.
            1 -> Keep the last max_frames.
            2 -> Uniformly sample max_frames across the entire clip.
        """
        video_mask = np.zeros(self.max_frames, dtype=np.int64)
        max_video_length = 0

        # Pre-allocate video array: (batch, max_frames, 1, C, H, W)
        video = np.zeros(
            (len(choice_video_ids), self.max_frames, 1, 3,
             self.rawVideoExtractor.size, self.rawVideoExtractor.size),
            dtype=float
        )

        for i, video_id in enumerate(choice_video_ids):
            video_path = os.path.join(self.video_path, video_id + '.mp4')

            # Extract frames via OpenCV
            raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
            raw_video_data = raw_video_data['video']

            # Only process when valid video data is returned (>3D tensor)
            if len(raw_video_data.shape) > 3:
                raw_video_data_clip = raw_video_data
                raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)

                # ---- Frame truncation / uniform sampling ----
                if self.max_frames < raw_video_slice.shape[0]:
                    if self.slice_framepos == 0:
                        video_slice = raw_video_slice[:self.max_frames, ...]
                    elif self.slice_framepos == 1:
                        video_slice = raw_video_slice[-self.max_frames:, ...]
                    else:
                        sample_indx = np.linspace(
                            0, raw_video_slice.shape[0] - 1,
                            num=self.max_frames, dtype=int
                        )
                        video_slice = raw_video_slice[sample_indx, ...]
                else:
                    video_slice = raw_video_slice

                # Apply temporal ordering augmentation
                video_slice = self.rawVideoExtractor.process_frame_order(
                    video_slice, frame_order=self.frame_order
                )

                # Update valid-frame mask
                slice_len = video_slice.shape[0]
                max_video_length = max_video_length if max_video_length > slice_len else slice_len
                if slice_len >= 1:
                    video[i][:slice_len, ...] = video_slice
            else:
                print("video path: {} error. video id: {}".format(video_path, video_id))

        # Mark valid frame positions as 1
        video_mask[:max_video_length] = [1] * max_video_length

        return video, video_mask

    def __getitem__(self, idx):
        """Return a single training sample.

        Returns:
            Tuple of (input_ids, input_mask, video, video_mask, idx, hash(idx)).
        """
        if self.subset == "train":
            vid, query = self.all_train_pairs[idx]

            input_ids, input_mask, choice_video_ids = self._get_text(vid, query)
            video, video_mask = self._get_rawvideo(choice_video_ids)

            return input_ids, input_mask, video, video_mask, idx, hash(idx)

class Charades_TestDataLoader(Dataset):
    """Charades test dataset loader.

    Loads per-video global descriptions from the test CSV.
    Used for inference and retrieval evaluation.
    """

    def __init__(
            self,
            subset,
            anno_path,
            video_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
    ):
        """Initialize the Charades test loader. Arguments mirror Charades_DataLoader."""
        self.anno_path = anno_path
        self.video_path = video_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.subset = subset
        assert self.subset in ["train", "test"]
        video_id_path_dict = {}
        video_id_path_dict["train"] = os.path.join(self.anno_path, "Charades_v1_train.csv")
        video_id_path_dict["test"] = os.path.join(self.anno_path, "Charades_v1_test.csv")
        
        # ---- Build test samples (reads from the test CSV) ----
        self.all_test_pairs = []
        with open(video_id_path_dict["test"]) as f:
            reader = csv.DictReader(f)
            for row in reader:
                id, descriptions = row["id"], row["descriptions"]
                self.all_test_pairs.append([id, descriptions])
        print("test len is", len(self.all_test_pairs))

        self.sample_len = len(self.all_test_pairs)
        self.rawVideoExtractor = RawVideoExtractor(
            framerate=feature_framerate, size=image_resolution)
        self.SPECIAL_TOKEN = {
            "CLS_TOKEN": "<|startoftext|>",
            "SEP_TOKEN": "<|endoftext|>",
            "MASK_TOKEN": "[MASK]",
            "UNK_TOKEN": "<|unk|>",
            "PAD_TOKEN": "<|pad|>",
        }

    def __len__(self):
        """Dataset length."""
        return self.sample_len

    def _get_text(self, video_id, caption):
        """Tokenize and pad a caption into model inputs. Same logic as training loader."""
        k = 1
        choice_video_ids = [video_id]
        pairs_text = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.int64)

        for i, video_id in enumerate(choice_video_ids):
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

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)

        return pairs_text, pairs_mask, choice_video_ids

    def _get_rawvideo(self, choice_video_ids):
        """Load and preprocess video frames. Same logic as training loader."""
        video_mask = np.zeros(self.max_frames, dtype=np.int64)
        max_video_length = 0

        video = np.zeros(
            (len(choice_video_ids), self.max_frames, 1, 3,
             self.rawVideoExtractor.size, self.rawVideoExtractor.size),
            dtype=float
        )

        for i, video_id in enumerate(choice_video_ids):
            video_path = os.path.join(self.video_path, video_id + '.mp4')
            raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
            raw_video_data = raw_video_data['video']

            if len(raw_video_data.shape) > 3:
                raw_video_data_clip = raw_video_data
                raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
                if self.max_frames < raw_video_slice.shape[0]:
                    if self.slice_framepos == 0:
                        video_slice = raw_video_slice[:self.max_frames, ...]
                    elif self.slice_framepos == 1:
                        video_slice = raw_video_slice[-self.max_frames:, ...]
                    else:
                        sample_indx = np.linspace(
                            0, raw_video_slice.shape[0] - 1,
                            num=self.max_frames, dtype=int
                        )
                        video_slice = raw_video_slice[sample_indx, ...]
                else:
                    video_slice = raw_video_slice

                video_slice = self.rawVideoExtractor.process_frame_order(
                    video_slice, frame_order=self.frame_order
                )

                slice_len = video_slice.shape[0]
                max_video_length = max_video_length if max_video_length > slice_len else slice_len
                if slice_len >= 1:
                    video[i][:slice_len, ...] = video_slice
            else:
                print("video path: {} error. video id: {}".format(video_path, video_id))

        video_mask[:max_video_length] = [1] * max_video_length

        return video, video_mask

    def __getitem__(self, idx):
        """Return a single test sample.

        Returns:
            Tuple of (input_ids, input_mask, video, video_mask, idx, hash(idx)).
        """
        vid, query = self.all_test_pairs[idx]

        input_ids, input_mask, choice_video_ids = self._get_text(vid, query)
        video, video_mask = self._get_rawvideo(choice_video_ids)

        return input_ids, input_mask, video, video_mask, idx, hash(idx)