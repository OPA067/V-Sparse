"""MSR-VTT dataset loader for video-text retrieval.

MSR-VTT (Microsoft Research Video to Text) is a standard benchmark for
video-to-text retrieval. It contains 10K web video clips paired with
~200K natural-language captions.

Training split (9K videos):
  Primary captions are loaded from MSRVTT_data.json. Each training sample
  consists of (caption_text, video_id).

Test split (1K videos):
  Primary captions come from MSRVTT_test.1000.csv.

Expected directory layout:
    anno_path/
        MSRVTT_train.9000.csv    # training video ID list
        MSRVTT_test.1000.csv     # test video ID list
        MSRVTT_data.json         # training annotations (200K sentences)
    video_path/
        {video_id}.mp4           # video files
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import json
import pandas as pd
from os.path import join, exists
from collections import OrderedDict
from .dataloader_retrieval import RetrievalDataset


class MSRVTTDataset(RetrievalDataset):
    """MSR-VTT dataset loader for text-video retrieval.

    Extends RetrievalDataset by overriding ``_get_anns`` to parse MSR-VTT's
    annotation files and build a pre-indexed dictionary for O(1) temporal
    caption lookup.
    """

    def __init__(self, subset, anno_path, video_path, tokenizer, max_words=32,
                 max_frames=12, video_framerate=1, image_resolution=224, mode='all', config=None):
        """Initialize the MSR-VTT dataset.

        Args:
            subset (str): Dataset split — 'train' (9K videos) or 'test' (1K videos).
            anno_path (str): Root directory containing all annotation files.
            video_path (str): Directory containing video MP4 files.
            tokenizer: Text tokenizer for converting sentences to token IDs.
            max_words (int): Maximum number of tokens per text input.
            max_frames (int): Maximum number of video frames to sample.
            video_framerate (int): Frame sampling rate in frames per second.
            image_resolution (int): Spatial resolution of resized video frames.
            mode (str): Sampling mode passed to the parent class.
            config: Optional configuration object.
        """
        super(MSRVTTDataset, self).__init__(subset, anno_path, video_path, tokenizer,
                 max_words, max_frames, video_framerate, image_resolution, mode, config=config)

    def _get_anns(self, subset='train'):
        """Load video paths and caption annotations for the given subset.

        Returns two OrderedDicts consumed by the parent dataloader:
            video_dict     : {video_id  -> absolute path to .mp4 file}
            sentences_dict : {sample_idx -> (video_id, caption_tuple)}
                             caption_tuple is a 3-tuple:
                               (caption_text, None, None)
                             - caption_text : primary sentence describing the video.

        Args:
            subset (str): 'train' (9K videos) or 'test' (1K videos).

        Returns:
            tuple: (video_dict, sentences_dict).

        Raises:
            FileNotFoundError: If the required CSV file does not exist.
        """
        # Step 1: Locate and read the video ID list for the requested split.
        csv_path = {
            'train': join(self.anno_path, 'MSRVTT_train.9000.csv'),
            'test':  join(self.anno_path, 'MSRVTT_test.1000.csv'),
        }[subset]
        if not exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        csv = pd.read_csv(csv_path)
        video_id_list = list(csv['video_id'].values)

        video_dict = OrderedDict()
        sentences_dict = OrderedDict()

        # Step 3: Build per-sample annotations, branching on split.
        if subset == 'train':
            # ----------------------------------------------------------
            # Training split
            #   Primary captions come from MSRVTT_data.json. Each entry
            #   under 'sentences' is a (video_id, caption) pair.
            # ----------------------------------------------------------
            anno_path = join(self.anno_path, 'MSRVTT_data.json')
            with open(anno_path, 'r') as f:
                data = json.load(f)

            for itm in data['sentences']:
                # Skip videos not in the current split
                if itm['video_id'] not in video_id_list:
                    continue
                sentences_dict[len(sentences_dict)] = (itm['video_id'], (itm['caption'], None, None))
                video_dict[itm['video_id']] = join(self.video_path, f"{itm['video_id']}.mp4")
        else:
            # ----------------------------------------------------------
            # Test split
            #   Primary captions come from the CSV column 'sentence'
            #   rather than from MSRVTT_data.json.
            # ----------------------------------------------------------
            for _, itm in csv.iterrows():
                sentences_dict[len(sentences_dict)] = (itm['video_id'], (itm['sentence'], None, None))
                video_dict[itm['video_id']] = join(self.video_path, f"{itm['video_id']}.mp4")

        return video_dict, sentences_dict