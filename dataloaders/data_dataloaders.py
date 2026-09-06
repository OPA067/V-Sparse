"""DataLoader factory for video-text retrieval datasets.

Provides train / test DataLoader constructors for three supported datasets
(MSRVTT, DiDeMo, Charades) with automatic DistributedSampler fallback.
All constructors are registered in DATALOADER_DICT for dynamic dispatch.

Removed modules (files do not exist and were historically referenced):
    - lsmdc / activitynet / msvd / vatex
"""

import torch
from torch.utils.data import DataLoader

from .dataloader_charades_retrieval import Charades_DataLoader, Charades_TestDataLoader
from .dataloader_msrvtt_retrieval import MSRVTTDataset
from .dataloader_didemo_retrieval import DiDeMoDataset


# ──────────────────────────────────────────────────────────
# region MSR-VTT
# ──────────────────────────────────────────────────────────

def dataloader_msrvtt_train(args, tokenizer):
    """Build the MSR-VTT training DataLoader.

    Args:
        args: Training hyperparameters. Must contain anno_path, video_path,
            max_words, max_frames, video_framerate, batch_size, world_size, workers.
        tokenizer: Text tokenizer (e.g., CLIPTokenizer).
    Returns:
        A triplet of (dataloader, dataset_len, train_sampler).
        In distributed training sampler is non-None; on single-GPU/CPU it becomes
        None and shuffle is automatically enabled.
    """
    msrvtt_dataset = MSRVTTDataset(
        subset='train',
        anno_path=args.anno_path,
        video_path=args.video_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        video_framerate=args.video_framerate,
        config=args
    )
    try:
        train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_dataset)
    except (RuntimeError, ValueError):
        # Non-distributed environment (#cpu)
        train_sampler = None
    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size // args.world_size,
        num_workers=args.workers,
        pin_memory=False,
        shuffle=(train_sampler is None),  # shuffle only when no sampler
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(msrvtt_dataset), train_sampler


def dataloader_msrvtt_test(args, tokenizer, subset="test"):
    """Build the MSR-VTT evaluation DataLoader.

    Args:
        args: Shared config with training.
        tokenizer: Text tokenizer.
        subset: Subset name, default "test", can also be "val".
    Returns:
        A pair of (dataloader, dataset_len).
    """
    msrvtt_testset = MSRVTTDataset(
        subset=subset,
        anno_path=args.anno_path,
        video_path=args.video_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        video_framerate=args.video_framerate,
        config=args
    )

    try:
        test_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_testset)
    except (RuntimeError, ValueError):
        test_sampler = None
    dataloader_msrvtt = DataLoader(
        msrvtt_testset,
        batch_size=args.batch_size_val // args.world_size,
        num_workers=args.workers,
        shuffle=False,
        sampler=test_sampler,
        drop_last=False,
    )
    return dataloader_msrvtt, len(msrvtt_testset)


# ──────────────────────────────────────────────────────────
# region DiDeMo
# ──────────────────────────────────────────────────────────

def dataloader_didemo_train(args, tokenizer):
    """Build the DiDeMo training DataLoader. Same logic as MSR-VTT."""
    didemo_dataset = DiDeMoDataset(
        subset="train",
        data_path=args.anno_path,
        features_path=args.video_path,
        max_words=args.max_words,
        feature_framerate=args.video_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(didemo_dataset)
    dataloader = DataLoader(
        didemo_dataset,
        batch_size=args.batch_size // args.world_size,
        num_workers=args.workers,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(didemo_dataset), train_sampler


def dataloader_didemo_test(args, tokenizer, subset="test"):
    """Build the DiDeMo evaluation DataLoader. Same logic as MSR-VTT."""
    didemo_testset = DiDeMoDataset(
        subset=subset,
        data_path=args.anno_path,
        features_path=args.video_path,
        max_words=args.max_words,
        feature_framerate=args.video_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames
    )
    try:
        test_sampler = torch.utils.data.distributed.DistributedSampler(didemo_testset)
    except (RuntimeError, ValueError):
        test_sampler = None
    dataloader_didemo = DataLoader(
        didemo_testset,
        batch_size=args.batch_size_val // args.world_size,
        num_workers=args.workers,
        shuffle=False,
        sampler=test_sampler,
        drop_last=False,
    )
    return dataloader_didemo, len(didemo_testset)


# ──────────────────────────────────────────────────────────
# region Charades
# ──────────────────────────────────────────────────────────

def dataloader_charades_train(args, tokenizer):
    """Build the Charades training DataLoader."""
    charades_dataset = Charades_DataLoader(
        subset='train',
        anno_path=args.anno_path,
        video_path=args.video_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        feature_framerate=args.video_framerate,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(charades_dataset)
    dataloader = DataLoader(
        charades_dataset,
        batch_size=args.batch_size // args.world_size,
        num_workers=args.workers,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(charades_dataset), train_sampler


def dataloader_charades_test(args, tokenizer, subset="test"):
    """Build the Charades evaluation DataLoader."""
    charades_dataset = Charades_TestDataLoader(
        subset=subset,
        anno_path=args.anno_path,
        video_path=args.video_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        feature_framerate=args.video_framerate,
    )
    try:
        test_sampler = torch.utils.data.distributed.DistributedSampler(charades_dataset)
    except (RuntimeError, ValueError):
        test_sampler = None
    dataloader_charades = DataLoader(
        charades_dataset,
        batch_size=args.batch_size_val // args.world_size,
        num_workers=args.workers,
        shuffle=False,
        sampler=test_sampler,
        drop_last=False,
    )
    return dataloader_charades, len(charades_dataset)


# ──────────────────────────────────────────────────────────
# region Unified registry: 
# looked up by dataset name at runtime
# ──────────────────────────────────────────────────────────

DATALOADER_DICT = {
    "msrvtt": {
        "train": dataloader_msrvtt_train,
        "test": dataloader_msrvtt_test,
        "val": None
    },
    "didemo": {
        "train": dataloader_didemo_train,
        "test": dataloader_didemo_test,
        "val": None
    },
    "charades": {
        "train": dataloader_charades_train,
        "test": dataloader_charades_test,
        "val": None
    },
}
