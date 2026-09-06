"""
TSI-TVR: Main entry-point script for text-video retrieval training and evaluation.

Pipeline stages:
    1. Command-line argument parsing and hyperparameter configuration.
    2. Random seed fix and distributed training (DDP) initialization.
    3. Model instantiation and pretrained weight loading.
    4. Training / testing data loader construction.
    5. BertAdam optimizer setup with 4 parameter groups (differentiated learning rates).
    6. Training loop execution (train_epoch) and evaluation (eval_epoch).
    7. Best model checkpointing and retrieval metric reporting (R@1/5/10, MdR, MnR).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import math
import os
import time
import random
import argparse
import numpy as np
from tqdm import tqdm
import datetime
from os.path import join, exists

import torch

from models.tokenization_clip import SimpleTokenizer as ClipTokenizer
from dataloaders.data_dataloaders import DATALOADER_DICT
from models.modeling import AllGather, Model
from models.optimization import BertAdam
from thop import profile as thop_profile
from utils.metric_logger import MetricLogger
from utils.metrics import compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim, np_softmax

from utils.comm import is_main_process, synchronize
from utils.logger import setup_logger

import warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# Region: Global definitions
# ==============================================================================

# AllGather op for distributed training: aggregates tensors across all processes
allgather = AllGather.apply

global logger


# ==============================================================================
# Region: Argument parser
# ==============================================================================

def get_args(description='Text-Video Retrieval.'):
    """Build the command-line argument parser with all hyperparameters for training and inference.

    Returns:
        argparse.Namespace: Parsed arguments object.
    """
    parser = argparse.ArgumentParser(description=description)

    # ---- Stage control ----
    parser.add_argument("--do_train", type=int, default=0, help="Enable training (1) or skip (0).")
    parser.add_argument("--do_eval", type=int, default=0, help="Run evaluation only (1) or skip (0).")

    # ---- Data configuration ----
    parser.add_argument("--datatype", type=str, default="msrvtt", help="Dataset name. Supported: msrvtt / charades / didemo / activitynet / lsmdc.")
    parser.add_argument('--anno_path', type=str, default='MSRVTT/anns', help='Directory containing annotation files.')
    parser.add_argument('--video_path', type=str, default='MSRVTT/videos', help='Directory containing video files.')
    parser.add_argument('--data_path', type=str, default='MSRVTT/', help='Path to the data pickle file.')

    # ---- Training hyperparameters ----
    parser.add_argument('--seed', type=int, default=42, help='Global random seed for reproducibility.')
    parser.add_argument('--workers', type=int, default=8, help='Number of DataLoader worker processes (affects I/O parallelism).')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate for downstream (non-CLIP) modules.')
    parser.add_argument('--coef_lr', type=float, default=1e-3, help='LR coefficient for the CLIP branch: lr_clip = lr * coef_lr.')
    parser.add_argument("--warmup_proportion", type=float, default=0.1, help='Fraction of total steps used for warmup, e.g. 0.1 means first 10%%.')
    parser.add_argument('--weight_decay', type=float, default=0.2, help='Weight decay coefficient.')
    parser.add_argument('--epochs', type=int, default=5, help='Total number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size.')
    parser.add_argument('--batch_size_val', type=int, default=32, help='Evaluation batch size.')

    # ---- Sequence / sample length constraints ----
    parser.add_argument('--max_words', type=int, default=24, help='Maximum number of text tokens per query.')
    parser.add_argument('--max_frames', type=int, default=12, help='Maximum number of sampled video frames.')
    parser.add_argument('--video_framerate', type=int, default=1, help='Video frame sampling rate.')
    parser.add_argument('--feature_framerate', type=int, default=1, help='Feature sampling rate.')

    # ---- Distributed training configuration ----
    parser.add_argument("--device", default='cpu', type=str, help='Device to run on: cpu or cuda.')
    parser.add_argument("--world_size", default=1, type=int, help='Total number of processes for distributed training.')
    # NOTE: on some systems --local_rank must be set manually
    parser.add_argument("--local-rank", default=0, type=int, help='Local rank of the current process in distributed training.')
    parser.add_argument("--distributed", default=0, type=int, help='Whether to use multi-machine DDP (1 to enable).')

    # ---- Logging & output ----
    parser.add_argument('--n_display', type=int, default=100, help='Logging frequency (log every n_display steps).')
    parser.add_argument("--output_dir", type=str, default=None, required=True, help='Output directory for model checkpoints and logs.')

    # ---- Model architecture configuration ----
    # Supported backbones: ViT-B/32, ViT-B/16
    parser.add_argument("--base_encoder", type=str, default="ViT-B/32", help='CLIP backbone variant to use.')
    parser.add_argument('--agg_module', type=str, default="seqTransf", choices=["None", "seqLSTM", "seqTransf"], help='Temporal aggregation module for video frames.')
    parser.add_argument('--interaction', type=str, default='wti', help='Text-video retrieval interaction type.')
    parser.add_argument('--num_hidden_layers', type=int, default=4, help='Number of Transformer layers in the video branch.')
    parser.add_argument("--init_model", type=str, default=None, required=False, help='Path to a pretrained checkpoint for resuming training or evaluation.')
    parser.add_argument('--split_batch', type=int, default=32, help='Chunk size for evaluation to prevent GPU OOM.')

    # ---- Loss function hyperparameters ----
    parser.add_argument('--alpha', type=float, default=0.5, help='Hyperparameter alpha.')
    parser.add_argument('--beta', type=float, default=0.1, help='Hyperparameter beta.')
    parser.add_argument('--gamma', type=float, default=0.01, help='Hyperparameter gamma.')

    args = parser.parse_args()
    return args


# ==============================================================================
# Region: Seed & Logger setup
# ==============================================================================

def set_seed_logger(args):
    """Set random seeds for reproducibility and initialize the distributed training environment.

    Steps:
        1. Fix random seeds across Python, NumPy, PyTorch, and CUDA.
        2. Disable cuDNN auto-tuner for deterministic results.
        3. If GPU is available, initialize gloo process group and synchronize all processes.
        4. Validate that batch sizes are evenly divisible by world_size.
        5. Log all effective hyperparameters.

    Returns:
        argparse.Namespace: Updated args (device and world_size may be modified).
    """
    global logger
    # Fix seeds across Python, NumPy, and PyTorch for deterministic behavior
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # Disable cuDNN auto-tuner to ensure deterministic results
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if torch.cuda.is_available():
        # Initialize process group using gloo backend (compatible with both Linux and Windows)
        torch.distributed.init_process_group(backend="gloo")
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        args.world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    if torch.cuda.is_available():
        # Barrier to ensure all processes finish initialization before proceeding
        torch.distributed.barrier()
    logger.info("local_rank: {} world_size: {}".format(args.local_rank, args.world_size))

    # Batch size must be evenly divisible by world_size in DDP
    if args.batch_size % args.world_size != 0 or args.batch_size_val % args.world_size != 0:
        raise ValueError(
            "Invalid batch_size/batch_size_val and world_size parameter: {}%{} and {}%{}, should be == 0".format(
                args.batch_size, args.world_size, args.batch_size_val, args.world_size))

    # Log all effective hyperparameters
    logger.info("Effective parameters:")
    for key in sorted(args.__dict__):
        logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args


# ==============================================================================
# Region: Model & DataLoader builders
# ==============================================================================

def build_model(args):
    """Instantiate the retrieval Model and optionally load a pretrained checkpoint.

    Args:
        args: Arguments containing model config and init_model path.

    Returns:
        torch.nn.Module: Model instance moved to args.device.
    """
    model = Model(args)
    if args.init_model:
        if not exists(args.init_model):
            raise FileNotFoundError("Pretrained model not found at: {}".format(args.init_model))
        model_state_dict = torch.load(args.init_model, map_location='cpu')
        model.load_state_dict(model_state_dict, strict=False)

    model.to(args.device)

    # ---- Log parameter counts ----
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params: {total_params / 1e6:.2f}M")
    logger.info(f"Trainable params: {trainable_params / 1e6:.2f}M")

    return model


def build_dataloader(args):
    """Construct training and testing data loaders with CLIP tokenization.

    Building order:
        1. Build test DataLoader first (if the dataset supports a test split).
        2. Build train DataLoader only when do_train is enabled, along with the
           train Sampler (DistributedSampler for distributed training).

    Returns:
        tuple: (test_dataloader, train_dataloader, train_sampler)
               Train-related items are None when do_train=False.
    """
    tokenizer = ClipTokenizer()

    # Build test data loader first if test split is available for the dataset
    test_dataloader, test_length = None, 0
    if DATALOADER_DICT[args.datatype]["test"] is not None:
        test_dataloader, test_length = DATALOADER_DICT[args.datatype]["test"](args, tokenizer)
        logger.info("***** Running testing *****")
        logger.info("  Num examples = %d", test_length)
        logger.info("  Batch size   = %d", args.batch_size_val)
        logger.info("  Num steps    = %d", len(test_dataloader))

    # Build training data loader only when training is enabled
    if args.do_train:
        train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_length)
        logger.info("  Batch size   = %d", args.batch_size)
        logger.info("  Num steps    = %d", len(train_dataloader))
    else:
        train_dataloader, train_sampler = None, None

    return test_dataloader, train_dataloader, train_sampler


# ==============================================================================
# Region: Optimizer setup
# ==============================================================================

def prep_optimizer(args, model, num_train_optimization_steps, local_rank):
    """Configure the BertAdam optimizer with 4 parameter groups using differentiated learning rates.

    Parameters are partitioned along two axes:
        A. Whether they belong to the CLIP backbone (lower LR, name contains "clip.").
        B. Whether weight decay should be applied (excludes bias / LayerNorm).
    This yields 4 groups:
        1. CLIP + decay       -> lr = args.lr * coef_lr
        2. Non-CLIP + decay   -> lr = args.lr
        3. CLIP + no_decay    -> lr = args.lr * coef_lr, weight_decay = 0
        4. Non-CLIP + no_decay -> lr = args.lr, weight_decay = 0

    Args:
        args: Arguments containing lr, coef_lr, weight_decay, warmup_proportion.
        model: Model to optimize (may be wrapped in DDP).
        num_train_optimization_steps: Total training steps (epoch * steps_per_epoch),
            used for warmup scheduling.
        local_rank: Local process rank, used for DDP device_ids.

    Returns:
        tuple: (optimizer, scheduler, model)
               - optimizer: BertAdam instance.
               - scheduler: None (schedule managed internally by BertAdam).
               - model: DDP-wrapped model if GPU is available.
    """
    if hasattr(model, 'module'):
        model = model.module
    lr = args.lr
    coef_lr = args.coef_lr
    weight_decay = args.weight_decay
    warmup_proportion = args.warmup_proportion
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    # Group by whether weight_decay is applied
    decay_param_tp = [(n, p) for n, p in param_optimizer
                      if not any(nd in n for nd in no_decay)]
    no_decay_param_tp = [(n, p) for n, p in param_optimizer
                         if any(nd in n for nd in no_decay)]

    # Further split by whether the parameter belongs to the CLIP branch
    decay_clip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." in n]
    decay_noclip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." not in n]

    no_decay_clip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." in n]
    no_decay_noclip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." not in n]

    optimizer_grouped_parameters = [
        {'params': [p for n, p in decay_clip_param_tp],
         'weight_decay': weight_decay, 'lr': lr * coef_lr},
        {'params': [p for n, p in decay_noclip_param_tp],
         'weight_decay': weight_decay},
        {'params': [p for n, p in no_decay_clip_param_tp],
         'weight_decay': 0.0, 'lr': lr * coef_lr},
        {'params': [p for n, p in no_decay_noclip_param_tp],
         'weight_decay': 0.0}
    ]

    scheduler = None

    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=warmup_proportion,
                        schedule='warmup_cosine', b1=0.9, b2=0.98, e=1e-6,
                        t_total=num_train_optimization_steps, weight_decay=weight_decay, max_grad_norm=1.0)

    if torch.cuda.is_available():
        model = torch.nn.parallel.DistributedDataParallel(
          model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    return optimizer, scheduler, model


# ==============================================================================
# Region: Checkpoint utilities
# ==============================================================================

def save_model(epoch, args, model, type_name=""):
    """Save model weights for the current epoch to disk.

    Args:
        epoch (int): Current epoch index.
        args: Arguments containing output_dir.
        model: Model to save (DDP wrapper auto-unwrapped if present).
        type_name (str, optional): Extra tag for the filename, e.g. 'best'.

    Returns:
        str: Full path of the saved file.
    """
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = join(args.output_dir, "pytorch_model.bin.{}{}".format(
        "" if type_name == "" else type_name + ".", epoch))
    torch.save(model_to_save.state_dict(), output_model_file)
    logger.info("Model saved to %s", output_model_file)
    return output_model_file


def reduce_loss(loss, args):
    """Aggregate loss across multiple GPUs and average it at rank 0.

    Args:
        loss (torch.Tensor): Scalar loss on the current process.
        args: Arguments containing world_size.

    Returns:
        torch.Tensor: Reduced loss (average at rank 0; undefined on non-dst ranks).
    """
    world_size = args.world_size
    if world_size < 2:
        return loss
    with torch.no_grad():
        torch.distributed.reduce(loss, dst=0)
        if torch.distributed.get_rank() == 0:
            loss /= world_size
    return loss


# ==============================================================================
# Region: Training
# ==============================================================================

def train_epoch(epoch, args, model, train_dataloader,
                device, n_gpu, optimizer, scheduler, global_step, max_steps):
    """Execute one full epoch of model training.

    Per-step flow:
        1. Fetch batch from DataLoader: (query_token, query_word_mask, video_frame,
           video_frame_mask, idx, ...).
        2. Forward pass: compute contrastive loss.
        3. Average loss when using multiple GPUs.
        4. Backward pass + gradient clipping (clip_grad_norm_ max_norm=1.0).
        5. Optimizer step, optional scheduler step.
        6. Read and log current learning rates across all param groups.
        7. zero_grad.
        8. Clamp CLIP temperature parameter logit_scale so it does not exceed ln(100).
        9. Update MetricLogger and periodically print logs (eta, epoch, iter, loss,
           lr, logit_scale, GPU memory).

    Args:
        epoch (int): Current epoch index (0-based).
        args: Parsed command-line arguments.
        model: Retrieval model (DDP-wrapped or bare).
        train_dataloader: Training data loader.
        device: Device for training.
        n_gpu (int): Number of GPUs available.
        optimizer: BertAdam optimizer instance.
        scheduler: Learning-rate scheduler (or None).
        global_step (int): Running step counter across epochs; updated in-place.
        max_steps (int): Total optimization steps across all epochs.

    Returns:
        tuple: (avg_loss, global_step)
            avg_loss    — Mean reduced loss over the epoch.
            global_step — Updated step counter.
    """
    global logger
    global best_score
    global meters

    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    total_loss = 0

    end = time.time()
    for step, batch in enumerate(train_dataloader, start=1):
        global_step += 1
        data_time = time.time() - end

        # Unpack batch: query token IDs, query mask, video frames, frame mask, sample indices
        query, query_word_mask, video, video_frame_mask, idx, _ = batch
        if n_gpu == 1:
            query, query_word_mask, video, video_frame_mask, idx = [
                x.to(device=device, non_blocking=True)
                for x in [query, query_word_mask, video, video_frame_mask, idx]]

        # Forward pass: compute contrastive loss
        loss = model(query, query_word_mask, video, video_frame_mask, idx, global_step)

        if n_gpu > 1:
            loss = loss.mean()

        with torch.autograd.detect_anomaly():
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Must call get_lr() before zero_grad(), because BertAdam.get_lr() skips
        # parameters whose p.grad is None (which zero_grad() would set).
        current_lrs = optimizer.get_lr()

        optimizer.zero_grad()

        # Clamp CLIP temperature parameter to prevent logit_scale from exceeding ln(100)
        if hasattr(model, 'module'):
            torch.clamp_(model.module.clip.logit_scale.data, max=np.log(100))
            logit_scale = model.module.clip.logit_scale.exp().item()
        else:
            torch.clamp_(model.clip.logit_scale.data, max=np.log(100))
            logit_scale = model.clip.logit_scale.exp().item()

        batch_time = time.time() - end
        end = time.time()

        # Synchronize loss across GPUs and update metric logger
        reduced_l = reduce_loss(loss, args)
        total_loss += float(reduced_l)
        meters.update(time=batch_time, data=data_time, loss=float(reduced_l))

        # Estimate remaining time based on average batch duration
        eta_seconds = meters.time.global_avg * (max_steps - global_step)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if (global_step % log_step == 0 or global_step == 1) and is_main_process():
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}, ",
                        "epoch: {epoch}/{max_epoch}, ",
                        "iter: {step}/{len}/{iter}/{max_iter}, ",
                        "{meters}",
                        "lr: {lr}, ",
                        "logit: {logit_scale}, ",
                        "memory: {memory:.2f}GB",
                    ]
                ).format(
                    eta=eta_string,
                    epoch=epoch + 1,
                    max_epoch=args.epochs,
                    step=step,
                    len=len(train_dataloader),
                    iter=global_step,
                    max_iter=max_steps,
                    meters=str(meters),
                    lr="/".join([str('%.9f' % itm) for itm in sorted(list(set(current_lrs)))]),
                    logit_scale=logit_scale,
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 / 1024.0,
                )
            )
    total_loss = total_loss / len(train_dataloader)

    return total_loss, global_step


# ==============================================================================
# Region: Evaluation
# ==============================================================================

def _run_on_single_gpu(model,
            batch_qs_feat, batch_qs_mask, batch_qw_feat, batch_qw_mask,
            batch_vf_feat, batch_vf_mask, batch_vp_feat, batch_vp_mask,
            split_batch=32):
    """Compute cross-modal similarity matrices chunk-by-chunk to avoid GPU OOM.

    Strategy: Split query features into chunks of size 1 and video features into
    chunks of size ``split_batch``, compute pairwise logits for each (query_chunk,
    video_chunk) pair via ``model.get_similarity_logits``, then concatenate to
    form the full similarity matrix.

    Args:
        model: Retrieval model with ``get_similarity_logits`` method.
        batch_qs_feat (Tensor): Query sentence-level features, [a, d].
        batch_qs_mask (Tensor): Query sentence mask, [a, 1].
        batch_qw_feat (Tensor): Query word-level features, [a, w, d].
        batch_qw_mask (Tensor): Query word mask, [a, w].
        batch_vf_feat (Tensor): Video frame-level features, [b, f, d].
        batch_vf_mask (Tensor): Video frame mask, [b, f].
        batch_vp_feat (Tensor): Video patch-level features, [b, total_patches, d].
        batch_vp_mask (Tensor): Video patch mask, [b, total_patches].
        split_batch (int): Chunk size for splitting video-side feature tensors
            (query-side features are always split by 1).

    Returns:
        list[list[np.ndarray]]: Nested list of chunk similarity matrices.
            Outer length = number of query chunks; inner length = number of video chunks.
    """
    sim_matrix = []

    # Chunk all feature tensors according to split_batch
    batch_qs_feat, batch_qs_mask, batch_qw_feat, batch_qw_mask = [
        torch.split(f, split_batch)
        for f in (batch_qs_feat, batch_qs_mask, batch_qw_feat, batch_qw_mask)
    ]

    batch_vf_feat, batch_vf_mask, batch_vp_feat, batch_vp_mask = [
        torch.split(f, split_batch)
        for f in (batch_vf_feat, batch_vf_mask, batch_vp_feat, batch_vp_mask)
    ]

    with torch.no_grad():
        for idx1, (qs_feat, qs_mask, qw_feat, qw_mask) in tqdm(enumerate(zip(batch_qs_feat, batch_qs_mask, batch_qw_feat, batch_qw_mask))):
            each_row = []
            for idx2, (vf_feat, vf_mask, vp_feat, vp_mask) in enumerate(zip(batch_vf_feat, batch_vf_mask, batch_vp_feat, batch_vp_mask)):
                logits = model.get_similarity_logits(qs_feat, qs_mask, qw_feat, qw_mask, vf_feat, vf_mask, vp_feat, vp_mask)
                logits = logits.cpu().detach().numpy()
                each_row.append(logits)
            each_row = np.concatenate(tuple(each_row), axis=-1)
            sim_matrix.append(each_row)

    return sim_matrix


def eval_epoch(args, model, test_dataloader, device):
    """Evaluate the model on the test set and report retrieval metrics.

    Evaluation pipeline:
        1. Extract all query text features and video features (forward, no grad).
        2. Synchronize features across all GPUs via allgather.
        3. Compute full similarity matrix chunk-by-chunk via ``_run_on_single_gpu``
           (query chunks of size 1, video chunks of size split_batch).
        4. Handle multi-sentence video scenarios (e.g. MSRVTT multiple captions per video).
        5. Calculate Text->Video and Video->Text retrieval metrics:
           R@1, R@5, R@10, MdR (Median Rank), MnR (Mean Rank), RSum.

    Args:
        args: Parsed command-line arguments (includes split_batch, local_rank).
        model: Retrieval model (DDP-wrapped or bare).
        test_dataloader: Test data loader yielding (query, video, ...) batches.
        device: Device for inference.

    Returns:
        float: Text->Video R@1 score for model selection.
    """
    global test_dataset

    # Unwrap DDP wrapper if present so we access the raw model methods
    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    # Detect multi-sentence video datasets (e.g., MSRVTT with multiple captions per video)
    multi_sentence_ = False
    cut_off_points_, sentence_num_, video_num_ = [], -1, -1
    if hasattr(test_dataloader.dataset, 'multi_sentence_per_video') \
            and test_dataloader.dataset.multi_sentence_per_video:
        multi_sentence_ = True
        cut_off_points_ = test_dataloader.dataset.cut_off_points
        sentence_num_ = test_dataloader.dataset.sentence_num
        video_num_ = test_dataloader.dataset.video_num
        cut_off_points_ = [itm - 1 for itm in cut_off_points_]

    if multi_sentence_:
        logger.warning("Eval under the multi-sentence per video clip setting.")
        logger.warning("sentence num: {}, video num: {}".format(
            sentence_num_, video_num_))

    model.eval()
    logger.info("Model begins to testing...")

    # Accumulate all query text features and video features
    ids = []
    # Query sentence / word features and masks
    batch_qs_feat, batch_qs_mask, batch_qw_feat, batch_qw_mask = [], [], [], []
    # Video frame / patch features and masks
    batch_vf_feat, batch_vf_mask, batch_vp_feat, batch_vp_mask = [], [], [], []

    with torch.no_grad():
        tic = time.time()
        for batch in tqdm(test_dataloader):
            query, query_word_mask, video, video_frame_mask, idx, _  = batch
            query, query_word_mask, video, video_frame_mask, idx = [
                x.to(device) for x in [query, query_word_mask, video, video_frame_mask, idx]]

            # Extract text-side features (sentence-level + word-level)
            qs_feat, qw_feat = model.get_text_feat(query, query_word_mask)
            # Extract video-side features (frame-level + patch-level)
            vf_feat, vp_feat = model.get_video_feat(video, video_frame_mask)

            # Query sentence mask always valid (one sentence per query)
            qs_mask = qs_feat.new_ones(qs_feat.size(0), 1)
            # Derive patch mask from frame mask: each frame is expanded to its patches
            patches_per_frame = vp_feat.size(1) // vf_feat.size(1) if vf_feat.size(1) > 0 else 0
            vp_mask = video_frame_mask.unsqueeze(-1).expand(-1, -1, patches_per_frame).reshape(video_frame_mask.size(0), -1)
  
            ids.append(idx)
            batch_qs_feat.append(qs_feat)
            batch_qs_mask.append(qs_mask)
            batch_qw_feat.append(qw_feat)
            batch_qw_mask.append(query_word_mask)

            batch_vf_feat.append(vf_feat)
            batch_vf_mask.append(video_frame_mask)
            batch_vp_feat.append(vp_feat)
            batch_vp_mask.append(vp_mask)

        # Gather features across all GPUs for full-dataset evaluation
        ids = allgather(torch.cat(ids, dim=0), args).squeeze()
        batch_qs_feat = allgather(torch.cat(batch_qs_feat, dim=0), args)
        batch_qs_mask = allgather(torch.cat(batch_qs_mask, dim=0), args)
        batch_qw_feat = allgather(torch.cat(batch_qw_feat, dim=0), args)
        batch_qw_mask = allgather(torch.cat(batch_qw_mask, dim=0), args)

        batch_vf_feat = allgather(torch.cat(batch_vf_feat, dim=0), args)
        batch_vf_mask = allgather(torch.cat(batch_vf_mask, dim=0), args)
        batch_vp_feat = allgather(torch.cat(batch_vp_feat, dim=0), args)
        batch_vp_mask = allgather(torch.cat(batch_vp_mask, dim=0), args)

    toc1 = time.time()

    with torch.no_grad():
        sim_matrix = _run_on_single_gpu(model,
            batch_qs_feat, batch_qs_mask, batch_qw_feat, batch_qw_mask,
            batch_vf_feat, batch_vf_mask, batch_vp_feat, batch_vp_mask,
            args.split_batch)
        sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
    toc2 = time.time()

    if multi_sentence_:
        # Handle multi-sentence videos: reshape [sentence_num, video_num] -> [video_num, sentence_num, video_num]
        logger.info("before reshape: {} x {}".format(
            sim_matrix.shape[0], sim_matrix.shape[1]))
        # Convert cut_off_points to segment boundaries
        cut_off_points2len_ = [itm + 1 for itm in cut_off_points_]
        max_length = max([e_ - s_ for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_)])
        sim_matrix_new = []
        for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_):
            # Pad shorter segments with -inf so stacking works uniformly
            sim_matrix_new.append(np.concatenate((sim_matrix[s_:e_],
                 np.full((max_length - e_ + s_, sim_matrix.shape[1]), -np.inf)), axis=0))
        sim_matrix = np.stack(tuple(sim_matrix_new), axis=0)
        logger.info("after reshape: {} x {} x {}".format(
            sim_matrix.shape[0], sim_matrix.shape[1], sim_matrix.shape[2]))

        # Compute Text->Video and Video->Text retrieval metrics
        tv_metrics = tensor_text_to_video_metrics(sim_matrix)
        vt_metrics = compute_metrics(tensor_video_to_text_sim(sim_matrix))
        toc3 = time.time()
        logger.info("time: feat {:.1f}s match {:.5f}s metrics {:.5f}s".format(
            toc1 - tic, toc2 - toc1, toc3 - toc2))

        tv_metrics['RSum'] = tv_metrics['R1'] + tv_metrics['R5'] + tv_metrics['R10']
        logger.info(
            "T->V: R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} "
            "- R@Sum: {:.1f} - MdR: {:.1f} - MnR: {:.1f}".format(
                tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'],
                tv_metrics['RSum'], tv_metrics['MR'], tv_metrics['MeanR']))
        vt_metrics['RSum'] = vt_metrics['R1'] + vt_metrics['R5'] + vt_metrics['R10']
        logger.info(
            "V->T: R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} "
            "- R@Sum: {:.1f} - MdR: {:.1f} - MnR: {:.1f}".format(
                vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'],
                vt_metrics['RSum'], vt_metrics['MR'], vt_metrics['MeanR']))
        return tv_metrics['R1']
    else:
        # Single-sentence video: normalize with softmax before computing metrics
        logger.info("sim matrix: {} x {}".format(
            sim_matrix.shape[0], sim_matrix.shape[1]))
        tv_metrics = compute_metrics(np_softmax(sim_matrix))
        vt_metrics = compute_metrics(np_softmax(sim_matrix.T))
        logger.info('Length-T: {}, Length-V: {}'.format(len(sim_matrix), len(sim_matrix[0])))
        toc3 = time.time()
        logger.info("time: feat {:.1f}s match {:.5f}s metrics {:.5f}s".format(
            toc1 - tic, toc2 - toc1, toc3 - toc2))

        tv_metrics['RSum'] = tv_metrics['R1'] + tv_metrics['R5'] + tv_metrics['R10']
        logger.info(
            "T->V: R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} "
            "- R@Sum: {:.1f} - MdR: {:.1f} - MnR: {:.1f}".format(
                tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'],
                tv_metrics['RSum'], tv_metrics['MR'], tv_metrics['MeanR']))
        vt_metrics['RSum'] = vt_metrics['R1'] + vt_metrics['R5'] + vt_metrics['R10']
        logger.info(
            "V->T: R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} "
            "- R@Sum: {:.1f} - MdR: {:.1f} - MnR: {:.1f}".format(
                vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'],
                vt_metrics['RSum'], vt_metrics['MR'], vt_metrics['MeanR']))
        return tv_metrics['R1']


# ==============================================================================
# Region: Main entry
# ==============================================================================

def main():
    """Orchestrate the full training or evaluation pipeline.

    Steps:
        1. Parse command-line arguments, create a timestamped output directory, and initialize logger.
        2. Fix random seeds and initialize distributed training (DDP).
        3. Build the model and optionally load pretrained weights.
        4. Construct training / testing data loaders.
        5. If do_train:
            a. Build the optimizer (BertAdam) and scheduler.
            b. For each epoch: train -> evaluate -> save best model.
            c. After training, reload the best checkpoint for final evaluation.
        6. If do_eval (evaluation only):
            Call eval_epoch directly.
        7. Synchronize across all ranks.
    """
    global logger
    global best_score
    global meters

    meters = MetricLogger(delimiter="")
    args = get_args()
    # Add timestamp to output directory to avoid overwriting previous runs
    args.output_dir = args.output_dir + "/" + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    if not exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger('Model', args.output_dir, args.local_rank)

    args = set_seed_logger(args)
    model = build_model(args)
    test_dataloader, train_dataloader, train_sampler = build_dataloader(args)

    if args.do_train:
        tic = time.time()
        max_steps = len(train_dataloader) * args.epochs
        _max_steps = len(train_dataloader) * args.epochs
        optimizer, scheduler, model = prep_optimizer(
            args, model, _max_steps, args.local_rank)

        best_score = 0.00001
        best_output_model_file = "None"
        global_step = 0
        logger.info("Model begins to training...")
        for epoch in range(args.epochs):
            if train_sampler is not None:
                # Set epoch for distributed sampler to reshuffle data each epoch
                train_sampler.set_epoch(epoch)
            synchronize()

            # Zero-shot baseline evaluation before any training updates in epoch 0.
            if epoch == 0:
                torch.cuda.empty_cache()
                logger.info("Model zero-shot text-video retrieval")
                R1 = eval_epoch(args, model, test_dataloader, args.device)
                logger.info("Zero-shot R1: {:.4f}".format(R1))

            # Training phase for the current epoch
            torch.cuda.empty_cache()
            tr_loss, global_step = train_epoch(
                epoch, args, model, train_dataloader, args.device,
                args.world_size, optimizer, scheduler, global_step, max_steps)
            torch.cuda.empty_cache()

            # Evaluation phase after the training epoch
            R1 = eval_epoch(args, model, test_dataloader, args.device)
            torch.cuda.empty_cache()
            synchronize()

            # Only rank 0 saves checkpoints and maintains best_score
            if args.local_rank == 0:
                output_model_file = save_model(epoch, args, model, type_name="")
                if best_score <= R1:
                    best_score = R1
                    best_output_model_file = output_model_file
                    torch.save(
                        model.module.state_dict()
                        if hasattr(model, 'module') else model.state_dict(),
                            os.path.join(args.output_dir, 'best.pth'))
                logger.info("Best model: {}, R1: {:.4f}".format(best_output_model_file, best_score))
            synchronize()

        toc = time.time() - tic
        training_time = time.strftime("%Hh %Mmin %Ss", time.gmtime(toc))
        logger.info("*" * 20 + '\n'
                    + f'training finished with {training_time}'
                    + "*" * 20 + '\n')

        # Reload the best checkpoint for final evaluation
        # Unwrap DDP wrapper so state_dict keys match the bare model
        model = model.module
        if args.local_rank == 0:
            model.load_state_dict(
                torch.load(best_output_model_file, map_location='cpu'),
                strict=False)
        if torch.cuda.is_available():
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.local_rank],
                find_unused_parameters=True)

        torch.cuda.empty_cache()
        eval_epoch(args, model, test_dataloader, args.device)
        synchronize()

    elif args.do_eval:
        eval_epoch(args, model, test_dataloader, args.device)


if __name__ == "__main__":
    main()
