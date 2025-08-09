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
from PIL import Image

import torch

from models.tokenization_clip import SimpleTokenizer as ClipTokenizer
from dataloaders.data_dataloaders import DATALOADER_DICT
from models.modeling import AllGather, Model
from models.optimization import BertAdam
from utils.metric_logger import MetricLogger
from utils.metrics import compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim, np_softmax

from utils.comm import is_main_process, synchronize
from utils.logger import setup_logger

import warnings
warnings.filterwarnings("ignore")

allgather = AllGather.apply

global logger

def get_args(description='Text-Video Retrieval.'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_train", type=int, default=0, help="Whether to run training.")
    parser.add_argument("--do_eval", type=int, default=0, help="Whether to run evaluation.")

    parser.add_argument("--datatype", type=str, default="msrvtt", help="Point the dataset to finetune.")
    parser.add_argument('--anno_path', type=str, default='MSRVTT/anns', help='annotation path')
    parser.add_argument('--video_path', type=str, default='MSRVTT/videos', help='video path')
    parser.add_argument('--data_path', type=str, default='MSRVTT/', help='data pickle file path')

    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--workers', type=int, default=8, help='number of data loading workers (default: 4)')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--coef_lr', type=float, default=1e-3, help='coefficient for bert branch.')
    parser.add_argument("--warmup_proportion", type=float, default=0.1, help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% of training.")
    parser.add_argument('--weight_decay', type=float, default=0.2, help='weight decay')
    parser.add_argument('--epochs', type=int, default=5, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=32, help='batch size eval')

    parser.add_argument('--max_words', type=int, default=24, help='max text token number')
    parser.add_argument('--max_frames', type=int, default=12, help='max key frames')
    parser.add_argument('--video_framerate', type=int, default=1, help='framerate to sample video frame')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')

    parser.add_argument("--device", default='cpu', type=str, help="cpu/cuda")
    parser.add_argument("--world_size", default=1, type=int, help="distribted training")
    # maybe you should set as --local_rank based on system.
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    parser.add_argument("--distributed", default=0, type=int, help="multi machine DDP")

    parser.add_argument('--n_display', type=int, default=100, help='Information display frequence')
    parser.add_argument("--output_dir", type=str, default=None, required=True, help="The output directory where the model predictions and checkpoints will be written.")
    # base_encoder = ["ViT-B/32", "ViT-B/16"]
    parser.add_argument("--base_encoder", type=str, default="ViT-B/32", help="Choose a CLIP version")
    parser.add_argument('--agg_module', type=str, default="seqTransf", choices=["None", "seqLSTM", "seqTransf"], help="choice a feature aggregation module for video.")
    parser.add_argument('--interaction', type=str, default='wti', help="interaction type for retrieval.")
    parser.add_argument('--num_hidden_layers', type=int, default=4)
    parser.add_argument("--init_model", type=str, default=None, required=False, help="Initial model.")
    parser.add_argument('--split_batch', type=int, default=32, help='test dataset split')
    parser.add_argument('--alpha', type=float, default=0.5, help='hyper-parameters alpha')
    parser.add_argument('--beta', type=float, default=0.5, help='hyper-parameters beta')
    parser.add_argument('--gamma', type=float, default=0.01, help='hyper-parameters gamma')

    args = parser.parse_args()

    return args

def set_seed_logger(args):
    global logger
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if torch.cuda.is_available():
        # torch.distributed.init_process_group(backend="nccl")
        torch.distributed.init_process_group(backend="gloo")
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        args.world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    if torch.cuda.is_available():
        torch.distributed.barrier()
    logger.info("local_rank: {} world_size: {}".format(args.local_rank, args.world_size))

    if args.batch_size % args.world_size != 0 or args.batch_size_val % args.world_size != 0:
        raise ValueError(
            "Invalid batch_size/batch_size_val and world_size parameter: {}%{} and {}%{}, should be == 0".format(
                args.batch_size, args.world_size, args.batch_size_val, args.world_size))

    logger.info("Effective parameters:")
    for key in sorted(args.__dict__):
        logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args

def build_model(args):
    model = Model(args)
    if args.init_model:
        if not exists(args.init_model):
            raise FileNotFoundError
        model_state_dict = torch.load(args.init_model, map_location='cpu')
        model.load_state_dict(model_state_dict, strict=False)

    model.to(args.device)
    return model

def build_dataloader(args):

    tokenizer = ClipTokenizer()

    # Here you can add val datasets.
    test_dataloader, test_length = None, 0
    if DATALOADER_DICT[args.datatype]["test"] is not None:
        test_dataloader, test_length = DATALOADER_DICT[args.datatype]["test"](args, tokenizer)
        logger.info("***** Running testing *****")
        logger.info("  Num examples = %d", test_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(test_dataloader))

    if args.do_train:
        train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_length)
        logger.info("  Batch size = %d", args.batch_size)
        logger.info("  Num steps = %d", len(train_dataloader))
    else:
        train_dataloader, train_sampler = None, None

    return test_dataloader, train_dataloader, train_sampler


def prep_optimizer(args, model, num_train_optimization_steps, local_rank):
    if hasattr(model, 'module'):
        model = model.module
    lr = args.lr
    coef_lr = args.coef_lr
    weight_decay = args.weight_decay
    warmup_proportion = args.warmup_proportion
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    decay_clip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." in n]
    decay_noclip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." not in n]

    no_decay_clip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." in n]
    no_decay_noclip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." not in n]

    optimizer_grouped_parameters = [
        {'params': [p for n, p in decay_clip_param_tp], 'weight_decay': weight_decay, 'lr': lr * coef_lr},
        {'params': [p for n, p in decay_noclip_param_tp], 'weight_decay': weight_decay},
        {'params': [p for n, p in no_decay_clip_param_tp], 'weight_decay': 0.0, 'lr': lr * coef_lr},
        {'params': [p for n, p in no_decay_noclip_param_tp], 'weight_decay': 0.0}
    ]

    scheduler = None

    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=warmup_proportion,
                         schedule='warmup_cosine', b1=0.9, b2=0.98, e=1e-6,
                         t_total=num_train_optimization_steps, weight_decay=weight_decay,
                         max_grad_norm=1.0)

    if torch.cuda.is_available():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                                          find_unused_parameters=True)
    return optimizer, scheduler, model


def save_model(epoch, args, model, type_name=""):
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = join(
        args.output_dir, "pytorch_model.bin.{}{}".format("" if type_name == "" else type_name + ".", epoch))
    torch.save(model_to_save.state_dict(), output_model_file)
    logger.info("Model saved to %s", output_model_file)
    return output_model_file

def reduce_loss(loss, args):
    world_size = args.world_size
    if world_size < 2:
        return loss
    with torch.no_grad():
        torch.distributed.reduce(loss, dst=0)
        if torch.distributed.get_rank() == 0:
            loss /= world_size
    return loss

def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer, scheduler, global_step, max_steps):
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

        if n_gpu == 1:
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

        text, text_mask, video, video_mask, idx, _ = batch
        loss = model(text, text_mask, video, video_mask, idx, global_step)

        if n_gpu > 1:
            loss = loss.mean()

        with torch.autograd.detect_anomaly():
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        optimizer.zero_grad()

        if hasattr(model, 'module'):
            torch.clamp_(model.module.clip.logit_scale.data, max=np.log(100))
            logit_scale = model.module.clip.logit_scale.exp().item()
        else:
            torch.clamp_(model.clip.logit_scale.data, max=np.log(100))
            logit_scale = model.clip.logit_scale.exp().item()

        batch_time = time.time() - end
        end = time.time()

        reduced_l = reduce_loss(loss, args)
        meters.update(time=batch_time, data=data_time, loss=float(reduced_l))

        eta_seconds = meters.time.global_avg * (max_steps - global_step)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if (global_step % log_step == 0 or global_step == 1) and is_main_process():
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}, ",
                        "epoch: {epoch}/{max_epoch}, ",
                        "iteration: {step}/{len}/{iteration}/{max_iteration}, ",
                        "{meters}",
                        "lr: {lr}, ",
                        "logit: {logit_scale}, ",
                        "memory: {memory:.2f}GB",
                    ]
                ).format(
                    eta=eta_string,
                    epoch=epoch+1,
                    max_epoch=args.epochs,
                    step=step,
                    len=len(train_dataloader),
                    iteration=global_step,
                    max_iteration=max_steps,
                    meters=str(meters),
                    lr="/".join([str('%.9f' % itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                    logit_scale=logit_scale,
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 / 1024.0,
                )
            )
    total_loss = total_loss / len(train_dataloader)
    return total_loss, global_step

def _run_on_single_gpu(model, batch_s_feat, batch_w_feat, batch_w_mask, batch_f_feat, batch_p_feat, batch_f_mask, split_batch=32):

    sim_matrix = []

    batch_s_feat = torch.split(batch_s_feat, split_batch)
    batch_w_feat = torch.split(batch_w_feat, split_batch)
    batch_w_mask = torch.split(batch_w_mask, split_batch)
    batch_f_feat = torch.split(batch_f_feat, split_batch)
    batch_p_feat = torch.split(batch_p_feat, split_batch)
    batch_f_mask = torch.split(batch_f_mask, split_batch)

    with torch.no_grad():
        for idx1, (s_feat, w_feat, w_mask) in tqdm(enumerate(zip(batch_s_feat, batch_w_feat, batch_w_mask))):
            each_row = []
            for idx2, (f_feat, p_feat, f_mask) in enumerate(zip(batch_f_feat, batch_p_feat, batch_f_mask)):
                logits = model.get_similarity_logits(s_feat, w_feat, w_mask, f_feat, p_feat, f_mask)
                logits = logits.cpu().detach().numpy()
                each_row.append(logits)
            each_row = np.concatenate(tuple(each_row), axis=-1)
            sim_matrix.append(each_row)
    return sim_matrix

def eval_epoch(args, model, test_dataloader, device):
    global test_dataset

    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

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
        logger.warning("sentence num: {}, video num: {}".format(sentence_num_, video_num_))

    model.eval()
    logger.info("Model begins to testing...")

    ids_t, batch_mask_t, batch_feat_s, batch_feat_w = [], [], [], []
    ids_v, batch_mask_v, batch_feat_f, batch_feat_p = [], [], [], []

    with torch.no_grad():
        tic = time.time()
        if multi_sentence_:
            total_video_num = 0
            for batch in tqdm(test_dataloader):
                batch = tuple(t.to(device) for t in batch)
                text, text_mask, video, video_mask, inds, index = batch

                b, *_t = video.shape
                s_feat, w_feat = model.get_text_feat(text, text_mask)
                ids_t.append(inds)
                batch_mask_t.append(text_mask)
                batch_feat_s.append(s_feat)
                batch_feat_w.append(w_feat)

                s_, e_ = total_video_num, total_video_num + b
                filter_inds = [itm - s_ for itm in cut_off_points_ if itm >= s_ and itm < e_]

                if len(filter_inds) > 0:
                    vide, video_mask = video[filter_inds, ...], video_mask[filter_inds, ...]

                    f_feat, p_feat = model.get_video_feat(video, video_mask)
                    ids_v.append(inds)
                    batch_feat_f.append(f_feat)
                    batch_mask_v.append(video_mask)
                    batch_feat_p.append(p_feat)
                total_video_num += b

            ids_t = torch.cat(ids_t, dim=0).squeeze()
            ids_v = torch.cat(ids_v, dim=0).squeeze()
            batch_mask_t = torch.cat(batch_mask_t, dim=0)
            batch_feat_s = torch.cat(batch_feat_s, dim=0)
            batch_feat_w = torch.cat(batch_feat_w, dim=0)
            batch_mask_v = torch.cat(batch_mask_v, dim=0)
            batch_feat_f = torch.cat(batch_feat_f, dim=0)
            batch_feat_p = torch.cat(batch_feat_p, dim=0)
        else:
            for batch in tqdm(test_dataloader):
                batch = tuple(t.to(device) for t in batch)
                text, text_mask, video, video_mask, inds, index = batch
                s_feat, w_feat, f_feat, p_feat = model.get_text_video_feat(text, text_mask, video, video_mask)
                ids_t.append(inds)
                ids_v.append(inds)
                batch_mask_t.append(text_mask)
                batch_mask_v.append(video_mask)
                batch_feat_s.append(s_feat)
                batch_feat_w.append(w_feat)
                batch_feat_f.append(f_feat)
                batch_feat_p.append(p_feat)
            ids_t = allgather(torch.cat(ids_t, dim=0), args).squeeze()
            batch_feat_s = allgather(torch.cat(batch_feat_s, dim=0), args)
            batch_feat_w = allgather(torch.cat(batch_feat_w, dim=0), args)
            batch_mask_t = allgather(torch.cat(batch_mask_t, dim=0), args)
            batch_feat_f = allgather(torch.cat(batch_feat_f, dim=0), args)
            batch_feat_p = allgather(torch.cat(batch_feat_p, dim=0), args)
            batch_mask_v = allgather(torch.cat(batch_mask_v, dim=0), args)
    toc1 = time.time()

    with torch.no_grad():
        sim_matrix = _run_on_single_gpu(model, batch_feat_s, batch_feat_w, batch_mask_t, batch_feat_f, batch_feat_p, batch_mask_v, args.split_batch)
        sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
    toc2 = time.time()
    if multi_sentence_:
        logger.info("before reshape, sim matrix size: {} x {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
        cut_off_points2len_ = [itm + 1 for itm in cut_off_points_]
        max_length = max([e_ - s_ for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_)])
        sim_matrix_new = []
        for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_):
            sim_matrix_new.append(np.concatenate((sim_matrix[s_:e_], np.full((max_length - e_ + s_, sim_matrix.shape[1]), -np.inf)), axis=0))
        sim_matrix = np.stack(tuple(sim_matrix_new), axis=0)
        logger.info("after reshape, sim matrix size: {} x {} x {}".format(sim_matrix.shape[0], sim_matrix.shape[1], sim_matrix.shape[2]))
        tv_metrics = tensor_text_to_video_metrics(sim_matrix)
        vt_metrics = compute_metrics(tensor_video_to_text_sim(sim_matrix))
        toc3 = time.time()
        logger.info("time profile: feat {:.1f}s match {:.5f}s metrics {:.5f}s".format(toc1 - tic, toc2 - toc1, toc3 - toc2))
        tv_metrics['RSum'] = tv_metrics['R1'] + tv_metrics['R5'] + tv_metrics['R10']
        logger.info("Text-to-Video: R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - R@Sum: {:.1f} - MdR: {:.1f} - MnR: {:.1f}".
                    format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['RSum'], tv_metrics['MR'], tv_metrics['MeanR']))
        vt_metrics['RSum'] = vt_metrics['R1'] + vt_metrics['R5'] + vt_metrics['R10']
        logger.info("Video-to-Text: R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - R@Sum: {:.1f} - MdR: {:.1f} - MnR: {:.1f}".
                    format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['RSum'], vt_metrics['MR'], vt_metrics['MeanR']))
        return tv_metrics['R1']
    else:
        logger.info("sim matrix size: {}, {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
        tv_metrics = compute_metrics(np_softmax(sim_matrix))
        vt_metrics = compute_metrics(np_softmax(sim_matrix.T))
        logger.info('Length-T: {}, Length-V:{}'.format(len(sim_matrix), len(sim_matrix[0])))
        logger.info('[end] compute_metrics')
        toc3 = time.time()
        logger.info("time profile: feat {:.1f}s match {:.5f}s metrics {:.5f}s".format(toc1 - tic, toc2 - toc1, toc3 - toc2))
        tv_metrics['RSum'] = tv_metrics['R1'] + tv_metrics['R5'] + tv_metrics['R10']
        logger.info("Text-to-Video: R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - R@Sum: {:.1f} - MdR: {:.1f} - MnR: {:.1f}".
                    format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['RSum'], tv_metrics['MR'], tv_metrics['MeanR']))
        vt_metrics['RSum'] = vt_metrics['R1'] + vt_metrics['R5'] + vt_metrics['R10']
        logger.info("Video-to-Text: R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - R@Sum: {:.1f} - MdR: {:.1f} - MnR: {:.1f}".
                    format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['RSum'], vt_metrics['MR'], vt_metrics['MeanR']))
        return tv_metrics['R1']

def main():
    global logger
    global best_score
    global meters

    meters = MetricLogger(delimiter="")
    args = get_args()
    args.output_dir = args.output_dir + "/" + datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    if not exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger('Model', args.output_dir, args.local_rank)

    args = set_seed_logger(args)

    model = build_model(args)

    test_dataloader, train_dataloader, train_sampler = build_dataloader(args)

    if args.do_train:
        tic = time.time()
        max_steps = len(train_dataloader) * args.epochs
        _max_steps = len(train_dataloader) * 5
        optimizer, scheduler, model = prep_optimizer(args, model, _max_steps, args.local_rank)

        best_score = 0.00001
        best_output_model_file = "None"
        global_step = 0
        logger.info("Model begins to training...")
        for epoch in range(args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            synchronize()
            if epoch == 0:
                torch.cuda.empty_cache()
                logger.info("Model zero-shot of text-to-video retrieval")
                R1 = eval_epoch(args, model, test_dataloader, args.device)
                logger.info("Zero-shot text-to-video retrieval of R1 is: {:.4f}".format(R1))

            torch.cuda.empty_cache()
            tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader,
                                               args.device, args.world_size, optimizer,
                                               scheduler, global_step, max_steps)
            torch.cuda.empty_cache()
            R1 = eval_epoch(args, model, test_dataloader, args.device)
            torch.cuda.empty_cache()
            synchronize()

            if args.local_rank == 0:
                output_model_file = save_model(epoch, args, model, type_name="")
                if best_score <= R1:
                    best_score = R1
                    best_output_model_file = output_model_file
                    torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(), 'best.pth')
                logger.info("The best model is: {}, the R1 is: {:.4f}".format(best_output_model_file, best_score))
            synchronize()
        toc = time.time() - tic
        training_time = time.strftime("%Hh %Mmin %Ss", time.gmtime(toc))
        logger.info("*" * 20 + '\n' + f'training finished with {training_time}' + "*" * 20 + '\n')

        model = model.module
        if args.local_rank == 0:
            model.load_state_dict(torch.load(best_output_model_file, map_location='cpu'), strict=False)
        if torch.cuda.is_available():
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)

        torch.cuda.empty_cache()
        eval_epoch(args, model, test_dataloader, args.device)
        synchronize()

    elif args.do_eval:
        eval_epoch(args, model, test_dataloader, args.device)
if __name__ == "__main__":
    main()