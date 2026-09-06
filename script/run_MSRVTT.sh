# ============================================================
# 1. Specify GPU(s) here directly. Examples:
#    GPU="0"       -> Single GPU
#    GPU="0,1"     -> 2 GPUs (DDP)
#    GPU="0,1,2,3" -> 4 GPUs (DDP)
# ============================================================
GPU="2"

# Derive number of processes from the GPU list automatically
NPROC=$(echo "$GPU" | tr ',' '\n' | wc -l)

CUDA_VISIBLE_DEVICES=$GPU \
python -m torch.distributed.launch \
--master_port 2502 \
--nproc_per_node=$NPROC \
main_retrieval.py \
--do_train 1 \
--workers 8 \
--n_display 100 \
--epochs 5 \
--lr 1e-4 \
--coef_lr 1e-3 \
--batch_size 32 \
--batch_size_val 32 \
--split_batch 32 \
--anno_path MSRVTT \
--video_path MSRVTT/videos \
--datatype msrvtt \
--max_words 32 \
--max_frames 12 \
--video_framerate 1 \
--output_dir experiments/MSRVTT \
# --init_model experiments/MSRVTT/2025-10-19_12:07:07/pytorch_model.bin.0 
