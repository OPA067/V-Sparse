CUDA_VISIBLE_DEVICES=0 \
python -m torch.distributed.launch \
--master_port 2502 \
--nproc_per_node=1 \
main_retrieval.py \
--do_train 1 \
--workers 8 \
--n_display 100 \
--epochs 5 \
--lr 1e-4 \
--coef_lr 1e-3 \
--batch_size 8 \
--batch_size_val 8 \
--anno_path LSMDC \
--video_path LSMDC \
--datatype lsmdc \
--max_words 24 \
--max_frames 12 \
--video_framerate 1 \
--split_batch 9 \
--output_dir experiments/LSMDC

