CUDA_VISIBLE_DEVICES=0 \
python -m torch.distributed.launch \
--master_port 2502 \
--nproc_per_node=1 \
main_retrieval.py \
--do_train 1 \
--workers 8 \
--n_display 1 \
--epochs 5 \
--lr 1e-4 \
--coef_lr 1e-3 \
--batch_size 32 \
--batch_size_val 32 \
--anno_path Charades \
--video_path Charades/videos \
--datatype charades \
--max_words 24 \
--max_frames 12 \
--video_framerate 1 \
--split_batch 23 \
--output_dir experiments/Charades

