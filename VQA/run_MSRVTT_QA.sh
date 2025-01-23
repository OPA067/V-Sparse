CUDA_VISIBLE_DEVICES=0 \
python -m torch.distributed.launch \
--master_port 2502 \
--nproc_per_node=1 \
main_vqa.py \
--do_train \
--num_thread_reader=8 \
--epochs=5 \
--batch_size=8 \
--n_display=100 \
--train_csv MSRVTT/train.jsonl \
--val_csv MSRVTT/test.jsonl \
--data_path MSRVTT/train_ans2label.json \
--features_path MSRVTT/videos \
--lr 1e-4 \
--max_words 32 \
--max_frames 12 \
--batch_size_val 8 \
--datatype msrvtt \
--expand_msrvtt_sentences \
--feature_framerate 1 \
--coef_lr 1e-3 \
--freeze_layer_num 0  \
--slice_framepos 2 \
--loose_type \
--linear_patch 2d \
--output_dir experiments