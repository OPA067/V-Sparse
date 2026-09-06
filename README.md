<div align="center">

# From temporal-spatial visual semantic compression to coarse-to-fine interaction for text-video retrieval

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.0](https://img.shields.io/badge/pytorch-2.0-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Accepted by Neural Networks 🎉**

</div>

## 📋 Overview

To address the inherent imbalance in cross-modal matching, we propose a novel text-video retrieval model, named **V-Sparse**, which includes visual semantic compression for feature enhancement and coarse-to-fine alignment for feature interaction. First, we propose a text-guided Visual Semantic Compression (**VSC**) module, consisting of Temporal (**TVSC**) frame-level and Spatial (**SVSC**) patch-level  compression, aimed at reducing feature redundancy and providing precision support for coarse-to-fine interaction. Second, benefiting from visual semantic compression, we propose a novel Coarse-to-Fine granularity Interaction module (**CFI**), which aligns sentences with frames, sentences with patches, and words with patches from a unified joint feature encoding perspective. VSC and CFI jointly facilitate cross-modal text-video alignment from the perspectives of feature enhancement and feature interaction, greatly mitigating the inherent imbalance in modal pairing. We evaluate the performance of V-Sparse on six benchmark datasets and achieve state-of-the-art results in both long-video and short-text retrieval. Importantly, V-Sparse demonstrates the importance of feature compression in cross-modal interaction through extensive ablations and offers an effective intermediate pathway for modality interaction.

### Key Features

- 🚀 **Efficient Spatial Clustering**: Progressive compression of video patch tokens using DPC-KNN density peak clustering
- 🎯 **Multi-Level Interaction**: Three complementary interaction modes (global, local, fine-grained)
- 🔄 **Dual-Branch Architecture**: Low-resolution and high-resolution branches with KL alignment
- ⚡ **Optimized Training**: BertAdam optimizer with differentiated learning rates for CLIP and non-CLIP modules
- 🌐 **Distributed Training**: Full support for multi-GPU training with DDP

## 📣 Updates

- **[2025/01/18]**: We have released the complete training and testing code.
- **[2025/01/20]**: The paper has been submitted to the journal Neural Networks.
- **[2026/04/10]**: Paper accepted by the journal Neural Networks.

## ⚡ Framework

<div align="center">
  <img src="figures/framework.pdf" alt="V-Sparse Framework" width="800" style="border: none;"/>
</div>

### Architecture Overview

V-Sparse follows a four-stage pipeline for text-video retrieval:

1. **Multi-Granularity Feature Extraction**: Extract multi-granularity text and video features via CLIP encoder:
   - **Text**: sentence-level features `[a, d]` and word-level features `[a, w, d]`
   - **Video**: frame-level features `[b, f, d]` and patch-level features `[b, p, d]`

2. **Dual-Branch Processing**:
   - **Low-Resolution Branch**: Retain original coarse frame and patch features as-is
   - **Visual Feature Compact**: Select top-k query-relevant frames using diagonal-slice query-frame similarity
   - **High-Resolution Branch**: Apply ActionFlow module (3-layer PCM + Cross-Attention) on compacted patches

3. **Multi-Level Spatial Interaction**:
   - **Global (qs-vf)**: Query sentence ↔ video frame matching
   - **Local (qw-vf)**: Query word ↔ video frame matching
   - **Fine-grained (qw-vp)**: Query word ↔ video patch matching

4. **Loss Computation**:
   - **Symmetric Contrastive Loss**: Bidirectional cross-entropy over multi-level similarities
   - **KL Alignment Loss**: Aligns similarity distributions between low-resolution and high-resolution branches

### Training Objectives

- **Symmetric Contrastive Loss**: Bidirectional cross-entropy over multi-level similarities
- **KL Alignment Loss**: Aligns similarity distributions between low-resolution and high-resolution branches

### Dual-Branch Architecture

V-Sparse employs a dual-branch design for robust feature learning:

```
Input Video → CLIP Encoder → Frame/Patch Features
                                    ↓
                        ┌─────────────────────────────┐
                        │     Low-Resolution Branch    │
                        │   (Original frame/patch      │
                        │    features, no PCM)         │
                        └─────────────────────────────┘
                                    ↓
                        ┌─────────────────────────────┐
                        │  Visual Feature Compact      │
                        │  (Select top-k relevant      │
                        │   frames based on query)     │
                        └─────────────────────────────┘
                                    ↓
                        ┌─────────────────────────────┐
                        │     High-Resolution Branch   │
                        │  (3-layer PCM + Cross-Att)   │
                        │  ActionFlow Module           │
                        └─────────────────────────────┘
                                    ↓
                        ┌─────────────────────────────┐
                        │     KL Alignment Loss        │
                        │  (Align branch distributions)│
                        └─────────────────────────────┘
```

**ActionFlow Module**: The core innovation that progressively compresses video patch tokens:
1. **PCM Layer 1**: Compress patches from N to N×0.5 tokens
2. **Cross-Attention**: Text queries attend to compressed visual tokens
3. **PCM Layer 2**: Further compress to N×0.25 tokens
4. **Cross-Attention**: Text-guided refinement
5. **PCM Layer 3**: Final compression to N×0.125 tokens
6. **Cross-Attention**: Final text-visual interaction

This progressive compression reduces computational complexity while preserving spatial information through text-guided attention.

## 😍 Visualization

<div align="center">
  <img src="figures/motivation.pdf" alt="Motivation" width="600" style="border: none;"/>
  <p><em>Motivation: why sparse spatial clustering is essential for efficient text-video retrieval.</em></p>
</div>

<div align="center">
  <img src="figures/svsc.pdf" alt="SVSC" width="600" style="border: none;"/>
  <p><em>Sparse Video Spatial Clustering (SVSC) module details.</em></p>
</div>

<div align="center">
  <img src="figures/video_similarity.pdf" alt="Video Similarity" width="600" style="border: none;"/>
  <p><em>Video similarity analysis across different spatial granularities.</em></p>
</div>

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n vsparse python=3.10
conda activate vsparse

# Install PyTorch (choose the appropriate CUDA version)
pip install torch torchvision

# Install dependencies
pip install -r requirements.txt

# Install additional packages
pip install thop  # For model FLOPs computation
```

### 2. Data Preparation

Download the following datasets and organize them as shown:

```
data/
├── MSRVTT/
│   ├── anns/              # Annotation files
│   │   ├── MSRVTT_train.9000.csv
│   │   ├── MSRVTT_test.1000.csv
│   │   └── MSRVTT_data.json
│   └── videos/            # Video files (.mp4)
├── DiDeMo/
│   ├── anns/
│   └── videos/
└── Charades/
    ├── anns/
    └── videos/
```

### 3. Pretrained Weights

Place CLIP pretrained weights in the `models/` directory:
- `ViT-B/32` → `models/ViT-B-32.pt`
- `ViT-B/16` → `models/ViT-B-16.pt`

These can be downloaded from the [official CLIP repository](https://github.com/openai/CLIP).

### 4. Video Preprocessing (Optional)

If you need to preprocess videos for faster loading:

```bash
python preprocess/compress_video.py \
    --input_root /path/to/original/videos \
    --output_root /path/to/compressed/videos
```

This will:
- Resize videos to 224px resolution
- Compress to 3 FPS for faster loading
- Use multiprocessing for parallel processing

**Preprocessing Benefits:**
- Reduces video file sizes by 80-90%
- Speeds up video loading by 3-5x
- Maintains visual quality for retrieval tasks
- Uses all available CPU cores for parallel processing

### 5. Training

```bash
# Train on MSRVTT (single GPU)
bash script/run_MSRVTT.sh

# Train on DiDeMo
bash script/run_DiDeMo.sh

# Train on Charades
bash script/run_Charades.sh
```

**Multi-GPU Training** (e.g., 4 GPUs):
```bash
# Edit script/run_MSRVTT.sh and set GPU="0,1,2,3"
GPU="0,1,2,3" bash script/run_MSRVTT.sh
```

**Training Process:**
1. **Zero-shot Evaluation**: Evaluate CLIP baseline before training
2. **Epoch Training**: 5 epochs with BertAdam optimizer
3. **Evaluation**: After each epoch, evaluate on test set
4. **Checkpointing**: Save best model based on R@1 score
5. **Final Evaluation**: Reload best checkpoint for final results

**Optimizer Configuration:**
- **BertAdam**: Adam with bias correction and warmup
- **4 Parameter Groups**: Differentiated learning rates for CLIP and non-CLIP modules
- **Warmup**: 10% of total steps with linear warmup
- **Schedule**: Warmup cosine annealing

**Command Line Arguments:**
```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --master_port 2502 --nproc_per_node=1 main_retrieval.py \
    --do_train 1 \
    --do_eval 1 \
    --datatype msrvtt \
    --anno_path MSRVTT \
    --video_path MSRVTT/videos \
    --base_encoder ViT-B/32 \
    --agg_module seqTransf \
    --num_hidden_layers 4 \
    --max_words 24 \
    --max_frames 12 \
    --lr 1e-4 \
    --coef_lr 1e-3 \
    --epochs 5 \
    --batch_size 32 \
    --output_dir experiments/MSRVTT
```

### 6. Evaluation

```bash
# Evaluate on MSRVTT
bash script/run_test.sh

# Or run evaluation directly
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --master_port 2502 --nproc_per_node=1 main_retrieval.py \
    --do_eval 1 \
    --init_model path/to/best.pth \
    --datatype msrvtt \
    --anno_path MSRVTT \
    --video_path MSRVTT/videos \
    --output_dir experiments/MSRVTT
```

**Evaluation Output:**
The evaluation will output the following metrics:
- **R@1, R@5, R@10**: Recall at different ranks
- **MdR**: Median Rank
- **MnR**: Mean Rank
- **RSum**: Sum of R@1 + R@5 + R@10

**Evaluation Process:**
1. Load the trained model checkpoint
2. Extract features for all test queries and videos
3. Compute similarity scores
4. Rank videos for each query (and vice versa)
5. Calculate retrieval metrics

## 🗂️ Project Structure

```
V-Sparse/
├── main_retrieval.py              # Main entry point for training & evaluation
├── models/
│   ├── modeling.py                # Core Model: CLIP encoder + PCM + spatial interaction
│   ├── cluster_.py                # Progressive Clustering Module (PCM) & cross-attention
│   ├── module_clip.py             # CLIP vision/language encoder
│   ├── module_cross.py            # Transformer blocks for cross-modal interaction
│   ├── module_transformer.py      # Transformer utilities
│   ├── until_module.py            # Utility modules (LayerNorm, AllGather, CrossEn, KL)
│   ├── optimization.py            # BertAdam optimizer
│   └── tokenization_clip.py       # CLIP text tokenizer
├── dataloaders/
│   ├── data_dataloaders.py        # DataLoader registry
│   ├── dataloader_msrvtt_retrieval.py   # MSRVTT data loader
│   ├── dataloader_didemo_retrieval.py   # DiDeMo data loader
│   ├── dataloader_charades_retrieval.py # Charades data loader
│   ├── dataloader_retrieval.py    # Base retrieval dataset class
│   ├── video_transforms.py        # Video augmentation transforms
│   ├── rand_augment.py            # RandAugment implementation
│   └── random_erasing.py          # Random erasing augmentation
├── utils/
│   ├── metrics.py                 # Retrieval metrics (R@1/5/10, MdR, MnR)
│   ├── metric_logger.py           # Training metric logger
│   ├── logger.py                  # Logging utilities
│   └── comm.py                    # Distributed communication utilities
├── script/
│   ├── run_MSRVTT.sh              # Training script for MSRVTT
│   ├── run_DiDeMo.sh              # Training script for DiDeMo
│   ├── run_Charades.sh            # Training script for Charades
│   └── run_test.sh                # Evaluation script
├── preprocess/
│   └── compress_video.py          # Video preprocessing utility
├── experiments/                   # Training logs and checkpoints
│   └── MSRVTT/                    # MSRVTT experiment results
├── figures/                       # Paper figures (PNG format)
├── docs/                          # Documentation and paper
└── requirements.txt               # Python dependencies
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--datatype` | `msrvtt` | Dataset: `msrvtt` / `charades` / `didemo` |
| `--base_encoder` | `ViT-B/32` | CLIP backbone: `ViT-B/32` / `ViT-B/16` |
| `--agg_module` | `seqTransf` | Temporal aggregation: `None` / `seqLSTM` / `seqTransf` |
| `--num_hidden_layers` | `4` | Number of Transformer layers in the video branch |
| `--max_words` | `24` | Maximum text tokens per query |
| `--max_frames` | `12` | Maximum sampled video frames |
| `--save_frames` | `6` | Frames retained after compact (`max_frames // 2`) |
| `--epochs` | `5` | Total training epochs |
| `--lr` | `1e-4` | Learning rate for non-CLIP modules |
| `--coef_lr` | `1e-3` | LR coefficient for the CLIP branch |
| `--batch_size` | `32` | Training batch size |
| `--batch_size_val` | `32` | Evaluation batch size |
| `--split_batch` | `32` | Split batch for evaluation to prevent OOM |
| `--workers` | `4` | Number of data loading workers |
| `--seed` | `42` | Random seed for reproducibility |

## 🔧 Key Components

### 1. Progressive Clustering Module (PCM)

The PCM module performs dynamic token compression with the following steps:

1. **Token Convolution**: 1D convolution for local feature transformation
2. **Importance Scoring**: Linear layer predicts per-token importance score `[B, N, 1]`
3. **Weight Conversion**: Convert scores to non-negative aggregation weights via exponential
4. **DPC-KNN Clustering**: Density peak clustering with K-nearest neighbors
5. **Canonical Merge**: Max-reduction merging within clusters (HV communication semantics)

```python
# Example PCM usage
pcm = PCM(sample_ratio=0.5, k=3)
downsampled_dict = pcm(token_dict)
```

### 2. Cross-Attention Block (Cross_Att_Block_Patch)

Text-guided visual token attention with mismatched dimensions:

- **Query**: Text word features `[B, Nq, dim_q]` (e.g., `[B, 24, 512]`)
- **Key/Value**: Video patch features `[B, Nkv, dim_kv]` (e.g., `[B, 588, 512]`)
- **Special Case**: If Nq == 1 (sentence-level feature), it is broadcast to match Nkv
- **Token Score Bias**: Uses PCM's importance score as confidence bias in attention logits
- **Mask Handling**: Both query and key/value masks are properly applied

```python
# Example cross-attention usage
cross_att = Cross_Att_Block_Patch(dim=512)
kv_dict = cross_att(qw_token_dict, vp_token_dict)
```

### 3. Multi-Level Similarity Computation

Three complementary interaction modes with symmetric attention:

- **qs-vf (Global)**: Query sentence ↔ video frame matching
  - Direction 1: qs → vf (max over frames)
  - Direction 2: vf → qs (weighted aggregation)
  
- **qw-vf (Local)**: Query word ↔ video frame matching
  - Direction 1: qw → vf (max over frames, then word-weighted)
  - Direction 2: vf → qw (max over words, then frame-weighted)
  
- **qw-vp (Fine-grained)**: Query word ↔ video patch matching
  - Direction 1: qw → vp (max over patches, then word-weighted)
  - Direction 2: vp → qw (max over words, then patch-weighted)

Each mode uses learnable aggregation weights with softmax normalization.

### 4. Visual Feature Compact

Selects the most query-relevant frames for high-resolution branch:

1. **Query-Frame Similarity**: Compute diagonal-slice similarity between query sentences and video frames
2. **Softmax Normalization**: Normalize similarity scores over frames
3. **Top-k Selection**: Select `save_frames = max_frames // 2` most relevant frames
4. **Patch Gathering**: Gather corresponding patches for the selected frames

### 5. KL Alignment Loss

Aligns similarity distributions between low-resolution and high-resolution branches:

```python
# Bidirectional KL divergence
loss_kl = (KL(sims_l, sims_h) + KL(sims_l.T, sims_h.T) +
           KL(sims_h, sims_l) + KL(sims_h.T, sims_l.T)) / 4.0
```

This ensures both branches produce consistent similarity distributions.

## 📊 Results

V-Sparse achieves competitive text-video retrieval performance on standard benchmarks.

### MSRVTT (Single GPU, 5 epochs)

| Metric | Zero-shot | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 | Epoch 5 (Best) |
|--------|-----------|---------|---------|---------|---------|----------------|
| T→V R@1 | 34.4 | 46.3 | 48.6 | 49.2 | 50.4 | **50.9** |
| T→V R@5 | 59.1 | 75.1 | 75.7 | 76.8 | 76.2 | **75.9** |
| T→V R@10 | 69.5 | 83.9 | 85.1 | 86.2 | 85.1 | **85.5** |
| V→T R@1 | 35.6 | 48.2 | 47.8 | 51.2 | 49.9 | **50.7** |
| V→T R@5 | 60.3 | 74.9 | 75.3 | 77.1 | 76.4 | **76.6** |
| V→T R@10 | 68.9 | 85.0 | 83.5 | 85.1 | 84.9 | **84.5** |

**Training Details:**
- Model: 171.55M parameters (169.19M trainable)
- Training time: ~6.5 hours (single GPU)
- GPU memory: ~10.57 GB
- Optimizer: BertAdam with warmup cosine annealing
- Batch size: 32

### Dataset Statistics

| Dataset | Train Videos | Test Videos | Captions | Task |
|---------|--------------|-------------|----------|------|
| MSRVTT | 9,000 | 1,000 | 200,000 | Video-to-Text Retrieval |
| DiDeMo | 8,543 | 1,045 | 42,729 | Temporal Video Grounding |
| Charades | 12,468 | 1,841 | 53,397 | Activity Recognition |

### Evaluation Metrics

- **R@K**: Recall at K (K=1, 5, 10) - percentage of queries where correct match is in top-K
- **MdR**: Median Rank - median rank of the correct match
- **MnR**: Mean Rank - average rank of the correct match
- **RSum**: Sum of R@1 + R@5 + R@10

For detailed results on DiDeMo and Charades, please refer to our paper.

## 🔬 Technical Details

### Model Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_encoder` | `ViT-B/32` | CLIP backbone variant |
| `agg_module` | `seqTransf` | Temporal aggregation module (`None` / `seqLSTM` / `seqTransf`) |
| `num_hidden_layers` | `4` | Transformer layers in video branch |
| `max_words` | `24` | Maximum text tokens per query |
| `max_frames` | `12` | Maximum sampled video frames |
| `save_frames` | `6` | Frames retained after compact (`max_frames // 2`) |
| `sample_ratio` | `0.5` | PCM token compression ratio (per layer) |
| `k` | `3` | KNN neighbors for DPC-KNN |

### Training Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | `1e-4` | Learning rate for non-CLIP modules |
| `coef_lr` | `1e-3` | LR coefficient for CLIP branch |
| `weight_decay` | `0.2` | Weight decay coefficient |
| `warmup_proportion` | `0.1` | Warmup proportion |
| `epochs` | `5` | Total training epochs |
| `batch_size` | `32` | Training batch size |

### Loss Function

The total loss combines three components:

1. **High-resolution contrastive loss**: `loss_h`
   - Computed on features from high-resolution branch (ActionFlow output)
   - Symmetric cross-entropy over qs-vf, qw-vf, qw-vp similarities

2. **Low-resolution contrastive loss**: `loss_l`
   - Computed on original (unprocessed) frame/patch features
   - Same three similarity modes as high-resolution branch

3. **KL alignment loss**: `loss_kl`
   - Bidirectional KL divergence between low-resolution and high-resolution similarity distributions
   - Aligns both row and column directions

```
total_loss = loss_h + loss_l + loss_kl
```

### Feature Dimensions

| Feature | Shape | Description |
|---------|-------|-------------|
| `qs_feat` | `[a, d]` | Query sentence features (CLIP [CLS] token) |
| `qw_feat` | `[a, w, d]` | Query word features (CLIP hidden states) |
| `vf_feat` | `[b, f, d]` | Video frame features (CLIP [CLS] tokens) |
| `vp_feat` | `[b, p, d]` | Video patch features (CLIP hidden states) |
| `vp_feat_h` | `[b, h, d]` | High-resolution frame features (after compact) |
| `vp_feat_` | `[b, p', d]` | ActionFlow output (3-layer concatenated patches) |

Where: `a` = batch size, `w` = max words, `f` = max frames, `p` = frames × patches per frame (e.g., 12×49=588), `d` = embed dim (512)

## ⚙️ Computational Efficiency

V-Sparse achieves significant computational savings through progressive spatial clustering:

### Model Efficiency

| Metric | Value |
|--------|-------|
| Total Parameters | 171.55M |
| Trainable Parameters | 169.19M |
| GPU Memory Usage | ~10.57 GB |
| Training Time (MSRVTT) | ~6.5 hours (single GPU) |
| Inference Speed | ~30s per 1000 samples |

### Progressive Compression Benefits

The ActionFlow module reduces computational complexity through 3-layer progressive compression:

```
Original patches: 12 frames × 49 patches/frame = 588 patches
After PCM Layer 1: 588 × 0.5 = 294 patches
After PCM Layer 2: 294 × 0.5 = 147 patches
After PCM Layer 3: 147 × 0.5 = 74 patches (final)
```

This **87.5% reduction** in patch tokens significantly reduces attention computation while preserving spatial information through text-guided clustering.

### Token Compression Pipeline

Each PCM layer performs:
1. **Token Convolution**: 1D convolution for local feature transformation
2. **Importance Scoring**: Linear layer predicts per-token importance
3. **DPC-KNN Clustering**: Density peak clustering with K-nearest neighbors
4. **Canonical Merge**: Max-reduction merging within clusters (HV communication semantics)

### Sparse vs Dense Computation

The `token2map` and `map2token` functions automatically choose between sparse and dense matrix multiplication based on computational complexity:
- **Sparse path**: When `N_init < N * H * W` (more efficient for small token counts)
- **Dense path**: When `N_init >= N * H * W` (more efficient for large token counts)

This adaptive strategy ensures optimal performance across different token configurations.

## 🎗️ Acknowledgments

This project is built upon [CLIP](https://github.com/openai/CLIP) and several excellent open-source projects. We thank the authors for releasing their code.

**Key Dependencies:**
- [CLIP](https://github.com/openai/CLIP): Contrastive Language-Image Pre-training
- [PyTorch](https://pytorch.org/): Deep learning framework
- [transformers](https://github.com/huggingface/transformers): Hugging Face transformers
- [decord](https://github.com/dmlc/decord): Video loading library
- [timm](https://github.com/rwightman/pytorch-image-models): PyTorch Image Models

## 📚 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{liu2026vsparse,
  title   = {V-Sparse: From temporal-spatial visual semantic compression to coarse-to-fine interaction for text-video retrieval},
  author  = {Liu, Xin and Yin, Shibai and Wang, Jun and Li, Wei and Wang, Xingyang and Zhu, Jiaxin and Shen, Yubing and Yang, Yee-Hong},
  journal = {Neural Networks},
  volume  = {201},
  pages   = {108982},
  year    = {2026},
  doi     = {10.1016/j.neunet.2026.108982}
}
```

## 📄 License

This project is for research purposes only. Please refer to the [LICENSE](LICENSE) file for details.

**Usage Rights:**
- ✅ Academic research and education
- ✅ Non-commercial purposes
- ✅ Modification and adaptation
- ❌ Commercial use without permission
- ❌ Redistribution without attribution

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues and pull requests.

**How to Contribute:**
1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

**Contribution Guidelines:**
- Follow the existing code style
- Add comments for complex logic
- Update documentation if needed
- Test your changes before submitting

## 🛠️ Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size
--batch_size 16
--batch_size_val 16

# Or reduce max_frames
--max_frames 8

# Or reduce split_batch for evaluation
--split_batch 16
```

**2. Video Loading Errors**
```bash
# Ensure videos are in correct format
# Check if video files exist
ls /path/to/videos/*.mp4

# Use video preprocessing for faster loading
python preprocess/compress_video.py \
    --input_root /path/to/original \
    --output_root /path/to/compressed
```

**3. Distributed Training Issues**
```bash
# Check GPU availability
nvidia-smi

# Ensure master_port is available
--master_port 2502

# Use gloo backend for compatibility
torch.distributed.init_process_group(backend="gloo")
```

**4. Memory Issues During Evaluation**
```bash
# Reduce split_batch to prevent OOM
--split_batch 16

# Or use smaller batch_size_val
--batch_size_val 16
```

**5. Import Errors**
```bash
# Install missing dependencies
pip install -r requirements.txt
pip install thop

# Check Python version
python --version  # Should be 3.10+

# Check PyTorch installation
python -c "import torch; print(torch.__version__)"
```

**6. Model Loading Errors**
```bash
# Ensure CLIP weights are in correct location
ls models/ViT-B-32.pt
ls models/ViT-B-16.pt

# Download from official CLIP repository if missing
# https://github.com/openai/CLIP
```

### Performance Tips

1. **Use SSD storage** for faster video loading
2. **Increase workers** for better I/O parallelism: `--workers 8`
3. **Use mixed precision** for faster training (if supported)
4. **Monitor GPU memory** during training to optimize batch size
5. **Use video preprocessing** to compress videos for faster loading

### Reproducibility

For reproducible results:
```bash
--seed 42  # Fixed random seed
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

## 📧 Contact

For questions or feedback, please open an issue on GitHub.
