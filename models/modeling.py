import os
from collections import OrderedDict
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .cluster_ import PCM, Cross_Att_Block_Patch
from .module_clip import CLIP, convert_weights, _PT_NAME
from .module_cross import Transformer as TransformerClip
from .until_module import LayerNorm, AllGather, AllGather2, CrossEn, KL

# Distributed training ops for cross-GPU feature aggregation
allgather = AllGather.apply
allgather2 = AllGather2.apply


class ResidualLinear(nn.Module):
    """Residual linear block: x + (Linear -> ReLU -> Linear)(x)."""

    def __init__(self, d_int: int):
        super(ResidualLinear, self).__init__()

        self.fc_relu = nn.Sequential(
            nn.Linear(d_int, d_int),
            nn.ReLU(inplace=True),
            nn.Linear(d_int, d_int),
        )

    def forward(self, x):
        x = x + self.fc_relu(x)
        return x


class Model(nn.Module):
    """
    Core model for TSI-TVR (Temporal-Spatial Interaction for Text-Video Retrieval).

    This model extracts multi-granularity text and video features via CLIP, applies
    progressive spatial clustering (ActionFlow) via PCM + Cross_Att_Block_Patch on
    video patch tokens, and computes text-to-video similarity through multi-level
    spatial interaction matching.

    Architecture:
        1. CLIP Encoder: extracts sentence/word-level text features and frame/patch-level
           visual features;
        2. ActionFlow (PCM + Cross_Att_Block_Patch): progressive spatial clustering on
           video patch tokens, guided by the query sentence vector as attention query;
        3. Spatial Interaction: query-sentence vs video-frame (global, qs-vf),
           query-word vs video-frame (local, qw-vf), and query-word vs video-patch
           (local, qw-vp);
        4. Symmetric Contrastive Loss: bidirectional cross-entropy over qs-vf, qw-vf
           and qw-vp;
        5. KL Alignment Loss: aligns similarity distributions between the low-resolution
           and high-resolution branches.
    """

    def __init__(self, config):
        """
        Initialize all model modules.

        Key config attributes:
            - interaction: interaction type string.
            - agg_module: video frame aggregation mode, one of 'meanP' (mean pooling),
              'seqLSTM', or 'seqTransf'.
            - base_encoder: CLIP backbone variant, e.g., "ViT-B/32".
            - num_hidden_layers: number of Transformer layers for seqTransf agg_module.
            - max_words: maximum number of words per text sequence.
            - max_frames: maximum number of video frames.
            - save_frames: number of top-relevance frames retained after compact (default max_frames // 2).
        """
        super(Model, self).__init__()

        self.config = config

        self.interaction = config.interaction
        self.agg_module = getattr(config, 'agg_module', 'meanP')
        backbone = getattr(config, 'base_encoder', "ViT-B/32")

        assert backbone in _PT_NAME
        # Pretrained CLIP weights are searched in ./models/, then in the project root
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _PT_NAME[backbone])
        if not os.path.exists(model_path):
            model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), _PT_NAME[backbone])
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
        try:
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = model.state_dict()
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        # Derive model hyperparameters from pretrained CLIP weights automatically
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)  # e.g. 7 for ViT-B/32
        image_resolution = vision_patch_size * grid_size  # e.g. 224 for a 32px patch size

        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

        # Initialize CLIP backbone (vision encoder + text encoder)
        self.clip = CLIP(embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size,
                         context_length, vocab_size, transformer_width, transformer_heads, transformer_layers)

        if torch.cuda.is_available():
            convert_weights(self.clip)

        # Cross-modal Transformer configuration for seqTransf frame aggregation
        cross_config = SimpleNamespace(**{
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 512,
            "initializer_range": 0.02,
            "intermediate_size": 2048,
            "max_position_embeddings": 128,
            "num_attention_heads": 8,
            "num_hidden_layers": 4,
            "vocab_size": 512,
            "soft_t": 0.07,
        })
        cross_config.max_position_embeddings = context_length
        cross_config.hidden_size = transformer_width
        self.cross_config = cross_config

        # Optional temporal aggregation module for video frames
        if self.agg_module in ["seqLSTM", "seqTransf"]:
            # Frame-level positional embeddings for temporal ordering
            self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings,
                                                          cross_config.hidden_size)
            if self.agg_module == "seqTransf":
                # Multi-head Transformer for temporal modeling across frame sequences
                self.transformerClip = TransformerClip(width=transformer_width,
                                                       layers=config.num_hidden_layers,
                                                       heads=transformer_heads)
            if self.agg_module == "seqLSTM":
                # Unidirectional LSTM for temporal modeling across frame sequences
                self.lstm_visual = nn.LSTM(input_size=cross_config.hidden_size,
                                           hidden_size=cross_config.hidden_size,
                                           batch_first=True, bidirectional=False, num_layers=1)

        # Loss functions
        self.loss_fct = CrossEn(config)  # Symmetric contrastive loss (cross-entropy)
        self.loss_kl = KL(config)        # KL divergence loss for branch alignment

        self.apply(self.init_weights)  # Init new modules before loading pretrained weights
        self.clip.load_state_dict(state_dict, strict=False)

        # Re-derive embed_dim because the cross_config may have overridden the above derivation.
        # ActionFlow and aggregation modules below use this fresh embed_dim.
        embed_dim = state_dict["text_projection"].shape[1]
        self.max_words = config.max_words
        self.max_frames = config.max_frames
        self.save_frames = config.max_frames // 2

        # ActionFlow: three-layer progressive spatial clustering (sampling ratio 0.5 each layer)
        sr_vp = [0.5, 0.5, 0.5]
        self.v_pcm_p_1 = PCM(sample_ratio=sr_vp[0], k=3)
        self.v_cross_att_block_p_1 = Cross_Att_Block_Patch(dim=embed_dim)
        self.v_pcm_p_2 = PCM(sample_ratio=sr_vp[1], k=3)
        self.v_cross_att_block_p_2 = Cross_Att_Block_Patch(dim=embed_dim)
        self.v_pcm_p_3 = PCM(sample_ratio=sr_vp[2], k=3)
        self.v_cross_att_block_p_3 = Cross_Att_Block_Patch(dim=embed_dim)

        # Learnable feature aggregation weight networks (one per granularity pair)
        self.qs_feat_w = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.ReLU(), nn.Linear(embed_dim * 2, 1))
        self.qw_feat_w = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.ReLU(), nn.Linear(embed_dim * 2, 1))
        self.vf_feat_w = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.ReLU(), nn.Linear(embed_dim * 2, 1))
        self.vp_feat_w = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.ReLU(), nn.Linear(embed_dim * 2, 1))

        self.sims_weights_h = nn.Parameter(torch.ones(3))
        self.sims_weights_l = nn.Parameter(torch.ones(3))

        # Warm-start trick: copy CLIP pretrained positional embeddings / Transformer layers
        # into the newly created agg_module to improve training stability at early steps.
        new_state_dict = OrderedDict()

        if self.agg_module in ["seqLSTM", "seqTransf"]:
            contain_frame_position = False
            for key in state_dict.keys():
                if key.find("frame_position_embeddings") > -1:
                    contain_frame_position = True
                    break
            if contain_frame_position is False:
                for key, val in state_dict.items():
                    if key == "positional_embedding":
                        new_state_dict["frame_position_embeddings.weight"] = val.clone()
                        continue
                    if self.agg_module in ["seqTransf"] and key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])
                        if num_layer < config.num_hidden_layers:
                            new_state_dict[key.replace("transformer.", "transformerClip.")] = val.clone()
                            continue
        self.load_state_dict(new_state_dict, strict=False)

    def forward(self, query, query_word_mask, video, video_frame_mask, idx=None, global_step=0):
        """
        Forward pass for training.

        Pipeline:
            1. Feature extraction: CLIP encodes query into sentence-level [a,d]
               and word-level [a,w,d] features; encodes video into frame-level
               [b,f,d] and patch-level [b,p,d] features.
            2. Low-resolution branch (_l): retain original coarse frame and
               patch features as-is; compute qs-vf, qw-vf and qw-vp similarities
               on raw features, fuse with learnable weights, then compute
               contrastive loss.
            3. Visual feature compact: diagonal-slice query-frame similarity to
               select save_frames most query-relevant frames per video, gather
               corresponding patches into high-resolution features [b,h,d] and
               [b,h*pn,d].
            4. High-resolution branch (_h): 3-layer PCM + Cross_Att_Block_Patch
               on compacted patches (PCM compresses first, then Q=text attends
               over KV=visual). Outputs from all three layers are concatenated,
               then qs-vf, qw-vf and qw-vp similarities are computed and fused
               with learnable weights.
            5. KL alignment: minimize bidirectional KL divergence between
               low-resolution and high-resolution fused similarity distributions.
            6. Total loss: sum of _h contrastive loss, _l contrastive loss,
               and KL alignment loss.

        Args:
            query (Tensor): query text token IDs, shape [a, L].
            query_word_mask (Tensor): query text padding mask, [a, L].
            video (Tensor): video frames, shape [b, n_v, channel, h, w] or
                [b, pair, bs, ts, channel, h, w] depending on data loader.
            video_frame_mask (Tensor): video frame mask, [b, n_v].
            idx (Tensor, optional): sample indices (unused, reserved).
            global_step (int, optional): current training step (unused, reserved).

        Returns:
            Tensor: total training loss in training mode; None otherwise.
        """
        # Flatten local batch dimensions for uniform processing
        query = query.reshape(-1, query.shape[-1])
        query_word_mask = query_word_mask.reshape(-1, query_word_mask.shape[-1])

        video = torch.as_tensor(video).float()
        video_frame_mask = video_frame_mask.reshape(-1, video_frame_mask.shape[-1])

        if len(video.size()) == 5:
            b, n_v, d, h, w = video.shape
            video = video.reshape(b * n_v, d, h, w)
        else:
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.reshape(b * pair * bs * ts, channel, h, w)

        # ========== Step 1: multi-granularity feature extraction ==========
        qs_feat, qw_feat = self.get_text_feat(query, query_word_mask)   # [a, d], [a, w, d]
        vf_feat, vp_feat = self.get_video_feat(video, video_frame_mask) # [b, f, d], [b, p, d]

        # Build granularity masks (qs_mask always valid since query is a single sentence)
        qs_mask = qs_feat.new_ones(qs_feat.size(0), 1)          # [a, 1]
        qw_mask = query_word_mask                               # [a, w]
        vf_mask = video_frame_mask                              # [b, f]
        patches_per_frame = vp_feat.size(1) // vf_feat.size(1) if vf_feat.size(1) > 0 else 0
        vp_mask = video_frame_mask.unsqueeze(-1).expand(-1, -1, patches_per_frame).reshape(video_frame_mask.size(0), -1)
  
        # Ensure memory-contiguous layout for efficient computation
        qs_feat, qw_feat = qs_feat.contiguous(), qw_feat.contiguous()
        qs_mask, qw_mask = qs_mask.contiguous(), qw_mask.contiguous()
        vf_feat, vp_feat = vf_feat.contiguous(), vp_feat.contiguous()
        vf_mask, vp_mask = vf_mask.contiguous(), vp_mask.contiguous()

        # Gather features across all GPUs for distributed contrastive training
        qs_feat, qw_feat, qs_mask, qw_mask = [allgather(x, self.config) for x in [qs_feat, qw_feat, qs_mask, qw_mask]]
        vf_feat, vp_feat, vf_mask, vp_mask = [allgather(x, self.config) for x in [vf_feat, vp_feat, vf_mask, vp_mask]]
        torch.distributed.barrier()  # Synchronize all GPUs before computing losses

        # Dimension aliases for readability
        a, s, w = qs_feat.size(0), 1, qw_feat.size(1)
        b, f, p = vf_feat.size(0), vf_feat.size(1), vp_feat.size(1)

        # CLIP-learnable temperature parameter for scaling similarity logits
        logit_scale = self.clip.logit_scale.exp()

        # ========== Step 2: low-resolution branch (_l) ==========
        # Retain original coarse frame and patch features as-is.
        vf_feat_l, vf_mask_l = vf_feat, vf_mask
        vp_feat_l, vp_mask_l = vp_feat, vp_mask

        # ========== Step 3: visual feature compact ==========
        # Diagonal-slice query-frame similarity: sims_qs_vf[i,i,:] gives the
        # i-th query's similarity to all frames of its paired video. Softmax
        # over frames, then top-k selects the save_frames most relevant frames.
        sims_qs_vf = torch.einsum("ad,bfd->abf", [self.norm(qs_feat), self.norm(vf_feat)])
        sims_per_video = sims_qs_vf[torch.arange(b), torch.arange(b)]   # [b, f] (diagonal slice)
        sims_per_video = torch.softmax(sims_per_video, dim=-1)          # [b, f]
        # Select top-h high-relevance frames and their corresponding patches
        _, vf_max_idx = torch.topk(sims_per_video, k=self.save_frames, dim=-1, largest=True)
        vf_feat_h = vf_feat[torch.arange(b)[:, None], vf_max_idx, :]      # [b, h, d]
        vf_mask_h = vf_mask[torch.arange(b)[:, None], vf_max_idx]         # [b, h]
        # Map selected frame indices to patch indices: each frame has patch_num patches
        patch_num = p // f
        patch_offset = torch.arange(patch_num, device=vp_feat.device)[None, :]  # [1, patch_num]
        vp_idx_h = (vf_max_idx.unsqueeze(-1) * patch_num + patch_offset).reshape(b, -1)
        vp_feat_h = vp_feat[torch.arange(b)[:, None], vp_idx_h, :]        # [b, h*patch_num, d]
        vp_mask_h = vp_mask[torch.arange(b)[:, None], vp_idx_h]           # [b, h*patch_num]

        # Build token_dict for high-resolution branch (compact patches)
        vp_idx_token = torch.arange(vp_feat_h.size(1), device=vp_feat_h.device)[None, :].repeat(vp_feat_h.size(0), 1)
        vp_agg_weight = vp_feat_h.new_ones(vp_feat_h.size(0), vp_feat_h.size(1), 1)
        vp_token_dict = {
            'x': vp_feat_h,
            'token_num': vp_feat_h.size(1),
            'idx_token': vp_idx_token,
            'agg_weight': vp_agg_weight,
            'mask': vp_mask_h.detach()
        }
        # Build text token_dict for cross-attention (Q from text, KV from visual)
        qw_idx_token = torch.arange(qw_feat.size(1), device=qw_feat.device)[None, :].repeat(qw_feat.size(0), 1)
        qw_agg_weight = qs_feat.new_ones(qw_feat.size(0), qw_feat.size(1), 1)
        qw_token_dict = {'x': qw_feat,
                        'token_num': qw_feat.size(1),
                        'idx_token': qw_idx_token,
                        'agg_weight': qw_agg_weight,
                        'mask': qw_mask.detach()}

        # ActionFlow: each layer compresses patches with PCM, then
        # Cross_Att_Block_Patch attends with Q=text over KV=visual. 
        vp_feat_ = []
        vp_token_dict = self.v_cross_att_block_p_1(qw_token_dict, self.v_pcm_p_1(vp_token_dict))
        vp_feat_.append(vp_token_dict['x'])
        vp_token_dict = self.v_cross_att_block_p_2(qw_token_dict, self.v_pcm_p_2(vp_token_dict))
        vp_feat_.append(vp_token_dict['x'])
        vp_token_dict = self.v_cross_att_block_p_3(qw_token_dict, self.v_pcm_p_3(vp_token_dict))
        vp_feat_.append(vp_token_dict['x'])
        vp_feat_h = torch.cat(vp_feat_, dim=1)                                  # [b, p', d]
        # PCM produces new tokens without padding; all outputs are valid
        vp_mask_h = vp_feat_h.new_ones(vp_feat_h.size(0), vp_feat_h.size(1))    # [b, p']

        # ========== Step 4: high-resolution branch similarity (_h) ==========
        # qs-vf: global sentence-frame matching on compacted frames;
        # qw-vf: local word-frame matching on compacted frames;
        # qw-vp: local word-patch matching on cross-attended patches.
        # The three similarities are fused with learnable weights.
        sims_qs_vf_h = self.qs_and_vf(qs_feat, qs_mask, vf_feat_h, vf_mask_h)  # [a, b]
        sims_qw_vf_h = self.qw_and_vf(qw_feat, qw_mask, vf_feat_h, vf_mask_h)  # [a, b]
        sims_qw_vp_h = self.qw_and_vp(qw_feat, qw_mask, vp_feat_h, vp_mask_h)  # [a, b]
        sims_weights_h = torch.softmax(self.sims_weights_h, dim=0)
        sims_h = (sims_weights_h[0] * sims_qs_vf_h +
                  sims_weights_h[1] * sims_qw_vf_h +
                  sims_weights_h[2] * sims_qw_vp_h)
        loss_sims_h = (self.loss_fct(sims_h * logit_scale) + self.loss_fct(sims_h.T * logit_scale)) / 2.0

        # ========== Step 5: low-resolution branch similarity (_l) ==========
        # Same three similarities computed on raw (unprocessed) frame/patch
        # features, fused with a separate set of learnable weights.
        sims_qs_vf_l = self.qs_and_vf(qs_feat, qs_mask, vf_feat_l, vf_mask_l)  # [a, b]
        sims_qw_vf_l = self.qw_and_vf(qw_feat, qw_mask, vf_feat_l, vf_mask_l)  # [a, b]
        sims_qw_vp_l = self.qw_and_vp(qw_feat, qw_mask, vp_feat_l, vp_mask_l)  # [a, b]
        sims_weights_l = torch.softmax(self.sims_weights_l, dim=0)
        sims_l = (sims_weights_l[0] * sims_qs_vf_l +
                  sims_weights_l[1] * sims_qw_vf_l +
                  sims_weights_l[2] * sims_qw_vp_l)
        loss_sims_l = (self.loss_fct(sims_l * logit_scale) + self.loss_fct(sims_l.T * logit_scale)) / 2.0

        # ========== Step 6: total loss with KL alignment ==========
        # Bidirectional KL divergence aligns low-resolution and high-resolution
        # fused similarity distributions in both row and column directions.
        loss_sims_kl = (self.loss_kl(sims_l, sims_h) + self.loss_kl(sims_l.T, sims_h.T) +
                        self.loss_kl(sims_h, sims_l) + self.loss_kl(sims_h.T, sims_l.T)) / 4.0

        total_loss = loss_sims_h + loss_sims_l + loss_sims_kl

        if self.training:
            return total_loss
        else:
            return None

    def get_text_feat(self, text_ids, text_mask):
        """
        Extract multi-granularity text features through CLIP.

        Args:
            text_ids (Tensor): text token IDs, shape [bs_pair, L].
            text_mask (Tensor): text attention mask, shape [bs_pair, L].

        Returns:
            tuple:
                - s_feat (Tensor): sentence-level [CLS] features, [bs_pair, d].
                - w_feat (Tensor): word-level hidden features, [bs_pair, num_words, d].
        """
        text_ids = text_ids.reshape(-1, text_ids.shape[-1])
        text_mask = text_mask.reshape(-1, text_mask.shape[-1])

        bs_pair = text_ids.size(0)
        s_feat, w_feat = self.clip.encode_text(text_ids, return_hidden=True, mask=text_mask)
        s_feat = s_feat.float().reshape(bs_pair, s_feat.size(-1))
        w_feat = w_feat.float().reshape(bs_pair, -1, w_feat.size(-1))
        return s_feat, w_feat

    def get_video_feat(self, video, video_mask):
        """
        Extract multi-granularity video features through CLIP.

        Args:
            video (Tensor): video frame pixels, shape [bs_pair * n_v, channel, h, w].
            video_mask (Tensor): video frame mask, shape [bs_pair, n_v].

        Returns:
            tuple:
                - f_feat (Tensor): frame-level [CLS] features, [bs_pair, num_frames, d].
                - p_feat (Tensor): patch-level hidden features, [bs_pair, num_patches, d].
                  Note: num_patches = num_frames * 49 for ViT-B/32 224x224 input.
        """
        if not self.training:
            # Reshape inputs for inference mode to match training dimensions
            video_mask = video_mask.reshape(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            if len(video.size()) == 5:
                b, n_v, d, h, w = video.shape
                video = video.reshape(b * n_v, d, h, w)
            else:
                b, pair, bs, ts, channel, h, w = video.shape
                video = video.reshape(b * pair * bs * ts, channel, h, w)

        bs_pair, n_v = video_mask.size()
        f_feat, p_feat = self.clip.encode_image(video, return_hidden=True, mask=video_mask)
        f_feat = f_feat.float().reshape(bs_pair, -1, f_feat.size(-1))
        f_feat = self.agg_video_feat(f_feat, video_mask, self.agg_module)
        p_feat = p_feat.float().reshape(bs_pair, -1, p_feat.size(-1))
        return f_feat, p_feat

    def agg_video_feat(self, video_feat, video_mask, agg_module):
        """Aggregate frame-level features into a unified temporal representation.

        Supports three aggregation strategies:
            - "None"   : identity (no aggregation).
            - "seqLSTM": unidirectional LSTM with residual connection.
            - "seqTransf": multi-head Transformer encoder with positional
              embeddings and residual connection.

        Args:
            video_feat (Tensor): frame features, shape [bs, num_frames, d].
            video_mask (Tensor): frame validity mask, shape [bs, num_frames].
            agg_module (str): aggregation mode, one of "None", "seqLSTM",
                "seqTransf".

        Returns:
            Tensor: aggregated frame features, same shape as input [bs, num_frames, d].
        """
        video_feat = video_feat.contiguous()
        if agg_module == "None":
            pass
        elif agg_module == "seqLSTM":
            # Sequential type: LSTM
            video_feat_original = video_feat
            video_feat = pack_padded_sequence(video_feat, torch.sum(video_mask, dim=-1).cpu(),
                                              batch_first=True, enforce_sorted=False)
            video_feat, _ = self.lstm_visual(video_feat)
            if self.training:
                self.lstm_visual.flatten_parameters()
            video_feat, _ = pad_packed_sequence(video_feat, batch_first=True)
            video_feat = torch.cat(
                (video_feat, video_feat_original[:, video_feat.size(1):, ...].contiguous()), dim=1)
            video_feat = video_feat + video_feat_original
        elif agg_module == "seqTransf":
            # Sequential type: Transformer Encoder
            video_feat_original = video_feat
            seq_length = video_feat.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=video_feat.device)
            position_ids = position_ids.unsqueeze(0).expand(video_feat.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            video_feat = video_feat + frame_position_embeddings
            extended_video_mask = (1.0 - video_mask.unsqueeze(1)) * -1000000.0
            extended_video_mask = extended_video_mask.expand(-1, video_mask.size(1), -1)
            video_feat = video_feat.permute(1, 0, 2)  # NLD -> LND
            video_feat = self.transformerClip(video_feat, extended_video_mask)
            video_feat = video_feat.permute(1, 0, 2)  # LND -> NLD
            video_feat = video_feat + video_feat_original
        return video_feat

    def norm(self, feat):
        """
        Apply L2 normalization along the last (feature) dimension.

        Args:
            feat (Tensor): input features, shape [..., d].

        Returns:
            Tensor: L2-normalized features, same shape as input.
        """
        return feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)

    def qs_and_vf(self, qs_feat, qs_mask, vf_feat, vf_mask):
        """
        Global-level spatial interaction: query-sentence <-> video-frame.

        Symmetric attention mechanism:
            1) qs -> vf: best-matching frame per query via max aggregation;
            2) vf -> qs: weighted frame aggregation.
        Final similarity is the average of both directions.

        Args:
            qs_feat (Tensor): query sentence features, shape [a, d].
            qs_mask (Tensor): query sentence mask, shape [a, 1].
            vf_feat (Tensor): video frame features, shape [b, f, d].
            vf_mask (Tensor): video frame mask, shape [b, f].

        Returns:
            Tensor: similarity matrix, shape [a, b].
        """
        # Compute learnable aggregation weights for each video frame
        vf_feat_w = self.vf_feat_w(vf_feat).squeeze(-1)  # [b, f]
        vf_feat_w = vf_feat_w.masked_fill((1 - vf_mask).to(torch.bool), float(-9e15))
        # Safe softmax: if a video has zero valid frames, fall back to uniform weights
        vf_feat_w = torch.softmax(vf_feat_w, dim=-1)

        # Compute pairwise similarity [a, b, f]
        sims_qs_vf = torch.einsum("ad,bfd->abf", [self.norm(qs_feat), self.norm(vf_feat)])
        sims_qs_vf = torch.einsum('abf,bf->abf', [sims_qs_vf, vf_mask])

        # Direction: query -> video, max over frame dimension
        sims_qs2vf, _ = sims_qs_vf.max(dim=-1)  # [a, b]

        # Direction: video -> query, frame-weighted aggregation
        sims_vf2qs = torch.einsum('abf,bf->ab', [sims_qs_vf, vf_feat_w])  # [a, b]

        sims_qs_vf = (sims_qs2vf + sims_vf2qs) / 2.0
        return sims_qs_vf

    def qw_and_vp(self, qw_feat, qw_mask, vp_feat, vp_mask):
        """
        Local-level spatial interaction: query-word <-> video-patch.

        Symmetric attention mechanism:
            1) qw -> vp: best-matching patch then word, then word-weighted aggregate;
            2) vp -> qw: best-matching word, then patch-weighted aggregate.
        Final similarity is the average of both directions.

        Args:
            qw_feat (Tensor): query word features, shape [a, w, d].
            qw_mask (Tensor): query word mask, shape [a, w].
            vp_feat (Tensor): video patch features after ActionFlow, shape [b, p, d].
            vp_mask (Tensor): video patch mask after ActionFlow, shape [b, p].

        Returns:
            Tensor: similarity matrix, shape [a, b].
        """
        # Compute learnable aggregation weights for each query word
        qw_feat_w = self.qw_feat_w(qw_feat).squeeze(-1)  # [a, w]
        qw_feat_w = qw_feat_w.masked_fill((1 - qw_mask).to(torch.bool), float(-9e15))
        qw_feat_w = torch.softmax(qw_feat_w, dim=-1)

        # Compute learnable aggregation weights for each video patch
        vp_feat_w = self.vp_feat_w(vp_feat).squeeze(-1)  # [b, p]
        vp_feat_w = vp_feat_w.masked_fill((1 - vp_mask).to(torch.bool), float(-9e15))
        vp_feat_w = torch.softmax(vp_feat_w, dim=-1)

        # Compute pairwise similarity [a, b, w, p]
        sims_qw_vp = torch.einsum("awd,bpd->abwp", [self.norm(qw_feat), self.norm(vp_feat)])
        sims_qw_vp = torch.einsum('abwp,aw->abwp', [sims_qw_vp, qw_mask])
        sims_qw_vp = torch.einsum('abwp,bp->abwp', [sims_qw_vp, vp_mask])

        # Direction: query-word -> video-patch, max over patch, then word-weighted
        sims_qw2vp, _ = sims_qw_vp.max(dim=-1)  # [a, b, w]
        sims_qw2vp = torch.einsum('abw,aw->ab', [sims_qw2vp, qw_feat_w])

        # Direction: video-patch -> query-word, max over word then patch-weighted
        sims_vp2qw, _ = sims_qw_vp.max(dim=-2)  # [a, b, p]
        sims_vp2qw = torch.einsum('abp,bp->ab', [sims_vp2qw, vp_feat_w])

        sims_qw_vp = (sims_qw2vp + sims_vp2qw) / 2.0

        return sims_qw_vp
    
    def qs_and_vp(self, qs_feat, qs_mask, vp_feat, vp_mask):
        """
        Global-to-local spatial interaction: query-sentence <-> video-patch.

        Symmetric attention mechanism:
            1) qs -> vp: best-matching patch per query via max aggregation;
            2) vp -> qs: weighted patch aggregation.
        Final similarity is the average of both directions.

        Args:
            qs_feat (Tensor): query sentence features, shape [a, d].
            qs_mask (Tensor): query sentence mask, shape [a, 1].
            vp_feat (Tensor): video patch features after ActionFlow, shape [b, p, d].
            vp_mask (Tensor): video patch mask after ActionFlow, shape [b, p].

        Returns:
            Tensor: similarity matrix, shape [a, b].
        """
        # Compute learnable aggregation weights for each video patch
        vp_feat_w = self.vp_feat_w(vp_feat).squeeze(-1)  # [b, p]
        vp_feat_w = vp_feat_w.masked_fill((1 - vp_mask).to(torch.bool), float(-9e15))
        vp_feat_w = torch.softmax(vp_feat_w, dim=-1)

        # Compute pairwise similarity [a, b, p]
        sims_qs_vp = torch.einsum("ad,bpd->abp", [self.norm(qs_feat), self.norm(vp_feat)])
        sims_qs_vp = torch.einsum('abp,bp->abp', [sims_qs_vp, vp_mask])

        # Direction: query -> video-patch, max over patch dimension
        sims_qs2vp, _ = sims_qs_vp.max(dim=-1)  # [a, b]

        # Direction: video-patch -> query, patch-weighted aggregation
        sims_vp2qs = torch.einsum('abp,bp->ab', [sims_qs_vp, vp_feat_w])  # [a, b]

        sims_qs_vp = (sims_qs2vp + sims_vp2qs) / 2.0
        return sims_qs_vp
    
    def qw_and_vf(self, qw_feat, qw_mask, vf_feat, vf_mask):
        """
        Local-to-global spatial interaction: query-word <-> video-frame.

        Symmetric attention mechanism:
            1) qw -> vf: best-matching frame per word, then word-weighted aggregate;
            2) vf -> qw: best-matching word per frame, then frame-weighted aggregate.
        Final similarity is the average of both directions.

        Args:
            qw_feat (Tensor): query word features, shape [a, w, d].
            qw_mask (Tensor): query word mask, shape [a, w].
            vf_feat (Tensor): video frame features, shape [b, f, d].
            vf_mask (Tensor): video frame mask, shape [b, f].

        Returns:
            Tensor: similarity matrix, shape [a, b].
        """
        # Compute learnable aggregation weights for each query word
        qw_feat_w = self.qw_feat_w(qw_feat).squeeze(-1)  # [a, w]
        qw_feat_w = qw_feat_w.masked_fill((1 - qw_mask).to(torch.bool), float(-9e15))
        qw_feat_w = torch.softmax(qw_feat_w, dim=-1)

        # Compute learnable aggregation weights for each video frame
        vf_feat_w = self.vf_feat_w(vf_feat).squeeze(-1)  # [b, f]
        vf_feat_w = vf_feat_w.masked_fill((1 - vf_mask).to(torch.bool), float(-9e15))
        vf_feat_w = torch.softmax(vf_feat_w, dim=-1)

        # Compute pairwise similarity [a, b, w, f]
        sims_qw_vf = torch.einsum("awd,bfd->abwf", [self.norm(qw_feat), self.norm(vf_feat)])
        sims_qw_vf = torch.einsum('abwf,aw->abwf', [sims_qw_vf, qw_mask])
        sims_qw_vf = torch.einsum('abwf,bf->abwf', [sims_qw_vf, vf_mask])

        # Direction: query-word -> video-frame, max over frame then word-weighted
        sims_qw2vf, _ = sims_qw_vf.max(dim=-1)  # [a, b, w]
        sims_qw2vf = torch.einsum('abw,aw->ab', [sims_qw2vf, qw_feat_w])

        # Direction: video-frame -> query-word, max over word then frame-weighted
        sims_vf2qw, _ = sims_qw_vf.max(dim=-2)  # [a, b, f]
        sims_vf2qw = torch.einsum('abf,bf->ab', [sims_vf2qw, vf_feat_w])

        sims_qw_vf = (sims_qw2vf + sims_vf2qw) / 2.0
        
        return sims_qw_vf

    def get_similarity_logits(self, qs_feat, qs_mask, qw_feat, qw_mask, vf_feat, vf_mask, vp_feat, vp_mask):
        """
        Compute fused similarity logits for inference.

        Computes three similarity matrices on the input features directly:
            - qs-vf: global sentence-frame matching;
            - qw-vf: local word-frame matching;
            - qw-vp: local word-patch matching.
        The three similarities are fused with learnable weights (softmax-normalized)
        into a single similarity matrix.

        Note: unlike training forward(), this method does NOT apply ActionFlow
        patch processing (PCM + Cross_Att_Block_Patch). If called with raw CLIP
        patch features, similarity is computed directly over them.

        Args:
            qs_feat (Tensor): query sentence features, [a, d].
            qs_mask (Tensor): query sentence mask, [a, 1].
            qw_feat (Tensor): query word features, [a, w, d].
            qw_mask (Tensor): query word mask, [a, w].
            vf_feat (Tensor): video frame features, [b, f, d].
            vf_mask (Tensor): video frame mask, [b, f].
            vp_feat (Tensor): video patch features, [b, p, d].
            vp_mask (Tensor): video patch mask, [b, p].

        Returns:
            Tensor: fused similarity matrix, shape [a, b].
        """
        # Dimension aliases
        a, w, d = qs_feat.size(0), qw_feat.size(1), qw_feat.size(2)
        b, f, p = vf_feat.size(0), vf_feat.size(1), vp_feat.size(1)

        sims_qs_vf_l = self.qs_and_vf(qs_feat, qs_mask, vf_feat, vf_mask)  # [a, b]
        sims_qw_vf_l = self.qw_and_vf(qw_feat, qw_mask, vf_feat, vf_mask)  # [a, b]
        sims_qw_vp_l = self.qw_and_vp(qw_feat, qw_mask, vp_feat, vp_mask)  # [a, b]
        sims_weights_l = torch.softmax(self.sims_weights_l, dim=0)
        sims_l = (sims_weights_l[0] * sims_qs_vf_l +
                  sims_weights_l[1] * sims_qw_vf_l +
                  sims_weights_l[2] * sims_qw_vp_l)

        return sims_l

    @property
    def dtype(self):
        """Return the dtype of the first model parameter.

        Falls back to scanning all tensor attributes if no parameters exist
        (e.g., for an empty module).
        """
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            def find_tensor_attributes(module: nn.Module):
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples

            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].dtype

    def init_weights(self, module):
        """Initialize weights for newly added modules.

        Applies:
            - Normal init (mean=0, std=0.02) to Linear and Embedding weights;
            - Zero bias for Linear layers;
            - Zero beta / one gamma (or zero bias / one weight) for LayerNorm.

        Intended to be passed to ``self.apply(self.init_weights)`` before
        loading pretrained CLIP weights so that only non-pretrained parameters
        are randomly initialized.

        Args:
            module (nn.Module): submodule to initialize.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()