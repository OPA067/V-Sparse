import os
from collections import OrderedDict
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F

from VQA.models.alignmentCoarse import alignmentCoarse
from VQA.models.alignmentFine import alignmentFine
from VQA.models.alignmentMedium import alignmentMedium
from module_clip import CLIP, convert_weights, _PT_NAME
from module_cross import CrossModel, Transformer as TransformerClip
from until_module import LayerNorm, AllGather, AllGather2, CrossEn, MSE, KL
from cluster import FCM, Att_Block, PCM


allgather = AllGather.apply
allgather2 = AllGather2.apply

class ResidualLinear(nn.Module):
    def __init__(self, d_int: int):
        super(ResidualLinear, self).__init__()
        self.fc_relu = nn.Sequential(nn.Linear(d_int, d_int), nn.ReLU(inplace=True))
    def forward(self, x):
        x = x + self.fc_relu(x)
        return x

class V_Sparse_VQA(nn.Module):
    def __init__(self, config):
        super(V_Sparse_VQA, self).__init__()

        self.config = config
        self.interaction = config.interaction
        self.agg_module = getattr(config, 'agg_module', 'meanP')
        backbone = getattr(config, 'base_encoder', "ViT-B/32")

        assert backbone in _PT_NAME
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _PT_NAME[backbone])
        if os.path.exists(model_path):
            FileNotFoundError
        try:
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = model.state_dict()
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size

        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

        self.clip = CLIP(embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size,
                         context_length, vocab_size, transformer_width, transformer_heads, transformer_layers)

        if torch.cuda.is_available():
            convert_weights(self.clip)  # fp16

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

        if self.agg_module in ["seqLSTM", "seqTransf"]:
            self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings,
                                                          cross_config.hidden_size)
            if self.agg_module == "seqTransf":
                self.transformerClip = TransformerClip(width=transformer_width,
                                                       layers=config.num_hidden_layers,
                                                       heads=transformer_heads)
            if self.agg_module == "seqLSTM":
                self.lstm_visual = nn.LSTM(input_size=cross_config.hidden_size, hidden_size=cross_config.hidden_size,
                                           batch_first=True, bidirectional=False, num_layers=1)
        self.clip.load_state_dict(state_dict, strict=False)
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

        self.apply(self.init_weights)

        # 1. First, semantic compression of frames in the temporal dimension.
        self.v_fcm_f = FCM(sample_ratio=0.75, embed_dim=embed_dim, dim_out=embed_dim, k=3)
        self.v_att_block_f = Att_Block(dim=embed_dim, num_heads=8)

        # 2. Next, semantic compression of patches in the spatial dimension.
        self.v_pcm_p_1 = PCM(sample_ratio=0.5, embed_dim=embed_dim, dim_out=embed_dim, k=3)
        self.v_att_block_p_1 = Att_Block(dim=embed_dim, num_heads=8)
        self.v_pcm_p_2 = PCM(sample_ratio=0.25, embed_dim=embed_dim, dim_out=embed_dim, k=3)
        self.v_att_block_p_2 = Att_Block(dim=embed_dim, num_heads=8)
        self.v_pcm_p_3 = PCM(sample_ratio=0.125, embed_dim=embed_dim, dim_out=embed_dim, k=3)
        self.v_att_block_p_3 = Att_Block(dim=embed_dim, num_heads=8)

        # 3. Then， coarse, medium, and fine-grained alignment.
        self.alignmentCoarse = alignmentCoarse()
        self.alignmentMedium = alignmentMedium()
        self.alignmentFine = alignmentFine()

        # 4. Finally, distillation between similarity matrices of coarse and fine granularity for alignment.
        self.kl = KL()
        self.mse = MSE()

        self.t_proj = nn.Linear(transformer_width, 4 * transformer_width)
        self.v_proj = nn.Linear(transformer_width, 4 * transformer_width)
        self.loss_fct = CrossEn()
        self.dropout = nn.Dropout(0.1)
        hidden_size = transformer_width * 8
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(True),
            nn.Linear(hidden_size * 2, self.config.num_labels)
        )

        self.t_proj_1 = nn.Linear(transformer_width, 4 * transformer_width)
        self.v_proj_1 = nn.Linear(transformer_width, 4 * transformer_width)
        self.dropout_1 = nn.Dropout(0.1)
        hidden_size = transformer_width * 8
        self.classifier_1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(True),
            nn.Linear(hidden_size * 2, self.config.num_labels)
        )

        self.t_proj_2 = nn.Linear(transformer_width, 4 * transformer_width)
        self.v_proj_2 = nn.Linear(transformer_width, 4 * transformer_width)
        self.dropout_2 = nn.Dropout(0.1)
        hidden_size = transformer_width * 8
        self.classifier_2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(True),
            nn.Linear(hidden_size * 2, self.config.num_labels)
        )

    def calc_loss(self, logits, labels):
        if labels is not None:
            loss_fct = CrossEntropyLoss(reduction="mean")
            loss = loss_fct(
                logits.view(-1, self.task_config.num_labels),
                labels.view(-1))
        else:
            loss = 0
        return loss

    def forward(self, text, text_mask, video, video_mask, idx=None, labels=None):

        text = text.view(-1, text.shape[-1])
        text_mask = text_mask.view(-1, text_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        video = torch.as_tensor(video).float()
        if len(video.size()) == 5:
            b, n_v, d, h, w = video.shape
            video = video.view(b * n_v, d, h, w)
        else:
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)

        s_feat, w_feat, f_feat, p_feat = self.get_text_video_feat(text, text_mask, video, video_mask, shaped=True)

        if self.training:
            if torch.cuda.is_available():
                idx = allgather(idx, self.config)
                text_mask = allgather(text_mask, self.config)
                s_feat = allgather(s_feat, self.config)
                w_feat = allgather(w_feat, self.config)
                video_mask = allgather(video_mask, self.config)
                f_feat = allgather(f_feat, self.config)
                p_feat = allgather(p_feat, self.config)
                torch.distributed.barrier()

            idx = idx.view(-1, 1)
            idx_all = idx.t()
            pos_idx = torch.eq(idx, idx_all).float()
            logit_scale = self.clip.logit_scale.exp()
            loss = 0.

            """Add w_idx_token:
            w_idx_token = torch.arange(w_feat.size(1))[None, :].repeat(w_feat.size(0), 1)
            w_agg_weight = w_feat.new_ones(w_feat.size(0), w_feat.size(1), 1)
            w_mask = torch.ones(w_feat.size(0), w_feat.size(1)).to(w_feat.device)
            w_token_dict = {'x': w_feat,
                            'token_num': w_feat.size(1),
                            'idx_token': w_idx_token,
                            'agg_weight': w_agg_weight,
                            'mask': w_mask.detach()}"""

            s_idx_token = torch.arange(s_feat.size(1))[None, :].repeat(s_feat.size(0), 1)
            s_agg_weight = s_feat.new_ones(s_feat.size(0), s_feat.size(1), 1)
            s_mask = torch.ones(s_feat.size(0), s_feat.size(1)).to(s_feat.device)
            s_token_dict = {'x': s_feat,
                            'token_num': s_feat.size(1),
                            'idx_token': s_idx_token,
                            'agg_weight': s_agg_weight,
                            'mask': s_mask.detach()}

            # ===>>> frame features compression
            f_idx_token = torch.arange(f_feat.size(1))[None, :].repeat(f_feat.size(0), 1)
            f_agg_weight = f_feat.new_ones(f_feat.size(0), f_feat.size(1), 1)
            f_mask = torch.ones(f_feat.size(0), f_feat.size(1)).to(f_feat.device)
            f_token_dict = {'x': f_feat,
                            'token_num': f_feat.size(1),
                            'idx_token': f_idx_token,
                            'agg_weight': f_agg_weight,
                            'mask': f_mask.detach()}
            f_token_dict = self.v_att_block_f(self.v_fcm_f(f_token_dict), s_token_dict)
            f_feat = f_token_dict["x"]

            # ===>>> patch features compression
            p_idx_token = torch.arange(p_feat.size(1))[None, :].repeat(p_feat.size(0), 1)
            p_agg_weight = p_feat.new_ones(p_feat.size(0), p_feat.size(1), 1)
            p_mask = torch.ones(p_feat.size(0), p_feat.size(1)).to(p_feat.device)
            p_token_dict = {'x': p_feat,
                            'token_num': p_feat.size(1),
                            'idx_token': p_idx_token,
                            'agg_weight': p_agg_weight,
                            'mask': p_mask.detach()}
            p_token_dict = self.v_att_block_p_1(self.v_pcm_p_1(p_token_dict), s_token_dict)
            p_token_dict = self.v_att_block_p_2(self.v_pcm_p_2(p_token_dict), s_token_dict)
            p_token_dict = self.v_att_block_p_3(self.v_pcm_p_3(p_token_dict), s_token_dict)
            p_feat = p_token_dict["x"]
            v_feat = torch.cat([f_feat, p_feat], dim=1)
            s_feat = s_feat.squeeze(1)

            # Example alignment coarse features
            coarse_t, coarse_v, CoarseScore = self.alignmentCoarse(s_feat, v_feat)
            coarse_t = self.t_proj(coarse_t)
            coarse_v = self.v_proj(coarse_v)
            input = torch.cat((coarse_t, coarse_v), dim=-1)
            coarse_output = self.dropout(input)
            logits = self.classifier(coarse_output)
            loss = loss + self.loss_fct(CoarseScore * logit_scale) + self.loss_fct(CoarseScore.T * logit_scale) + self.calc_loss(logits, labels)

            medium_t, medium_v, MediumScore = self.alignmentMedium(s_feat, v_feat)
            medium_t = self.t_proj_1(medium_t)
            medium_v = self.v_proj_1(medium_v)
            input = torch.cat((medium_t, medium_v), dim=-1)
            medium_output = self.dropout(input)
            logits = self.classifier_1(medium_output)
            loss = loss + self.loss_fct(MediumScore * logit_scale) + self.loss_fct(MediumScore.T * logit_scale) + self.calc_loss(logits, labels)

            return loss
        else:
            return None

    def sim_matrix_training(self, text_embeds, vid_embeds_pooled):
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        vid_embeds_pooled = vid_embeds_pooled / vid_embeds_pooled.norm(dim=-1, keepdim=True)

        sims = torch.mm(text_embeds, vid_embeds_pooled.t())

        return sims

    def get_text_feat(self, text, text_mask, shaped=False):
        if shaped is False:
            text = text.view(-1, text.shape[-1])
            text_mask = text_mask.view(-1, text_mask.shape[-1])

        bs_pair = text.size(0)
        s_feat, w_feat = self.clip.encode_text(text, return_hidden=True, mask=text_mask)
        s_feat, w_feat = s_feat.float(), w_feat.float()
        s_feat = s_feat.view(bs_pair, -1, s_feat.size(-1))
        w_feat = w_feat.view(bs_pair, -1, w_feat.size(-1))
        return s_feat, w_feat

    def get_video_feat(self, video, video_mask, shaped=False):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            if len(video.size()) == 5:
                b, n_v, d, h, w = video.shape
                video = video.view(b * n_v, d, h, w)
            else:
                b, pair, bs, ts, channel, h, w = video.shape
                video = video.view(b * pair * bs * ts, channel, h, w)

        bs_pair, n_v = video_mask.size()
        f_feat, p_feat = self.clip.encode_image(video, return_hidden=True)
        f_feat, p_feat = f_feat.float(), p_feat.float()
        f_feat = f_feat.float().view(bs_pair, -1, f_feat.size(-1))
        p_feat = p_feat.float().view(bs_pair, -1, p_feat.size(-1))

        return f_feat, p_feat

    def get_text_video_feat(self, text, text_mask, video, video_mask, shaped=False):
        if shaped is False:
            text = text.view(-1, text.shape[-1])
            text_mask = text_mask.view(-1, text_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            if len(video.shape) == 5:
                b, n_v, d, h, w = video.shape
                video = video.view(b * n_v, d, h, w)
            else:
                b, pair, bs, ts, channel, h, w = video.shape
                video = video.view(b * pair * bs * ts, channel, h, w)

        s_feat, w_feat = self.get_text_feat(text, text_mask, shaped=True)
        f_feat, p_feat = self.get_video_feat(video, video_mask, shaped=True)

        return s_feat, w_feat, f_feat, p_feat

    def get_video_avg_feat(self, video_feat, video_mask):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        video_feat = video_feat * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_feat = torch.sum(video_feat, dim=1) / video_mask_un_sum
        return video_feat

    def get_text_sep_feat(self, text_feat, text_mask):
        text_feat = text_feat.contiguous()
        text_feat = text_feat[torch.arange(text_feat.shape[0]), torch.sum(text_mask, dim=-1) - 1, :]
        text_feat = text_feat.unsqueeze(1).contiguous()
        return text_feat

    def agg_video_feat(self, video_feat, video_mask, agg_module):
        video_feat = video_feat.contiguous()
        if agg_module == "None":
            pass
        elif agg_module == "seqLSTM":
            video_feat_original = video_feat
            video_feat = pack_padded_sequence(video_feat, torch.sum(video_mask, dim=-1).cpu(), batch_first=True,
                                              enforce_sorted=False)
            video_feat, _ = self.lstm_visual(video_feat)
            if self.training:
                self.lstm_visual.flatten_parameters()
            video_feat, _ = pad_packed_sequence(video_feat, batch_first=True)
            video_feat = torch.cat((video_feat, video_feat_original[:, video_feat.size(1):, ...].contiguous()), dim=1)
            video_feat = video_feat + video_feat_original
        elif agg_module == "seqTransf":

            video_feat_original = video_feat
            seq_length = video_feat.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=video_feat.device)
            position_ids = position_ids.unsqueeze(0).expand(video_feat.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            video_feat = video_feat + frame_position_embeddings
            extended_video_mask = (1.0 - video_mask.unsqueeze(1)) * -1000000.0
            extended_video_mask = extended_video_mask.expand(-1, video_mask.size(1), -1)
            video_feat = video_feat.permute(1, 0, 2)
            video_feat = self.transformerClip(video_feat, extended_video_mask)
            video_feat = video_feat.permute(1, 0, 2)
            video_feat = video_feat + video_feat_original
        return video_feat

    def get_similarity_logits(self, text_ids, text_mask, video, video_mask, idx=None, labels=None):
        # ... ...
        return

    @property
    def dtype(self):
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