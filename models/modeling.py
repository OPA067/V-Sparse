import os
from collections import OrderedDict
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F

from differential_topk import VisualTokenSelection
from module_clip import CLIP, convert_weights, _PT_NAME
from module_cross import CrossModel, Transformer as TransformerClip
from until_module import LayerNorm, AllGather, AllGather2, CrossEn, MSE, ArcCrossEn, KL
import numpy as np
from cluster import CTM, TCBlock
from video_pooling import video_pooling
from video_spliting import video_spliting
from video_transfomer import video_transformer

allgather = AllGather.apply
allgather2 = AllGather2.apply

class ResidualLinear(nn.Module):
    def __init__(self, d_int: int):
        super(ResidualLinear, self).__init__()

        self.fc_relu = nn.Sequential(nn.Linear(d_int, d_int), nn.ReLU(inplace=True))

    def forward(self, x):
        x = x + self.fc_relu(x)
        return x

class VSC_HA(nn.Module):
    def __init__(self, config):
        super(VSC_HA, self).__init__()

        self.config = config
        self.interaction = config.interaction
        self.agg_module = getattr(config, 'agg_module', 'meanP')
        backbone = getattr(config, 'base_encoder', "ViT-B/32")

        assert backbone in _PT_NAME
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _PT_NAME[backbone])
        if os.path.exists(model_path):
            FileNotFoundError
        try:
            # loading JIT archive
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

        self.loss_fct = CrossEn(config)

        self.apply(self.init_weights)
        self.clip.load_state_dict(state_dict, strict=False)

        # Frames
        self.v_ctm_f = CTM(sample_ratio=0.75, embed_dim=512, dim_out=512, k=3)
        self.v_block_f = TCBlock(dim=512, num_heads=8)

        # Patches
        self.v_ctm_p_1 = CTM(sample_ratio=0.5, embed_dim=512, dim_out=512, k=3)
        self.v_block_p_1 = TCBlock(dim=512, num_heads=8)
        self.v_ctm_p_2 = CTM(sample_ratio=0.25, embed_dim=512, dim_out=512, k=3)
        self.v_block_p_2 = TCBlock(dim=512, num_heads=8)
        self.v_ctm_p_3 = CTM(sample_ratio=0.125, embed_dim=512, dim_out=512, k=3)
        self.v_block_p_3 = TCBlock(dim=512, num_heads=8)

        embed_dim = state_dict["text_projection"].shape[1]
        self.visual_token_selector = VisualTokenSelection(self.config.max_frames, embed_dim, topk=3)
        self.video_transformer = video_transformer()

        self.video_pooling = video_pooling()
        self.video_spliting = video_spliting()

        self.mse = MSE()
        self.kl = KL()

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

    def forward(self, text_ids, text_mask, video, video_mask=None, idx=None, global_step=0):
        text_ids = text_ids.view(-1, text_ids.shape[-1])
        text_mask = text_mask.view(-1, text_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        video = torch.as_tensor(video).float()
        if len(video.size()) == 5:
            b, n_v, d, h, w = video.shape
            video = video.view(b * n_v, d, h, w)
        else:
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)

        text_feat, video_feat, visual_feat = self.get_text_video_feat(text_ids, text_mask, video, video_mask, shaped=True)

        if self.training:
            if torch.cuda.is_available():
                idx = allgather(idx, self.config)
                text_feat = allgather(text_feat, self.config)
                video_feat = allgather(video_feat, self.config)
                text_mask = allgather(text_mask, self.config)
                video_mask = allgather(video_mask, self.config)
                visual_feat = allgather(visual_feat, self.config)
                torch.distributed.barrier()

            idx = idx.view(-1, 1)
            idx_all = idx.t()
            pos_idx = torch.eq(idx, idx_all).float()
            logit_scale = self.clip.logit_scale.exp()
            loss = 0.

            # Frames Features
            v_idx_token = torch.arange(video_feat.size(1))[None, :].repeat(video_feat.size(0), 1)
            v_agg_weight = video_feat.new_ones(video_feat.size(0), video_feat.size(1), 1)
            v_mask = torch.ones(video_feat.size(0), video_feat.size(1)).to(video_feat.device)
            v_token_dict = {'x': video_feat,
                            'token_num': video_feat.size(1),
                            'idx_token': v_idx_token,
                            'agg_weight': v_agg_weight,
                            'mask': v_mask.detach()}
            v_token_dict = self.v_block_f(self.v_ctm_f(v_token_dict), text_feat)
            video_feat_f = v_token_dict["x"]

            # Patches Features
            v_idx_token = torch.arange(visual_feat.size(1))[None, :].repeat(visual_feat.size(0), 1)
            v_agg_weight = visual_feat.new_ones(visual_feat.size(0), visual_feat.size(1), 1)
            v_mask = torch.ones(visual_feat.size(0), visual_feat.size(1)).to(visual_feat.device)
            v_token_dict = {'x': visual_feat,
                            'token_num': visual_feat.size(1),
                            'idx_token': v_idx_token,
                            'agg_weight': v_agg_weight,
                            'mask': v_mask.detach()}
            v_token_dict = self.v_block_p_1(self.v_ctm_p_1(v_token_dict), text_feat)
            v_token_dict = self.v_block_p_2(self.v_ctm_p_2(v_token_dict), text_feat)
            v_token_dict = self.v_block_p_3(self.v_ctm_p_3(v_token_dict), text_feat)
            video_feat_p = v_token_dict["x"]

            video_feat = torch.cat([video_feat_f, video_feat_p], dim=1)

            output_coarse = self.video_pooling(text_feat, video_feat)
            loss = loss + self.loss_fct(output_coarse * logit_scale) + self.loss_fct(output_coarse.T * logit_scale)

            video_pool = self.video_transformer(text_feat, video_feat)
            output_medium = self.sim_matrix_training(text_feat, video_pool)
            loss = loss + self.loss_fct(output_medium * logit_scale) + self.loss_fct(output_medium.T * logit_scale)

            output_fine = self.video_spliting(text_feat, video_feat)
            loss = loss + self.loss_fct(output_fine * logit_scale) + self.loss_fct(output_fine.T * logit_scale)

            return loss
        else:
            return None

    def sim_matrix_training(self, text_embeds, vid_embeds_pooled):
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        vid_embeds_pooled = vid_embeds_pooled / vid_embeds_pooled.norm(dim=-1, keepdim=True)

        sims = torch.mm(text_embeds, vid_embeds_pooled.t())

        return sims
    def get_text_feat(self, text_ids, text_mask, shaped=False):
        if shaped is False:
            text_ids = text_ids.view(-1, text_ids.shape[-1])
            text_mask = text_mask.view(-1, text_mask.shape[-1])

        bs_pair = text_ids.size(0)
        text_feat = self.clip.encode_text(text_ids, return_hidden=False, mask=text_mask)
        text_feat = text_feat.float()
        text_feat = text_feat.view(bs_pair, -1, text_feat.size(-1)).squeeze(1)
        return text_feat

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
        video_feat, visual_feat = self.clip.encode_image(video, return_hidden=True)
        video_feat = video_feat.float()
        visual_feat = visual_feat.float()
        video_feat = video_feat.float().view(bs_pair, -1, video_feat.size(-1))
        visual_feat = visual_feat.float().view(bs_pair, -1, visual_feat.size(-1))
        # visual_feat = self.visual_token_selector(visual_feat)

        return video_feat, visual_feat

    def get_text_video_feat(self, text_ids, text_mask, video, video_mask, shaped=False):
        if shaped is False:
            text_ids = text_ids.view(-1, text_ids.shape[-1])
            text_mask = text_mask.view(-1, text_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            if len(video.shape) == 5:
                b, n_v, d, h, w = video.shape
                video = video.view(b * n_v, d, h, w)
            else:
                b, pair, bs, ts, channel, h, w = video.shape
                video = video.view(b * pair * bs * ts, channel, h, w)

        text_feat = self.get_text_feat(text_ids, text_mask, shaped=True)
        video_feat, visual_feat = self.get_video_feat(video, video_mask, shaped=True)

        return text_feat, video_feat, visual_feat

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
            video_feat = pack_padded_sequence(video_feat, torch.sum(video_mask, dim=-1).cpu(), batch_first=True, enforce_sorted=False)
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

    def get_similarity_logits(self, text_feat, video_feat, visual_feat, text_mask, video_mask, shaped=False):
        if shaped is False:
            text_mask = text_mask.view(-1, text_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        # Frames Features
        v_idx_token = torch.arange(video_feat.size(1))[None, :].repeat(video_feat.size(0), 1)
        v_agg_weight = video_feat.new_ones(video_feat.size(0), video_feat.size(1), 1)
        v_mask = torch.ones(video_feat.size(0), video_feat.size(1)).to(video_feat.device)
        v_token_dict = {'x': video_feat,
                        'token_num': video_feat.size(1),
                        'idx_token': v_idx_token,
                        'agg_weight': v_agg_weight,
                        'mask': v_mask.detach()}
        v_token_dict = self.v_block_f(self.v_ctm_f(v_token_dict), text_feat)
        video_feat_f = v_token_dict["x"]

        # Patches Features
        v_idx_token = torch.arange(visual_feat.size(1))[None, :].repeat(visual_feat.size(0), 1)
        v_agg_weight = visual_feat.new_ones(visual_feat.size(0), visual_feat.size(1), 1)
        v_mask = torch.ones(visual_feat.size(0), visual_feat.size(1)).to(visual_feat.device)
        v_token_dict = {'x': visual_feat,
                        'token_num': visual_feat.size(1),
                        'idx_token': v_idx_token,
                        'agg_weight': v_agg_weight,
                        'mask': v_mask.detach()}
        v_token_dict = self.v_block_p_1(self.v_ctm_p_1(v_token_dict), text_feat)
        v_token_dict = self.v_block_p_2(self.v_ctm_p_2(v_token_dict), text_feat)
        v_token_dict = self.v_block_p_3(self.v_ctm_p_3(v_token_dict), text_feat)
        video_feat_p = v_token_dict["x"]
        video_feat = torch.cat([video_feat_f, video_feat_p], dim=1)

        output_coarse = self.video_pooling(text_feat, video_feat)
        video_pool = self.video_transformer(text_feat, video_feat)
        output_medium = self.sim_matrix_training(text_feat, video_pool)
        output_fine = self.video_spliting(text_feat, video_feat)

        return (output_coarse + output_medium + output_fine)/3.0

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