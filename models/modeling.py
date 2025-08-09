import os
from collections import OrderedDict
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from types import SimpleNamespace
import torch
from torch import nn

from .cluster import Att_Block_Patch, PCM, FCM, Att_Block_Frame
from .module_CAttention import CAM
from .module_clip import CLIP, convert_weights, _PT_NAME
from .module_cross import Transformer as TransformerClip
from .module_pts import PatchTokenSelection
from .until_module import LayerNorm, AllGather, AllGather2, CrossEn, KL

allgather = AllGather.apply
allgather2 = AllGather2.apply

class ResidualLinear(nn.Module):
    def __init__(self, d_int: int):
        super(ResidualLinear, self).__init__()

        self.fc_relu = nn.Sequential(
            nn.Linear(d_int, d_int),
            nn.ReLU(inplace=True),
            nn.Linear(d_int, d_int), )

    def forward(self, x):
        x = x + self.fc_relu(x)
        return x

class Model(nn.Module):
    def __init__(self, config):

        super(Model, self).__init__()

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
            convert_weights(self.clip)

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
        self.loss_kl = KL(config)

        self.apply(self.init_weights)  # random init must before loading pretrain
        self.clip.load_state_dict(state_dict, strict=False)
        self.alpha, self.beta = self.config.alpha, self.config.beta

        embed_dim = state_dict["text_projection"].shape[1]

        sr_f, sr_p = 0.5, [0.5, 0.5, 0.5]
        self.v_fcm_f_0 = FCM(sample_ratio=sr_f, embed_dim=embed_dim, dim_out=embed_dim, k=3)
        self.v_att_block_f_0 = Att_Block_Frame(dim=embed_dim, num_heads=8)

        self.v_pcm_p_1 = PCM(sample_ratio=sr_p[0], embed_dim=embed_dim, dim_out=embed_dim, k=3)
        self.v_att_block_p_1 = Att_Block_Patch(dim=embed_dim, num_heads=8)
        self.v_pcm_p_2 = PCM(sample_ratio=sr_p[1], embed_dim=embed_dim, dim_out=embed_dim, k=3)
        self.v_att_block_p_2 = Att_Block_Patch(dim=embed_dim, num_heads=8)
        self.v_pcm_p_3 = PCM(sample_ratio=sr_p[2], embed_dim=embed_dim, dim_out=embed_dim, k=3)
        self.v_att_block_p_3 = Att_Block_Patch(dim=embed_dim, num_heads=8)

        # self.cam_sf_h = CAM(embed_dim=embed_dim, dropout=0.1)
        # self.cam_sp_h = CAM(embed_dim=embed_dim, dropout=0.1)
        # self.cam_sf_o = CAM(embed_dim=embed_dim, dropout=0.1)
        # self.cam_sp_o = CAM(embed_dim=embed_dim, dropout=0.1)

        self.s_feat_w = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.ReLU(), nn.Linear(embed_dim * 2, 1), )
        self.w_feat_w = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.ReLU(), nn.Linear(embed_dim * 2, 1), )
        self.f_feat_w = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.ReLU(), nn.Linear(embed_dim * 2, 1), )
        self.p_feat_w = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.ReLU(), nn.Linear(embed_dim * 2, 1), )

        ## ===> Initialization trick [HARD CODE]
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

    def forward(self, text, text_mask, video, video_mask, idx=None, global_step=0):

        text_mask = text_mask.view(-1, text_mask.shape[-1])
        text = text.view(-1, text.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        video = torch.as_tensor(video).float()
        if len(video.size()) == 5:
            b, n_v, d, h, w = video.shape
            video = video.view(b * n_v, d, h, w)
        else:
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)

        s_feat, w_feat, f_feat, p_feat = self.get_text_video_feat(text, text_mask, video, video_mask, shaped=True)
        w_mask, f_mask = text_mask, video_mask
        s_feat, w_feat, w_mask = s_feat.contiguous(), w_feat.contiguous(), w_mask.contiguous()
        f_feat, p_feat, f_mask = f_feat.contiguous(), p_feat.contiguous(), f_mask.contiguous()

        logit_scale = self.clip.logit_scale.exp()

        if self.training:
            s_feat = allgather(s_feat, self.config)
            w_feat = allgather(w_feat, self.config)
            w_mask = allgather(w_mask, self.config)
            f_feat = allgather(f_feat, self.config)
            p_feat = allgather(p_feat, self.config)
            f_mask = allgather(f_mask, self.config)
            torch.distributed.barrier()

        ###### Step-0: Init params ######
        a, s, w, d = s_feat.size(0), 1,              w_feat.size(1), w_feat.size(-1)
        b, f, p, d = f_feat.size(0), f_feat.size(1), p_feat.size(1), p_feat.size(-1)
        s_token_dict = {'x': s_feat, 'token_num': s_feat.size(1),
                        'idx_token': torch.arange(s_feat.size(1))[None, :].repeat(s_feat.size(0), 1),
                        'agg_weight': s_feat.new_ones(s_feat.size(0), s_feat.size(1), 1),
                        'mask': s_feat.new_ones(s_feat.size(0), s_feat.size(1)).detach()}
        f_token_dict = {'x': f_feat, 'token_num': f_feat.size(1),
                        'idx_token': torch.arange(f_feat.size(1))[None, :].repeat(f_feat.size(0), 1),
                        'agg_weight': f_feat.new_ones(f_feat.size(0), f_feat.size(1), 1),
                        'mask': f_feat.new_ones(f_feat.size(0), f_feat.size(1)).detach()}
        p_token_dict = {'x': p_feat, 'token_num': p_feat.size(1),
                        'idx_token': torch.arange(p_feat.size(1))[None, :].repeat(p_feat.size(0), 1),
                        'agg_weight': p_feat.new_ones(p_feat.size(0), p_feat.size(1), 1),
                        'mask': p_feat.new_ones(p_feat.size(0), p_feat.size(1)).detach()}

        ###### Step-I: See frame (s_feat & f_feat) ######
        f_token_dict = self.v_att_block_f_0(s_token_dict, self.v_fcm_f_0(f_token_dict))
        f_feat_h = f_token_dict['x']
        sims_sf_h = self.s_and_f(s_feat, f_feat_h)
        loss_sf_h = (self.loss_fct(sims_sf_h * logit_scale) + self.loss_fct(sims_sf_h.T * logit_scale)) / 2.0

        ###### Step-II: See patch (w_feat & p_feat) ######
        p_token_dict = self.v_att_block_p_1(s_token_dict, self.v_pcm_p_1(p_token_dict))
        p_token_dict = self.v_att_block_p_2(s_token_dict, self.v_pcm_p_2(p_token_dict))
        p_token_dict = self.v_att_block_p_3(s_token_dict, self.v_pcm_p_3(p_token_dict))
        p_feat_h = p_token_dict['x']
        sims_wp_h = self.w_and_p(w_feat, p_feat_h)
        loss_wp_h = (self.loss_fct(sims_wp_h * logit_scale) + self.loss_fct(sims_wp_h.T * logit_scale)) / 2.0

        ###### Step-III: Server for sampling ######
        f_token_dict = {'x': f_feat, 'token_num': f_feat.size(1),
                        'idx_token': torch.arange(f_feat.size(1))[None, :].repeat(f_feat.size(0), 1),
                        'agg_weight': f_feat.new_ones(f_feat.size(0), f_feat.size(1), 1),
                        'mask': f_feat.new_ones(f_feat.size(0), f_feat.size(1)).detach()}
        p_token_dict = {'x': p_feat, 'token_num': p_feat.size(1),
                        'idx_token': torch.arange(p_feat.size(1))[None, :].repeat(p_feat.size(0), 1),
                        'agg_weight': p_feat.new_ones(p_feat.size(0), p_feat.size(1), 1),
                        'mask': p_feat.new_ones(p_feat.size(0), p_feat.size(1)).detach()}
        f_token_dict = self.v_fcm_f_0(f_token_dict)
        f_feat = f_token_dict['x']
        sims_sf = self.s_and_f(s_feat, f_feat)
        loss_sf = (self.loss_fct(sims_sf * logit_scale) + self.loss_fct(sims_sf.T * logit_scale)) / 2.0

        p_token_dict = self.v_pcm_p_1(p_token_dict)
        p_token_dict = self.v_pcm_p_2(p_token_dict)
        p_token_dict = self.v_pcm_p_3(p_token_dict)
        p_feat = p_token_dict['x']
        sims_wp = self.w_and_p(w_feat, p_feat)
        loss_wp = (self.loss_fct(sims_wp * logit_scale) + self.loss_fct(sims_wp.T * logit_scale)) / 2.0

        ###### Step-IV: Total loss
        loss_kl_sf = (self.loss_kl(sims_sf, sims_sf_h) + self.loss_kl(sims_sf.T, sims_sf_h.T)) / 2.0
        loss_kl_wp = (self.loss_kl(sims_wp, sims_wp_h) + self.loss_kl(sims_wp.T, sims_wp_h.T)) / 2.0
        total_loss = (loss_sf_h + loss_wp_h) + (loss_sf + loss_wp) + (loss_kl_sf + loss_kl_wp)

        if self.training:
            return total_loss
        else:
            return None

    def norm(self, feat):
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat

    def s_and_f(self, s_feat, f_feat):
        s_feat = s_feat.squeeze(1)
        f_w = torch.softmax(self.f_feat_w(f_feat).squeeze(-1), dim=-1)
        sims_sf = torch.einsum("ad,bfd->abf", [self.norm(s_feat), self.norm(f_feat)])
        sims_sf = torch.einsum("abf,bf->ab", [sims_sf, f_w])
        return sims_sf

    def w_and_p(self, w_feat, p_feat):
        w_w = torch.softmax(self.w_feat_w(w_feat).squeeze(-1), dim=-1)
        p_w = torch.softmax(self.p_feat_w(p_feat).squeeze(-1), dim=-1)
        sims_wp = torch.einsum("awd,bpd->abwp", [self.norm(w_feat), self.norm(p_feat)])
        sims_w2p, _ = sims_wp.max(dim=-1)
        sims_w2p = torch.einsum('abw,aw->ab', [sims_w2p, w_w])
        sims_p2w, _ = sims_wp.max(dim=-2)
        sims_p2w = torch.einsum('abf,bf->ab', [sims_p2w, p_w])
        sims_wp = (sims_w2p + sims_p2w) / 2.0
        return sims_wp

    def get_text_feat(self, text_ids, text_mask, shaped=False):
        if shaped is False:
            text_ids = text_ids.view(-1, text_ids.shape[-1])
            text_mask = text_mask.view(-1, text_mask.shape[-1])

        bs_pair = text_ids.size(0)
        s_feat, w_feat = self.clip.encode_text(text_ids, return_hidden=True, mask=text_mask)
        s_feat = s_feat.float().view(bs_pair, -1, s_feat.size(-1))
        w_feat = w_feat.float().view(bs_pair, -1, w_feat.size(-1))
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
        f_feat, p_feat = self.clip.encode_image(video, return_hidden=True, mask=video_mask)
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

    def get_similarity_logits(self, s_feat, w_feat, w_mask, f_feat, p_feat, f_mask):
        ###### Step-III: Server for sampling ######
        f_token_dict = {'x': f_feat, 'token_num': f_feat.size(1),
                        'idx_token': torch.arange(f_feat.size(1))[None, :].repeat(f_feat.size(0), 1),
                        'agg_weight': f_feat.new_ones(f_feat.size(0), f_feat.size(1), 1),
                        'mask': f_feat.new_ones(f_feat.size(0), f_feat.size(1)).detach()}
        p_token_dict = {'x': p_feat, 'token_num': p_feat.size(1),
                        'idx_token': torch.arange(p_feat.size(1))[None, :].repeat(p_feat.size(0), 1),
                        'agg_weight': p_feat.new_ones(p_feat.size(0), p_feat.size(1), 1),
                        'mask': p_feat.new_ones(p_feat.size(0), p_feat.size(1)).detach()}
        f_token_dict = self.v_fcm_f_0(f_token_dict)
        f_feat = f_token_dict['x']
        sims_sf = self.s_and_f(s_feat, f_feat)

        p_token_dict = self.v_pcm_p_1(p_token_dict)
        p_token_dict = self.v_pcm_p_2(p_token_dict)
        p_token_dict = self.v_pcm_p_3(p_token_dict)
        p_feat = p_token_dict['x']
        sims_wp = self.w_and_p(w_feat, p_feat)

        sims = (sims_sf + sims_wp) / 2.0

        return sims

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