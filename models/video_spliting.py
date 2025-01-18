import torch
import torch.nn as nn

class video_spliting(nn.Module):
    def __init__(self,):
        super(video_spliting, self).__init__()

        self.embed_dim = 512
        self.center = 8
        self.temp = 5

        self.linear_layer_text = nn.Linear(self.embed_dim, self.center * (self.embed_dim // self.center))
        self.linear_layer_video = nn.Linear(self.embed_dim, self.center * (self.embed_dim // self.center))

        transformer_width = self.embed_dim
        self.video_weight_fc = nn.Sequential(
            nn.Linear(transformer_width, 2 * transformer_width), nn.ReLU(inplace=True),
            nn.Linear(2 * transformer_width, 1))

        width = int(self.embed_dim // self.center)
        self.weight_fc = nn.Sequential(
            nn.Linear(2 * width, 4 * width),
            nn.ReLU(inplace=True),
            nn.Linear(4 * width, 1))

    def forward(self, text_feat, video_feat):

        video_weight = self.video_weight_fc(video_feat).squeeze(2)
        video_weight = torch.softmax(video_weight, dim=-1)

        video_feat = torch.einsum('bfd,bf->bd', [video_feat, video_weight])

        text_feat = text_feat.view(text_feat.shape[0], self.center, -1)
        video_feat = video_feat.view(video_feat.shape[0], self.center, -1)

        temp = torch.cat([text_feat, video_feat], dim=-1)

        weight = self.weight_fc(temp).squeeze(2)

        _t_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        _v_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)

        retrieve_logits = torch.einsum('acd,bcd->abc', [_t_feat, _v_feat])

        retrieve_logits = torch.einsum('abc,ac->ab', [retrieve_logits, weight])

        return retrieve_logits






