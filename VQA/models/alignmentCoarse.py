import torch
import torch.nn as nn

class alignmentCoarse(nn.Module):
    def __init__(self, ):
        super(alignmentCoarse, self).__init__()

        self.embed_dim = 512
        self.temp = 5

        transformer_width = self.embed_dim
        self.video_weight_fc = nn.Sequential(
            nn.Linear(transformer_width, 2 * transformer_width), nn.ReLU(inplace=True),
            nn.Linear(2 * transformer_width, 1))

    def forward(self, text_feat, video_feat):

        video_weight = self.video_weight_fc(video_feat).squeeze(2)
        video_weight = torch.softmax(video_weight, dim=-1)

        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        video_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)
        retrieve_logits = torch.einsum('ad,bvd->abv', [text_feat, video_feat])
        retrieve_logits = torch.einsum('abv,bv->ab', [retrieve_logits, video_weight])
        video_feat = torch.einsum('bfd,bf->bd', [video_feat, video_weight])

        return text_feat, video_feat, retrieve_logits

