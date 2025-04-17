import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class alignmentMedium(nn.Module):
    def __init__(self, ):
        super(alignmentMedium, self).__init__()
        self.embed_dim = 512
        dropout = 0.3

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.linear_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.layer_norm3 = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)

        self._init_parameters()

    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'linear' in name or 'proj' in name:
                if 'weight' in name:
                    nn.init.eye_(param)
                elif 'bias' in name:
                    param.data.fill_(0.)

    def forward(self, text_embeds, video_embeds):
        text_embeds = self.layer_norm1(text_embeds)
        video_embeds = self.layer_norm1(video_embeds)

        q = self.q_proj(text_embeds)
        k = self.k_proj(video_embeds)
        v = self.v_proj(video_embeds)

        q = q.unsqueeze(1)
        k = k.transpose(1, 2)
        attention_logits = torch.matmul(q, k)
        attention_logits = attention_logits / math.sqrt(self.embed_dim)

        attention_weights = F.softmax(attention_logits, dim=2)

        attention = torch.matmul(attention_weights, v)
        attention = attention.squeeze(1)

        attention = self.out_proj(attention)
        attn_out = self.layer_norm2(attention)
        linear_out = self.linear_proj(attn_out)
        video_embeds = attn_out + self.dropout(linear_out)
        video_embeds = self.layer_norm3(video_embeds)

        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        video_embeds = video_embeds / video_embeds.norm(dim=-1, keepdim=True)
        sims = torch.mm(text_embeds, video_embeds.t())

        return sims
