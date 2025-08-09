import math
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadedAttention(nn.Module):
    def __init__(self, embed_dim):
        super(MultiHeadedAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = 1
        assert self.embed_dim % self.num_heads == 0
        self.head_dim = self.embed_dim // self.num_heads

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, s_feat, f_feat):

        a, _ = s_feat.shape
        q = self.q_proj(s_feat)
        q = q.reshape(a, self.num_heads, self.head_dim)     # [A, H, D//H]
        q = q.permute(1, 2, 0)                              # [H, D//H, A]

        b, f, _ = f_feat.shape
        k = self.k_proj(f_feat)
        k = k.reshape(b, f, self.num_heads, self.head_dim)  # [B, F, H, D//H]
        k = k.permute(0, 2, 1, 3)                           # [B, H, F, D//H]

        v = self.v_proj(f_feat)
        v = v.reshape(b, f, self.num_heads, self.head_dim)  # [B, F, H, D//H]
        v = v.permute(0, 2, 3, 1)                           # [B, H, D//H, F]

        attention_logits = k @ q                            # [B, H, F, A]
        attention_logits = attention_logits / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_logits, dim=2)

        attention = v @ attention_weights                   # [B, H, D//H, A]
        attention = attention.permute(0, 3, 1, 2)           # [B, A, H, D//H]
        attention = attention.reshape(b, a, self.embed_dim) # [B, A, D]

        o = self.out_proj(attention)
        return o

class CAM(nn.Module):
    def __init__(self, embed_dim, dropout):
        super(CAM, self).__init__()
        self.embed_dim = embed_dim
        self.dropout = dropout

        self.cross_attn = MultiHeadedAttention(embed_dim)

        self.linear_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.ln1 = nn.LayerNorm(self.embed_dim)
        self.ln2 = nn.LayerNorm(self.embed_dim)
        self.ln3 = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(self.dropout)

        self._init_parameters()

    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'linear' in name or 'proj' in name:
                if 'weight' in name:
                    nn.init.eye_(param)
                elif 'bias' in name:
                    param.data.fill_(0.)

    def forward(self, s_feat, f_feat):

        s_feat = self.ln1(s_feat)
        f_feat = self.ln2(f_feat)

        attn_out = self.cross_attn(s_feat, f_feat)
        attn_out = self.ln2(attn_out)

        linear_out = self.linear_proj(attn_out)
        out = attn_out + self.dropout(linear_out)
        out = self.ln3(out)

        return out