"""
Adapted from: https://github.com/openai/CLIP/blob/main/clip/clip.py
"""
from collections import OrderedDict
from typing import Tuple, Union

import hashlib
import os
import urllib
import warnings
from tqdm import tqdm
from .module_transformer import Transformer as TransformerClip
import torch
import torch.nn.functional as F
from torch import nn

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt"
}
_PT_NAME = {
    "RN50": "RN50.pt",
    "RN101": "RN101.pt",
    "RN50x4": "RN50x4.pt",
    "RN50x16": "RN50x16.pt",
    "ViT-B/32": "ViT-B-32.pt",
    "ViT-B/16": "ViT-B-16.pt",
    "ViT-L/14": "ViT-L-14.pt",
}


def _download(url: str, root: str = os.path.expanduser("~/.cache/clip")):
    """
    Download a CLIP model checkpoint from a URL to the local cache directory.
    Validates the downloaded file using an embedded SHA256 checksum.
    """
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


def available_models():
    """Returns the names of available CLIP models."""
    return list(_MODELS.keys())


# =============================

class Bottleneck(nn.Module):
    """
    Standard ResNet Bottleneck block with expansion factor 4.
    Uses 1x1 -> 3x3 -> 1x1 convolutions, with optional downsampling for stride > 1.
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    """
    Multi-head attention pooling layer.
    Treats the flattened spatial grid as a sequence and pools via multi-head self-attention.
    Prepends a mean-pooled class token and adds learned positional embeddings.
    """
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super(AttentionPool2d, self).__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        # Convert NCHW feature map to (HW)NC sequence format, each spatial position as a token
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        # Prepend class token from global average pooling: (HW+1)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        # Add learnable positional embeddings
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        # Use PyTorch native multi-head attention with shared Q/K/V input
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super(ModifiedResNet, self).__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            # 3-layer stem convolutions, each followed by BN and ReLU, then average pooling
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        # Four residual stages with progressive downsampling for multi-scale features
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # Use Attention Pooling instead of global average pooling
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    """
    Faster approximation of the GELU activation function.
    Computes x * sigmoid(1.702 * x) for speed benefits on some hardware.
    """
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """
    Pre-normalized Transformer block with multi-head self-attention and MLP.
    Supports optional causal / callable attention masks for autoregressive modeling.
    """
    def __init__(self, d_model: int, n_head: int, attn_mask=None):
        super(ResidualAttentionBlock, self).__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        attn_mask_ = self.attn_mask
        if self.attn_mask is not None and hasattr(self.attn_mask, '__call__'):
            attn_mask_ = self.attn_mask(x.size(0))  # LND

        # If mask exists, ensure its dtype and device match the input tensor
        attn_mask_ = attn_mask_.to(dtype=x.dtype, device=x.device) if attn_mask_ is not None else None
        # Self-attention: Q, K, V are all the input x itself
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask_)[0]

    def forward(self, x):
        # LayerNorm before Attention, residual connection
        x = x + self.attention(self.ln_1(x))
        # LayerNorm before MLP, residual connection
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    """
    A stack of ResidualAttentionBlocks.
    Expects input shape (L, N, D) where L is sequence length and N is batch size.
    """
    def __init__(self, width: int, layers: int, heads: int, attn_mask=None):
        super(Transformer, self).__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisualTransformer(nn.Module):
    """
    Vision Transformer (ViT) for image encoding.
    Splits input image into fixed-size patches via a convolution,
    prepends a learnable class token, adds positional embeddings,
    and processes the sequence through a standard Transformer.
    """
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super(VisualTransformer, self).__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim

        # Use convolution with stride=patch_size to split image into non-overlapping patches
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        for param in self.conv1.parameters():
            param.requires_grad = False  # not update by gradient

        # self.class_embedding.requires_grad = False  # not update by gradient
        # self.positional_embedding.requires_grad = False  # not update by gradient


    def forward(self, x: torch.Tensor, mask=None):

        # Patch extraction: [B, 3, H, W] -> [B, width, grid, grid]
        x = self.conv1(x)  # shape = [*, width, grid, grid]

        # Flatten spatial dimension and transpose to [B, grid**2, width] for Transformer sequence input
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        # Insert class token at the beginning of the sequence for global image aggregation
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        # Add positional embeddings and apply pre-normalization
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        # zero = torch.zeros((x.size(1), x.size(1))).repeat(x.size(0), 1, 1).to(x.device)
        # inf = torch.zeros((x.size(1), x.size(1))).fill_(float("-inf")).repeat(x.size(0), 1, 1).to(mask.device)
        # _mask = mask.view(-1).unsqueeze(1).unsqueeze(1).expand(-1, x.size(1), x.size(1))
        # attn_mask = torch.where(_mask>0, zero, inf)
        # attn_mask[:, 0, 0] = 0

        # Adjust dimensions to (L, N, D) to match Transformer convention: L = grid**2 + 1
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        # Convert back to (N, L, D)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # mask = mask.view(-1).unsqueeze(1).unsqueeze(1).expand(-1, x.size(1), x.size(2))
        # zero = torch.zeros((x.size(0), x.size(1), x.size(2))).to(x.device).type(x.dtype)
        # x = torch.where(mask>0, x, zero)
        # Move the three lines below to `encode_image` for entire hidden sequence
        # x = self.ln_post(x[:, 0, :])
        # if self.proj is not None:
        #     x = x @ self.proj

        return x


class CLIP(nn.Module):
    """
    Main CLIP model combining a visual encoder and a text encoder.
    Projects images and text into a shared embedding space for contrastive learning.
    """
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super(CLIP, self).__init__()

        self.context_length = context_length

        # Choose ResNet or ViT as visual encoder based on vision_layers type
        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisualTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
            )

        self.transformer = TransformerClip(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            # attn_mask=self.build_attention_mask
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]))

        self.initialize_parameters()

        # for param in self.transformer.parameters():
        #     param.requires_grad = False  # not update by gradient
        self.token_embedding.requires_grad = False  # not update by gradient
        # self.positional_embedding.requires_grad = False  # not update by gradient

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    @staticmethod
    def get_config(pretrained_clip_name="ViT-B/32"):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ViT-B-32.pt")
        if pretrained_clip_name in _MODELS and pretrained_clip_name in _PT_NAME:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _PT_NAME[pretrained_clip_name])

        if pretrained_clip_name in ["ViT-B/32", "ViT-B/16"] and os.path.exists(model_path):
            pass
        else:
            if pretrained_clip_name in _MODELS:
                model_path = _download(_MODELS[pretrained_clip_name])
            elif os.path.isfile(pretrained_clip_name):
                model_path = pretrained_clip_name
            else:
                raise RuntimeError(f"Model {pretrained_clip_name} not found; available models = {available_models()}")

        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = model.state_dict()
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        return state_dict

    def build_attention_mask(self, context_length):
        """
        Build a causal (lower-triangular) attention mask for text encoding.
        Tokens can only attend to themselves and previous tokens; future positions are masked.
        """
        # Build causal attention mask: fill upper triangle (excluding diagonal) with -inf to mask future tokens
        mask = torch.zeros(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # Keep diagonal and lower triangle as 0, mask upper triangle
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, return_hidden=False, mask=None):
        """
        Encode a batch of images into the shared embedding space.
        Uses the visual encoder (ResNet or ViT) and returns the CLS token embedding.
        If return_hidden is True, also returns the rest of the patch-level tokens.
        """
        # Image encoding: get patch/token level features from visual encoder
        hidden = self.visual(image.type(self.dtype))
        # Post-normalize and project to shared semantic space: [B, num_tokens, embed_dim]
        hidden = self.visual.ln_post(hidden) @ self.visual.proj

        # Take the first token (CLS token) as the global image feature
        x = hidden[:, 0, :]
        # Remaining tokens correspond to local patch features for fine-grained alignment
        y = hidden[:, 1:, :]

        if return_hidden:
            return x, y

        return x

    def encode_text(self, text, return_hidden=False, mask=None):
        """
        Encode a batch of tokenized text into the shared embedding space.
        Uses causal self-attention to ensure tokens only attend to previous positions.
        Extracts the representation at the end-of-text (EOT) token as the sentence embedding.
        """
        # Text encoding: map token ids to embedding vectors [batch_size, n_ctx, d_model]
        x = self.token_embedding(text).type(self.dtype)

        # Slice positional embeddings to match current sequence length
        pos_emd = self.positional_embedding[:x.size(1), :].type(self.dtype)

        # Build causal mask and broadcast to batch dimension: [B, n_ctx, n_ctx]
        attn_mask = self.build_attention_mask(x.size(1)).repeat(x.size(0), 1, 1).to(mask.device)
        inf = torch.zeros((x.size(1), x.size(1))).fill_(float("-inf")).repeat(x.size(0), 1, 1).to(mask.device)
        # Combine with external mask (e.g. padding mask), invalid positions set to -inf
        mask = mask.unsqueeze(1).expand(-1, mask.size(1), -1)
        attn_mask = torch.where(mask > 0, attn_mask, inf)

        # Add positional embeddings
        x = x + pos_emd
        # Convert to (L, N, D) format required by Transformer
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass through text Transformer with causal mask for autoregressive attention
        x = self.transformer(x, attn_mask)
        # Convert back to (N, L, D)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # Normalize last layer output and project to shared semantic space
        hidden = self.ln_final(x).type(self.dtype) @ self.text_projection

        # Extract features at EOT (end-of-text) token position as sentence representation
        # EOT token is the largest token id in the sequence (appended by tokenizer)
        x = hidden[torch.arange(hidden.shape[0]), text.argmax(dim=-1)]

        if return_hidden:
            return x, hidden

        return x

    def forward(self, image, text):
        """
        Compute contrastive similarity logits between images and text.
        Returns two logit matrices: image-to-text and text-to-image.
        """
        # Encode image and text into fixed-dimensional feature vectors
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # L2 normalize to unit sphere so dot product equals cosine similarity
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute cosine similarity matrix across global batch, scaled by learnable logit_scale
        # logits_per_image[i, j] is the similarity between i-th image and j-th text
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16."""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)
