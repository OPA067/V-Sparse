import torch
from torch import nn
import torch.nn.functional as F

class PerturbedTopK(nn.Module):
    def __init__(self, k: int, num_samples: int=500, sigma: float=0.05):
        super().__init__()
        self.num_samples = num_samples
        self.sigma = sigma
        self.k = k
    
    def __call__(self, x):
        return PerturbedTopKFuntion.apply(x, self.k, self.num_samples, self.sigma)

class PerturbedTopKFuntion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k: int, num_samples: int=500, sigma: float=0.05):
        # input here is scores with (bs, num_patches)
        b, d = x.shape
        noise = torch.normal(mean=0.0, std=1.0, size=(b, num_samples, d)).to(dtype=x.dtype, device=x.device)
        perturbed_x = x.unsqueeze(1) + noise*sigma # b, nS, d
        topk_results = torch.topk(perturbed_x, k=k, dim=-1, sorted=False)
        indices = topk_results.indices # b, nS, k
        indices = torch.sort(indices, dim=-1).values # b, nS, k

        perturbed_output = F.one_hot(indices, num_classes=d).float() # b, nS, k, d
        indicators = perturbed_output.mean(dim=1) # b, k, d

        # context for backward
        ctx.k = k
        ctx.num_samples = num_samples
        ctx.sigma = sigma

        ctx.perturbed_output = perturbed_output
        ctx.noise = noise

        return indicators
    
    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return tuple([None]*5)
        
        noise_gradient = ctx.noise
        expected_gradient = (
            torch.einsum("bnkd,bnd->bkd", ctx.perturbed_output, noise_gradient)
            / ctx.num_samples
            / ctx.sigma
        )
        grad_input = torch.einsum("bkd,bkd->bd", grad_output, expected_gradient)
        return (grad_input,) + tuple([None]*5)

class PredictorLG(nn.Module):
    """ Image to Patch Embedding """
    def __init__(self, embed_dim=512):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim//2, bias=False),
            nn.GELU()
        )

        self.out_conv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2, bias=False),
            nn.GELU(),
            # nn.Linear(embed_dim // 2, embed_dim // 4, bias=False),
            # nn.GELU(),
            nn.Linear(embed_dim // 2, 1, bias=False),
            nn.Tanh()
            # nn.Sigmoid()
            # nn.Softmax(dim=-1)
            # nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        ''' x: shape (bs*n_length, num_tokens, hid_dim) '''
        x = self.in_conv(x)
        B, N, C = x.size()
        local_x = x[:, :, :]
        global_x = x[:, :1, :]
        x = torch.cat([local_x, global_x.expand(B, N, C)], dim=-1)
        return self.out_conv(x)

class PatchTokenSelection(nn.Module):
    def __init__(self, max_frames, embed_dim=512, topk=3):
        super().__init__()
        self.max_frames = max_frames
        self.score_predictor = PredictorLG(embed_dim=embed_dim)
        self.topk_selector = PerturbedTopK(topk)
    
    def forward(self, x, training=True):
        '''
        x: input embed, shape is [b, f*p, d]
        use cls token as global representation
        prob = Tanh(MLP(x))
        '''
        
        B, L, D = x.shape
        N = L // self.max_frames
        x = x.reshape(B, -1, N, D)  # [b, f, p ,d]
        x = x.reshape(-1, N, D)     # [b*f, p, d]
        pred_score = self.score_predictor(x).squeeze()  # [b*f, p]

        spatial_pred_score = pred_score[:, 1:]  # Separate the cls_token [b*f, p-1]
        topk_indicator = self.topk_selector(spatial_pred_score)     # [b*f, k, p-1]

        cls_x_feature = x[:, :1, :]         # cls_token of [b*f, 1, d]

        spatial_x_feature = x[:, 1:, :]     # patch_token of [b*f, p-1, d]
        selected_patch_feature = torch.einsum("bkp,bpd->bkd", topk_indicator, spatial_x_feature)    # [b*f, k, d]

        output = torch.cat((cls_x_feature, selected_patch_feature), dim=1)  # [b*f, k+1, d]
        output = output.reshape(B, self.max_frames, -1, D).reshape(B, -1, D)       # [b, f*(k+1), d]

        return output

p = torch.rand(32, 300, 512)

model_1 = PatchTokenSelection(max_frames=12, embed_dim=512, topk=12)
model_2 = PatchTokenSelection(max_frames=12, embed_dim=512, topk=6)
model_3 = PatchTokenSelection(max_frames=12, embed_dim=512, topk=3)

a = model_3(model_2(model_1(p)))
print(a.shape)
