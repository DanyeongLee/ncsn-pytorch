import torch
import torch.nn.functional as F


def linear_noise_scale(start=1., end=0.01, length=10):
    return torch.linspace(start, end, length)


def q_sample(x, sigma, noise=None):
    # x: (B, C, H, W)
    # sigma: (B, )
    if noise is None:
        noise = torch.randn_like(x)
    while sigma.dim() < x.dim():
        sigma = sigma.unsqueeze(-1)
    return x + sigma * noise


def score_matching_loss(model, x, noise_scales):
    noise_scale_idx = torch.randint(0, noise_scales.shape[0], (x.shape[0],), device=x.device)
    noise_scale_batch = noise_scales[noise_scale_idx].view(-1, 1, 1, 1)

    noise = torch.randn_like(x)
    x_noisy = q_sample(x, noise_scale_batch, noise=noise)
    score = model(x_noisy, noise_scale_idx)

    loss = F.mse_loss(noise_scale_batch * score, -noise)

    return loss