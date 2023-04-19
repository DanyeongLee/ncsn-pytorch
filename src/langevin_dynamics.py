from tqdm import tqdm
import torch

@torch.no_grad()
def langevin_step(model, x, step_size, noise_idx):
    score = model(x, noise_idx)
    noise = torch.randn_like(x)
    x = x + step_size * score / 2 + torch.sqrt(step_size) * noise
    torch.cuda.empty_cache()
    return x

@torch.no_grad()
def langevin_dynamics(model, x, step_size, noise_idx, n_steps=100):
    for i in tqdm(range(n_steps)):
        x = langevin_step(model, x, step_size, noise_idx)
    return x

@torch.no_grad()
def annealed_langevin_dynamics(model, x, noise_scales, n_steps=100, eps=2e-5):
    bsz = x.size(0)
    for i in tqdm(range(len(noise_scales))):
        noise_idx = torch.ones(bsz, dtype=torch.long) * i
        step_size = eps * (noise_scales[i] / noise_scales[-1]) ** 2
        x = langevin_dynamics(model, x, step_size, noise_idx, n_steps)
    return x


@torch.no_grad()
def sample(model, shape, noise_scales, device, n_steps=100, eps=2e-5):
    model.eval()
    x = torch.rand(shape).to(device)
    x = annealed_langevin_dynamics(model, x, noise_scales, n_steps, eps)
    return x