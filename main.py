from tqdm import tqdm
import torch
import torch.nn as nn

from torchvision.datasets import CelebA, MNIST
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader


from src.model import RefineNet
from src.score_matching import linear_noise_scale, score_matching_loss


'''train_dataset = CelebA(root='../../data', split='train', download=True,
                          transform=transforms.Compose([
                                transforms.CenterCrop(140),
                                transforms.Resize(32),
                                transforms.ToTensor()
                            ]))'''


bsz = 128
device = torch.device('cuda:1')
noise_steps = 10


train_dataset = MNIST(root='../../data', train=True, download=True, transform=transforms.ToTensor())
train_dataloader = DataLoader(train_dataset, batch_size=bsz, shuffle=True, num_workers=8, drop_last=True)
model = RefineNet(
    in_channels=1,
    hidden_channels=(64, 128, 256, 512),
    n_noise_scale=noise_steps
).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
noise_scales = linear_noise_scale(start=1., end=0.01, length=10).to(device)


for epoch in range(50):
    print(f'Epoch {epoch}')
    epoch_loss = 0.
    for i, (x, _) in enumerate(tqdm(train_dataloader)):
        optimizer.zero_grad()
        x = x.to(device)
        loss = score_matching_loss(model, x, noise_scales)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f'Epoch loss: {epoch_loss / len(train_dataloader)}')
torch.save(model.state_dict(), 'ckpts/refinenet_mnist.pth')