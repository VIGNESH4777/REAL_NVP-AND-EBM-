import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# --- 1. DATASET PREPARATION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToTensor()])

train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)

# --- 2. REALNVP ARCHITECTURE ---
class CouplingLayer(nn.Module):
    def __init__(self, dim, hid_dim, mask):
        super().__init__()
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.s_net = nn.Sequential(nn.Linear(dim, hid_dim), nn.ReLU(), nn.Linear(hid_dim, dim), nn.Tanh())
        self.t_net = nn.Sequential(nn.Linear(dim, hid_dim), nn.ReLU(), nn.Linear(hid_dim, dim))

    def forward(self, x, reverse=False):
        x1 = x * self.mask
        s = self.s_net(x1) * (1 - self.mask)
        t = self.t_net(x1) * (1 - self.mask)
        if not reverse:
            y = x1 + (1 - self.mask) * (x * torch.exp(s) + t)
            return y, s.sum(dim=1)
        else:
            return x1 + (1 - self.mask) * (x - t) * torch.exp(-s)

class RealNVP(nn.Module):
    def __init__(self, dim=784, hid_dim=256, n_layers=6):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            mask = torch.arange(dim) % 2
            if i % 2 == 1: mask = 1 - mask
            self.layers.append(CouplingLayer(dim, hid_dim, mask.float()))

    def forward(self, x):
        log_det_jac = 0
        for layer in self.layers:
            x, ldj = layer(x)
            log_det_jac += ldj
        return x, log_det_jac

    def sample(self, num_samples):
        z = torch.randn(num_samples, 784).to(device)
        for layer in reversed(self.layers):
            z = layer(z, reverse=True)
        return z

# --- 3. ENERGY-BASED MODEL ARCHITECTURE ---
class EBM(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )
    def forward(self, x): return self.net(x)

def sgld_sample(model, x, steps=30, lr=10.0):
    model.eval()
    for _ in range(steps):
        x.requires_grad = True
        out = model(x)
        grad = torch.autograd.grad(out.sum(), x)[0]
        x.data = x.data - 0.5 * lr * grad + torch.randn_like(x) * 0.005
        x = x.detach().clamp(0, 1)
    return x

# --- 4. TRAINING LOGIC ---
def train():
    # Setup Models
    nvp = RealNVP().to(device)
    ebm = EBM().to(device)
    nvp_opt = optim.Adam(nvp.parameters(), lr=1e-3)
    ebm_opt = optim.Adam(ebm.parameters(), lr=1e-4)

    print("Starting Training (2 Epochs for Demo)...")
    for epoch in range(2):
        pbar = tqdm(train_loader)
        for imgs, _ in pbar:
            imgs = imgs.view(-1, 784).to(device)

            # Train RealNVP (Maximum Likelihood)
            nvp_opt.zero_grad()
            z, ldj = nvp(imgs)
            log_pz = -0.5 * (z**2 + np.log(2 * np.pi)).sum(dim=1)
            nvp_loss = -(log_pz + ldj).mean()
            nvp_loss.backward()
            nvp_opt.step()

            # Train EBM (Contrastive Divergence)
            ebm_opt.zero_grad()
            fake_imgs = torch.rand_like(imgs).to(device)
            fake_imgs = sgld_sample(ebm, fake_imgs)
            ebm_loss = ebm(imgs).mean() - ebm(fake_imgs).mean()
            ebm_loss.backward()
            ebm_opt.step()

            pbar.set_description(f"NVP Loss: {nvp_loss.item():.2f} | EBM Loss: {ebm_loss.item():.2f}")

    # --- 5. VISUALIZATION ---
    nvp.eval(); ebm.eval()
    nvp_samples = nvp.sample(8).view(-1, 28, 28).detach().cpu()
    ebm_init = torch.rand(8, 784).to(device)
    ebm_samples = sgld_sample(ebm, ebm_init).view(-1, 28, 28).detach().cpu()

    fig, axes = plt.subplots(2, 8, figsize=(12, 4))
    for i in range(8):
        axes[0, i].imshow(nvp_samples[i], cmap='gray'); axes[0, i].axis('off')
        axes[1, i].imshow(ebm_samples[i], cmap='gray'); axes[1, i].axis('off')
    axes[0, 0].set_title("RealNVP Samples", loc='left')
    axes[1, 0].set_title("EBM Samples", loc='left')
    plt.show()

if __name__ == "__main__":
    train()