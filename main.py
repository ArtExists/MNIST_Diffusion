import torch
import numpy
import deepinv
from torchvision import datasets, transforms
from tqdm import tqdm

dev='cuda'
batch_size=16
img_siz=32

transform=transforms.Compose([
    transforms.Resize(img_siz),
    transforms.ToTensor(),
    transforms.Normalize((0.0,),(1.0,)),

])
train_loader=torch.utils.data.DataLoader(
    datasets.MNIST(root="./data",train=True,download=True,transform=transform),
    batch_size=batch_size,
    shuffle=True,

)

lr=1e-4
epochs=25

modelwa = deepinv.models.DiffUNet(in_channels=1, out_channels=1, pretrained=None).to(dev)

optim=torch.optim.Adam(modelwa.parameters(),lr=lr)
losswa=deepinv.loss.MSE()


beta_start=1e-4
beta_end=0.02
timesteps=250

betas=torch.linspace(beta_start,beta_end, timesteps, device=dev)
alphas=1.0-betas
alphas_cp=torch.cumprod(alphas,dim=0)
sqrt_acp=torch.sqrt(alphas_cp)
sqrt_omacp=torch.sqrt(1.0-alphas_cp)
print(next(modelwa.parameters()).device)

scaler = torch.cuda.amp.GradScaler()
for epoch in tqdm(range(epochs), desc='Training Epochs'):
    el=0
    modelwa.train()
    torch.cuda.empty_cache()

    for data, _ in tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False):
        imgs=data.to(dev)
        noise=torch.randn_like(imgs)
        t=torch.randint(0,timesteps,(imgs.size(0),), device=dev)
        sqrt_acp_t = sqrt_acp[t].view(-1, 1, 1, 1)
        sqrt_omacp_t = sqrt_omacp[t].view(-1, 1, 1, 1)

        noised_imgs=(
            sqrt_acp_t*imgs+sqrt_omacp_t*noise
        )
        optim.zero_grad()

        estim_n=modelwa(noised_imgs, t, type_t='timestep')
        loss=torch.nn.functional.mse_loss(estim_n,noise,reduction='mean')
        loss.backward()
        optim.step()

        el += loss.item()
    print(f"Epoch [{epoch}/{epochs}] - Avg Loss: {el / len(train_loader):.6f}")
torch.save(modelwa.state_dict(), "trained_diff.pth")
print(next(modelwa.parameters()).device)


"""Train over pred startsss           """


import matplotlib.pyplot as plt
modelwa.load_state_dict(torch.load("trained_diff.pth"))
modelwa.eval()
n_samples = 4
img_size = (1, 32, 32)
timesteps = 250
betas = torch.linspace(1e-4, 0.02, timesteps, device=dev)
alphas = 1.0 - betas
alphas_cp = torch.cumprod(alphas, dim=0)
sqrt_1_over_alpha = torch.sqrt(1.0 / alphas)
sqrt_1_over_acp = torch.sqrt(1.0 / alphas_cp)
one_minus_acp = 1.0 - alphas_cp
x = torch.randn(n_samples, *img_size).to(dev)

for t in reversed(range(timesteps)):
    t_tensor = torch.full((n_samples,), t, dtype=torch.long, device=dev)

    with torch.no_grad():
        pred_noise = modelwa(x, t_tensor, type_t='timestep')

    alpha_t = alphas[t]
    alpha_cp_t = alphas_cp[t]
    beta_t = betas[t]
    x_0_pred = (x - torch.sqrt(1 - alpha_cp_t) * pred_noise) / torch.sqrt(alpha_cp_t)
    if t > 0:
        noise = torch.randn_like(x)
        x = torch.sqrt(alpha_t) * x_0_pred + torch.sqrt(beta_t) * noise
    else:
        x = x_0_pred

samples = x.clamp(0, 1).cpu()
fig, axes = plt.subplots(1, n_samples, figsize=(10, 3))
for i in range(n_samples):
    axes[i].imshow(samples[i][0], cmap="gray")
    axes[i].axis("off")
plt.tight_layout()
plt.show()
