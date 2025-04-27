import os
import time
from tqdm import tqdm

import cv2
import einops
import numpy as np
import torch
import torch.nn as nn

from dataset import download_dataset, get_dataloader, get_mnist_tensor_shape
from simple_diffusion_model import DDPM
from denoising_model import UNet


batch_size = 512
epochs = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
output_dir = 'out'
sampler = None
n_steps = 1000
model_name = 'model_unet_res.pth'
train_model = True
dataset_name = 'mnist'



def train(sampler, net, device:str):
    n_steps = sampler.n_steps

    dataset, dir_path = download_dataset(dataset_name)
    dataloader = get_dataloader(batch_size, dir_path)

    net = net.to(device)
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    tic = time.time()
    for epochi in tqdm(range(epochs)):
        total_loss = 0
        for x, _ in dataloader:  # x_0 ~ q(x_0)
            B = x.shape[0]
            x = x.to(device)
            t = torch.randint(0, n_steps, (B, )).to(device)  # t ~ Uniform({1, ..., T})
            eps = torch.randn_like(x).to(device)
            x_t = sampler.forward_process(x, t, eps)
            eps_theta = net(x_t, t.reshape(B, 1))
            loss = mse_loss(eps_theta, eps)  # MSE(eps - eps_theta)
            optimizer.zero_grad()
            loss.backward()  # gradient descent step
            optimizer.step()
            total_loss += loss.item() * B
        total_loss /= len(dataloader.dataset)
        toc = time.time()
        torch.save(net.state_dict(), f"{output_dir}/{model_name}")
        print(f'epoch {epochi}, avg loss {total_loss}, time elapsed {(toc - tic):.2f}s')


def generate_images(sampler, net, output_path:str, device:str, n_sample_per_side:int = 10):
    net = net.to(device).eval()
    with torch.no_grad():
        shape = (int(n_sample_per_side**2), *get_mnist_tensor_shape())
        samples = sampler.sample_backward(net, shape).detach()  # generate samples 
        samples = ((samples + 1) / 2) * 255 
        samples = samples.clamp(0, 255)
        samples = einops.rearrange(samples, '(b1 b2) c h w -> (b1 h) (b2 w) c', b1=n_sample_per_side)  # arrange samples to a square image
        image = samples.cpu().numpy().astype(np.uint8)  # default image coding
        cv2.imwrite(output_path, image)  # save the image


if __name__ == '__main__':
    net = UNet(n_steps)
    if train_model:
        sampler = DDPM(n_steps, 0.0001, 0.02, device)
        # os.makedirs(f"{output_dir}/", exist_ok=True)
        train(sampler, net, device)
    else:
        net.load_state_dict(torch.load(f"{output_dir}/{model_name}"))
    
    generate_images(sampler, net, f"{output_dir}/generate.png", device)