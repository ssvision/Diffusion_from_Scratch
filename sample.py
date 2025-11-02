#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
This script is used to denoise process Noise -> Images
This requires a pre-trained model(like U-Net) and a sampler
'''
import torch, torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import time
import itertools
import torchvision

from dataset import download_dataset, get_dataloader
from alternate_diffusion_model import DeepDenoisingProbModel
from UNet import Unet

'''..................................................................'''

minibatch_size  = 8         
device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset_name = 'cifar10'

# Setup dataset
dataset, dir_path, _ = download_dataset(dataset_name)
dataloader = get_dataloader(minibatch_size, dir_path)
img, _ = next(iter(dataloader))

def sample(sampler, pre_trained_model, img, device):
    
    X_t = torch.randn_like(img).to(device)
    batch_size = img.shape[0]

    for k in tqdm(range(sampler.num_steps-1,-1,-1)):
    
        t_step = torch.full((batch_size,),k).to(device)
        
        noise_pred = pre_trained_model(X_t, t_step)
        X_t = sampler.sample_previous_step(X_t, noise_pred, k)

    return X_t #X_0






if __name__=="__main__":

    sampler = DeepDenoisingProbModel(num_steps=100, beta_min=1e-3, beta_max=0.1, device=device)
    pre_trained_model = Unet()
    pre_trained_model.to(device)
    checkpoint = torch.load('./model/cifar102025-11-02', map_location=torch.device(device))
    pre_trained_model.load_state_dict(checkpoint)
    print(f"✅ Weights loaded from pretrained model")

    img = torch.randn(8,3,32,32).to(device)

    pre_trained_model.eval()

    with torch.no_grad():
        og_img = sample(sampler, pre_trained_model, img, device)

    og_img = og_img.clamp(-1, 1)
    # Make a grid of images (normalize=True scales images to [0,1] for display)
    grid = torchvision.utils.make_grid(og_img, nrow=4, normalize=True, padding=2)

    # Convert to numpy and permute dimensions for matplotlib (C,H,W -> H,W,C)
    np_grid = grid.permute(1, 2, 0).cpu().numpy()

    # Plot using matplotlib
    plt.figure(figsize=(6, 6))
    plt.imshow(np_grid)
    plt.axis('off')
    plt.title("Minibatch of 8 Images")
    plt.show()

