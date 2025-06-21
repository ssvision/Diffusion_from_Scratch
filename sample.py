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

def sample(sampler, pre_trained_model, img):
    
    X_t = torch.randn_like(img)
    batch_size = img.shape[0]

    for k in tqdm(range(sampler.num_steps-1,-1,-1)):
        
        noise_pred = pre_trained_model(X_t, torch.full((batch_size,),k))
        X_t = sampler.sample_previous_step(X_t, noise_pred, k)

    return X_t #X_0






if __name__=="__main__":

    sampler = DeepDenoisingProbModel(num_steps=10, beta_min=1e-3, beta_max=0.1, device='cpu')
    pre_trained_model = Unet()

    img = torch.randn(8,3,32,32)

    og_img = sample(sampler, pre_trained_model, img)



