#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
This script is is used to setup and train the diffusion model
for the reverse process of denoising Noise -> Images
'''
import torch, torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import time
import itertools
import os
import datetime

from dataset import download_dataset, get_dataloader
from alternate_diffusion_model import DeepDenoisingProbModel
from UNet import Unet

# Setup Hyperparameters for training

minibatch_size = 32 # use smaller values to test use powers of 2
num_epochs = 5 #use higher for training

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset_name = 'cifar10'

'''Train the model'''
def train_model(sampler, denoising_model, device:str):

    num_steps = sampler.num_steps

    # Setup dataset
    dataset, dir_path, _ = download_dataset(dataset_name)
    dataloader = get_dataloader(minibatch_size, dir_path)

    # Setup Model
    denoising_model = denoising_model.to(device)
    loss_func = nn.MSELoss()
    optim = torch.optim.AdamW(denoising_model.parameters(), lr=1e-4, weight_decay=0.1)

    tic = time.time()
    # Training Loop
    for epochi in tqdm(range(num_epochs)):

        batch_Loss = []
        epoch_Loss = []

        for X, _ in itertools.islice(dataloader, None, 15): # for early stopping of batches
            
            batch_size = X.shape[0]
            X = X.to(device)

            # Setup and execute the fwd noising process
            timestep = torch.randint(0, num_steps, (batch_size, )).to(device)  # t ~ Uniform({1, ..., T})
            eps = torch.rand_like(X).to(device)
            X_t = sampler.fwd_process(X, timestep, eps)

            # Run forward pass through the denoising model
            eps_theta = denoising_model(X_t, timestep)

            #loss, optim, backpropogate
            loss = loss_func(eps_theta, eps)
            optim.zero_grad()
            loss.backward()
            optim.step()
            batch_Loss.append(loss.item())
        # print(f"Len of batch is : {len(batch_Loss)}")
        epoch_Loss = np.mean(batch_Loss)
        toc = time.time()
        print('Finished epoch:{} | Loss:{:.4f} | Time Elapsed:{:.2f}'.format(epochi,np.mean(epoch_Loss), toc-tic))

    '''Save the model'''
    model_save_path = 'model'
    model_name = str(datetime.date.today())
    os.makedirs(model_save_path, exist_ok=True)
    torch.save(denoising_model.state_dict(), f"{model_save_path}/{dataset_name+model_name}")






if __name__ == "__main__":

    sampler = DeepDenoisingProbModel(num_steps=10, beta_min=0.001, beta_max=0.1, device='cuda')
    denoising_model = Unet()


    train_model(sampler, denoising_model, device)



