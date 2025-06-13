#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Code to visualise noise addition
'''


import torch, numpy as np
from dataset import *
from simple_diffusion_model import DDPM
from alternate_diffusion_model import DeepDenoisingProbModel
import einops, cv2



def visualize_noise_addition(n_steps, output_dir, device):

    # Write your dataset name here e.g MNIST, CIFAR, etc
    dataset_name = 'mnist'
    # dataset_name = 'cifar10'

    dataset, dir_path = download_dataset(dataset_name)

    batch_size = 10
    dataset_dataloader = get_dataloader(batch_size, dir_path)

    # adding noise from 0 to 99% in 30 steps
    noise_percents = torch.linspace(0,0.999, 30)

    X, _ = next(iter(dataset_dataloader))
    X = X.to(device)

    ddpm_sampler = DeepDenoisingProbModel(n_steps, 0.0001, 0.02, device)
    x_ts = []
    

    for noise_percent in noise_percents:

        # Calculate the timestep t based on the noise percentage
        t = torch.tensor([int(n_steps * noise_percent)])
        t = torch.full((batch_size,), t.item())
       
        # print(f"Shape of t is : {t.shape} and the tensor 't' is : {t}")
        
        # Apply forward sampling to add noise to the images
        x_t = ddpm_sampler.fwd_process(X, t)
        x_ts.append(x_t)

    # print(f"CheckPoint1")
    # Stack the noisy images into a single tensor
    x_ts = torch.stack(x_ts, 0)
    print(f"Shape of x_ts is : {x_ts.shape}")



    # Normalize the pixel values to [0, 255] range
    x_ts = ((x_ts + 1) / 2) * 255
    x_ts = x_ts.clamp(0, 255)
    print(f"Shape of x_ts is : {x_ts.shape}")


    # Rearrange the tensor to create a grid of images
    # n1: number of noise levels, n2: number of images, c: channels, h: height, w: width
    x_ts = einops.rearrange(x_ts, 'n1 n2 c h w -> (n2 h) (n1 w) c')
    print(f"Shape of x_ts after einops is : {x_ts.shape}")


    #Convert the tensor to a numpy array and change data type to uint8
    image = x_ts.cpu().numpy().astype(np.uint8)
        
    # Create the output directory for this sampler if it doesn't exist
    os.makedirs(f"{output_dir}", exist_ok=True)
        
    # Save the visualization as a PNG image
    cv2.imwrite(f"{output_dir}/adding_noise_mnist_new.png", image)

def visualize_noise_add(n_steps, output_dir, device):

    # Write your dataset name here e.g MNIST, CIFAR, etc
    dataset_name = 'mnist'
    # dataset_name = 'cifar10'

    dataset, dir_path = download_dataset(dataset_name)

    batch_size = 10
    dataset_dataloader = get_dataloader(batch_size, dir_path)

    noise_percents = torch.linspace(0,0.999, 30)

    X, _ = next(iter(dataset_dataloader))
    X = X.to(device)

    ddpm_sampler = DDPM(n_steps, 0.0001, 0.02, device)
    x_ts = []
    

    for noise_percent in noise_percents:

        # Calculate the timestep t based on the noise percentage
        t = torch.tensor([int(n_steps * noise_percent)]).unsqueeze(1)

         # Apply forward sampling to add noise to the images
        x_t = ddpm_sampler.forward_process(X, t)
        x_ts.append(x_t)

    
    # Stack the noisy images into a single tensor
    x_ts = torch.stack(x_ts, 0)


    # Normalize the pixel values to [0, 255] range
    x_ts = ((x_ts + 1) / 2) * 255
    x_ts = x_ts.clamp(0, 255)


    # Rearrange the tensor to create a grid of images
    # n1: number of noise levels, n2: number of images, c: channels, h: height, w: width
    x_ts = einops.rearrange(x_ts, 'n1 n2 c h w -> (n2 h) (n1 w) c')


    #Convert the tensor to a numpy array and change data type to uint8
    image = x_ts.cpu().numpy().astype(np.uint8)
        
    # Create the output directory for this sampler if it doesn't exist
    os.makedirs(f"{output_dir}", exist_ok=True)
        
    # Save the visualization as a PNG image
    cv2.imwrite(f"{output_dir}/adding_noise_mnist.png", image)


if __name__ == "__main__":
    n_steps = 1000
    output_dir = 'out'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    visualize_noise_addition(n_steps, output_dir, device)
