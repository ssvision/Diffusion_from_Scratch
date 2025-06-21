#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Script related to downloading, managing and converting datasets into tensors
'''

import os
import torchvision
from torch.utils.data import DataLoader
import torchvision.datasets
from torchvision.transforms import Compose, Lambda, ToTensor



def download_dataset(dataset_name : str):

    dir_path = 'data/' + dataset_name
    os.makedirs(dir_path, exist_ok=True)

    # Write your dataset name here e.g MNIST, CIFAR, etc
    # dataset = torchvision.datasets.MNIST(root=dir_path, download=True)
    dataset = torchvision.datasets.CIFAR10(root=dir_path, download=True)

    print(f"Dataset downloaded sucess with size : {len(dataset)}")
    dataset_tensor_shape = tuple(ToTensor()(dataset[0][0]).shape)
    print(f"Tensor shapes are : {dataset_tensor_shape}")
    return dataset, dir_path, dataset_tensor_shape

# def get_dataset_tensor_shape(dataset : torchvision.datasets):


def get_dataloader(batch_size: int, dir_path: str):
    transform = Compose([ToTensor(), Lambda(lambda x: (x - 0.5) * 2)])  # scale to [-1, 1]
    
    # Write your dataset name here e.g MNIST, CIFAR, etc
    # dataset = torchvision.datasets.MNIST(root=f'./{dir_path}', transform=transform)
    dataset = torchvision.datasets.CIFAR10(root=f'./{dir_path}', transform=transform)


    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def get_mnist_tensor_shape():
    return (1,28,28)




if __name__ == "__main__":

    # Write your dataset name here e.g MNIST, CIFAR, etc
    dataset_name = 'mnist'
    # dataset_name = 'cifar10'

    dataset, dir_path = download_dataset(dataset_name)
    dataset_dataloader = get_dataloader(10, dir_path)

