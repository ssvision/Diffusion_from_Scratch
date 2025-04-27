import os
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from alternate_diffusion_model import DeepDenoisingProbModel



def get_cifar10_dataloader(root: str,
                           batch_size: int = 64,
                           shuffle: bool = True,
                           num_workers: int = 2,
                           download: bool = False):
    """
    Returns a DataLoader for the CIFAR-10 dataset without normalization.

    Args:
        root (str): Path to dataset directory.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data at every epoch.
        num_workers (int): Number of subprocesses for data loading.
        download (bool): If True, downloads the dataset if not present.

    Returns:
        DataLoader: An iterable over the CIFAR-10 dataset.
    """
    # Define transformation: convert to tensor only
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load the CIFAR-10 dataset
    cifar10 = datasets.CIFAR10(root=root,
                               train=True,
                               transform=transform,
                               download=download)

    # Create DataLoader
    loader = DataLoader(cifar10,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=num_workers)
    return loader


def imshow_batch(images, labels=None, classes=None, save=False, filename=None, dir_name = None):
    """
    Display a batch of images (no normalization applied).

    Args:
        images (Tensor): Batch of images [B, C, H, W].
        labels (Tensor): Corresponding labels.
        classes (list): List of class names.
    """
    # Create a grid of images
    grid = torchvision.utils.make_grid(images, nrow=8)
    if save is True:
        os.makedirs(dir_name, exist_ok=True)
        torchvision.utils.save_image(grid, os.path.join(dir_name, filename))
    # print(f"Shape of grid  is : {grid.shape}")
    ''' The grid is a tensor of size [C, H1, W1]
         H1 = H + 2 * 2  (a padding of 2 pixels added up & down)
         W1 = (W * n_rows) + (n_rows+1) * 2 (a padding of 2 pixels on left edge, 2 pixels between adjacent pair of images, 2 on right edge)  '''
    npimg = grid.numpy()
    # print(f"Shape of npimg is : {npimg.shape}")
    plt.figure(figsize=(8, 4))
    ''' Transpose the image to (H,W,C) format to make it compatible with matplotlib format'''
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    # Add labels as title
    if classes is not None and labels is not None:
        title = ' '.join('%5s' % classes[labels[j]] for j in range(len(labels)))
        plt.title(title)
    plt.show()


def create_gif_from_images(
    image_folder: str,
    output_path: str,
    duration: int = 100,
    loop: int = 0,
    file_extensions: tuple = ('.png', '.jpg', '.jpeg')
):
    """
    Create a GIF from a set of images in a folder.

    Parameters:
        image_folder (str): Path to the folder containing image files.
        output_path (str): File path to save the output GIF.
        duration (int): Duration per frame in milliseconds. Default is 100ms.
        loop (int): Number of times the GIF should loop (0 = infinite). Default is 0.
        file_extensions (tuple): Allowed image file extensions. Default is ('.png', '.jpg', '.jpeg').
    """
    # Get sorted list of image file paths
    image_files = sorted(
        [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(file_extensions)]
    )

    if not image_files:
        raise ValueError("No valid image files found in the specified folder.")

    # Load images
    frames = [Image.open(f) for f in image_files]

    # Save as GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop
    )

    print(f"GIF successfully saved to {output_path}")



if __name__ == '__main__':

    torch.manual_seed(123)
    # Path to dataset
    data_root = os.path.expanduser('./data/cifar10')

    batch_size = 8
    # Get loader
    loader = get_cifar10_dataloader(root=data_root,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=4,
                                    download=False)

    # CIFAR-10 classes
    classes = ['plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Iterate and visualize a batch
    data_iter = iter(loader)
    images, labels = next(data_iter)
    imshow_batch(images, save=True, filename=f'out_0.jpg', dir_name='temp')
    

    num_steps = 10
    device = 'cpu'
    ddpm = DeepDenoisingProbModel(num_steps=10, beta_min=0.001, beta_max=0.1, device='cpu')


    # timesteps = torch.full((batch_size,), 1)
    # noised_img = ddpm.fwd_process(images, timesteps)
    # imshow_batch(noised_img, save=True, filename='random.jpg', dir_name='temp')

    for idx in range(0,num_steps):

        timesteps = torch.full((batch_size,), idx)
        noised_img = ddpm.fwd_process(images, timesteps)
        imshow_batch(noised_img, save=True, filename=f'out_{idx+1}.jpg', dir_name='temp')


    create_gif_from_images(image_folder='./temp/', output_path='./out/animated.gif')


