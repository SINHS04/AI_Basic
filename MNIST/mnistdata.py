from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import os
import multiprocessing
import wandb
from torch import nn
import torch


def get_mnist_data(flatten=False):
    data_path = os.path.join(str(Path(__file__).parent), "mnist")
    if not os.path.isdir(data_path):
        os.makedirs(data_path)

    mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transforms.ToTensor())
    mnist_train, mnist_validation = random_split(mnist_train, [55_000, 5_000])

    li = []
    for d in mnist_train:
        li.append(d[0])
    
    print(li[0].shape)
    li = torch.stack(li, dim=0)
    mean = torch.mean(li)
    std = torch.std(li)

    print("# train size :", len(mnist_train))
    print("# validation size :", len(mnist_validation))

    num_workers = multiprocessing.cpu_count()
    
    print("# of data load workers :", num_workers)

    train_data_loader = DataLoader(
        dataset=mnist_train, batch_size=wandb.config.batch_size, shuffle=True, num_workers=num_workers
    )

    validation_data_loader = DataLoader(
        dataset=mnist_validation, batch_size=wandb.config.batch_size, shuffle=True, num_workers=num_workers
    )
    
    mnist_transforms = nn.Sequential(
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=mean, std=std),
    )

    if flatten:
        mnist_transforms.append(nn.Flatten())

    return train_data_loader, validation_data_loader, mnist_transforms, mean, std
