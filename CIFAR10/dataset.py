from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch import nn
import os
import multiprocessing
import wandb
import torch
import json

def get_CIFAR10(train=True):
    data_path = os.path.join(str(Path(__file__).parent), "data")
    if not os.path.isdir(data_path):
        os.makedirs(data_path)

    num_workers = multiprocessing.cpu_count()

    if train == True:
        train = datasets.CIFAR10(data_path, train=train, download=True, transform=transforms.ToTensor())
        train, valid = random_split(train, [49_900, 100])

        li_1 = []
        li_2 = []
        li_3 = []
        for d in train:
            li_1.append(d[0][0])
            li_2.append(d[0][1])
            li_3.append(d[0][2])
        
        li_1 = torch.stack(li_1, dim=0)
        li_2 = torch.stack(li_2, dim=0)
        li_3 = torch.stack(li_3, dim=0)
        print(li_1.shape)
        mean_1 = torch.mean(li_1).item()
        mean_2 = torch.mean(li_2).item()
        mean_3 = torch.mean(li_3).item()
        std_1 = torch.std(li_1).item()
        std_2 = torch.std(li_2).item()
        std_3 = torch.std(li_3).item()

        train_data_loader = DataLoader(dataset=train, batch_size=wandb.config.batch_size, shuffle=True, num_workers=num_workers)
        valid_data_loader = DataLoader(dataset=valid, batch_size=wandb.config.batch_size, shuffle=True, num_workers=num_workers)
        CIFAR10_transforms = nn.Sequential(
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=(mean_1, mean_2, mean_3), std=(std_1, std_2, std_3))
        )

        value = {
            'mean_1': mean_1, 'mean_2': mean_2, 'mean_3': mean_3,
            'std_1': std_1, 'std_2': std_2, 'std_3': std_3
        }
        json_path = os.path.join(Path(__file__).parent, "model/lastest_value.json")

        with open(json_path, 'w') as json_file:
            json.dump(value, json_file)

        return train_data_loader, valid_data_loader, CIFAR10_transforms
    
    else:
        json_path = os.path.join(Path(__file__).parent, "model/lastest_value.json")
        with open(json_path, 'r') as json_file:
            data = json.load(json_file)
        
        mean_1 = data['mean_1']
        mean_2 = data['mean_2']
        mean_3 = data['mean_3']
        std_1 = data['std_1']
        std_2 = data['std_2']
        std_3 = data['std_3']
        test_datasets = datasets.CIFAR10(data_path, train=train, download=True)
        test_data = datasets.CIFAR10(data_path, train=train, download=True, transform=transforms.ToTensor())
        test_data_loader = DataLoader(dataset=test_data, batch_size=len(test_data), num_workers=num_workers)
        CIFAR10_transforms = nn.Sequential(
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=(mean_1, mean_2, mean_3), std=(std_1, std_2, std_3))
        )

        return test_datasets, test_data_loader, CIFAR10_transforms


def get_CIFAR100(train=True, mean=0, std=0):
    data_path = os.path.join(str(Path(__file__).parent), "data")
    if not os.path.isdir(data_path):
        os.makedirs(data_path)

    if train == True:
        train = datasets.CIFAR100(data_path, train=train, download=True, transform=transforms.ToTensor())
        train, valid = random_split(train, [49_000, 1_000])

        li_1 = []
        li_2 = []
        li_3 = []
        for d in train:
            li_1.append(d[0][0])
            li_2.append(d[0][1])
            li_3.append(d[0][2])
        
        li_1 = torch.stack(li_1, dim=1)
        li_2 = torch.stack(li_2, dim=1)
        li_3 = torch.stack(li_3, dim=1)
        mean_1 = torch.mean(li_1).item()
        mean_2 = torch.mean(li_2).item()
        mean_3 = torch.mean(li_3).item()
        std_1 = torch.std(li_1).item()
        std_2 = torch.std(li_2).item()
        std_3 = torch.std(li_3).item()

        num_workers = multiprocessing.cpu_count()

        train_data_loader = DataLoader(dataset=train, batch_size=wandb.config.batch_size, shuffle=True, num_workers=num_workers)
        valid_data_loader = DataLoader(dataset=valid, batch_size=wandb.config.batch_size, shuffle=True, num_workers=num_workers)
        CIFAR100_transforms = nn.Sequential(
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=(mean_1, mean_2, mean_3), std=(std_1, std_2, std_3))
        )

        value = {
            'mean_1': mean, 'mean_2': mean_2, 'mean_3': mean_3,
            'std_1': std_1, 'std_2': std_2, 'std_3': std_3
        }
        json_path = os.path.join(Path(__file__).parent, "model/lastest_value.json")

        with open(json_path, 'w') as json_file:
            json.dump(value, json_file)

        return train_data_loader, valid_data_loader, CIFAR100_transforms
    
    else:
        json_path = os.path.join(Path(__file__).parent, "model/lastest_value.json")
        with open(json_path, 'r') as json_file:
            data = json.load[json_file]
        
        mean_1 = data['mean_1']
        mean_2 = data['mean_2']
        mean_3 = data['mean_3']
        std_1 = data['std_1']
        std_2 = data['std_2']
        std_3 = data['std_3']

        test_datasets = datasets.CIFAR10(data_path, train=train, download=True)
        test_data_loader = datasets.CIFAR10(data_path, train=train, download=True, transform=transforms.ToTensor())
        CIFAR100_transforms = nn.Sequential(
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=(mean_1, mean_2, mean_3), std=(std_1, std_2, std_3))
        )

        return test_data_loader, test_datasets, CIFAR100_transforms
