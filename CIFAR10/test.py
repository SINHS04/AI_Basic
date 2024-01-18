from dataset import get_CIFAR10
from pathlib import Path
import os
import json
from torch import nn
from tester import Tester
from model import GoogleNet
import torch

def test(args):
    project = "GoogleNet"
    model = GoogleNet(n_outputs=10)
    checkpoint_file_path = os.path.join(Path(__file__).parent, "model")
    device = "cpu"

    test_data_loader, test_datasets, CIFAR10_transforms = get_CIFAR10(train=False)
    tester = Tester(project, model, device, test_data_loader, test_datasets, CIFAR10_transforms, checkpoint_file_path)

    tester.do_test()

    tester.test_random_input()