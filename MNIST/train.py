import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from datetime import datetime
import os
import wandb
from pathlib import Path
import sys
from torch import optim

BASE_PATH = str(Path(__file__).resolve().parent)

CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_FILE_PATH = os.path.dirname(os.path.join(CURRENT_FILE_PATH, "checkpoints"))
CHECKPOINT_FILE_PATH = os.path.join(CURRENT_FILE_PATH, "checkpoints")

from arg_parser import get_parser
from mnistdata import get_mnist_data
from mymodel import get_model
from classification import ClassificationTraniner


def main(args):
    runtime = datetime.now().astimezone().strftime('%Y-%m-%d_%H-%M-%S')

    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'validation_intervals': args.validation_intervals,
        'early_stop_patience': args.early_stop_patience,
        'early_stop_delta': args.early_stop_delta,
    }

    project = "MNIST train"
    wandb.init(
        mode='online' if args.wandb else 'disabled',
        project=project,
        notes="mnist experiment with fcn",
        tags=['fcn', 'mnist'],
        name=runtime,
        config=config
    )

    print(wandb.config)

    print(torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on device {device}")

    train_data_loader, validation_data_loader, mnist_transforms, mean, std = get_mnist_data(flatten=True)
    model = get_model()
    model.to(device)
    wandb.watch(model)

    optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)

    classification_trainer = ClassificationTraniner(
        project, model, optimizer, train_data_loader, validation_data_loader, mnist_transforms,
        runtime, wandb, device, CHECKPOINT_FILE_PATH
    )

    classification_trainer.train_loop()

    wandb.finish()


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)