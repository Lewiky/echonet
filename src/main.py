#!/usr/bin/env python3
import time
from multiprocessing import cpu_count
from typing import Union, NamedTuple

import torch
import torch.backends.cudnn
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import argparse
from pathlib import Path
from cnn import CNN, MLMC_CNN
from fusion_trainer import FusionTrainer
from trainer import Trainer
from dataset import UrbanSound8KDataset

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(
    description="Train a simple CNN on CIFAR-10",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
default_dataset_dir = Path.home() / ".cache" / "torch" / "datasets"
parser.add_argument("--dataset-root", default=default_dataset_dir)
parser.add_argument("--log-dir", default=Path("logs"), type=Path)
parser.add_argument("--learning-rate", default=0.001, type=float, help="Learning rate")
parser.add_argument(
    "--batch-size",
    default=32,
    type=int,
    help="Number of images within each mini-batch",
)
parser.add_argument(
    "--epochs",
    default=50,
    type=int,
    help="Number of epochs (passes through the entire dataset) to train for",
)
parser.add_argument(
    "--val-frequency",
    default=2,
    type=int,
    help="How frequently to test the model on the validation set in number of epochs",
)
parser.add_argument(
    "--log-frequency",
    default=10,
    type=int,
    help="How frequently to save logs to tensorboard in number of steps",
)
parser.add_argument(
    "--print-frequency",
    default=10,
    type=int,
    help="How frequently to print progress to the command line in number of steps",
)
parser.add_argument(
    "-j",
    "--worker-count",
    default=cpu_count(),
    type=int,
    help="Number of worker processes used to load data.",
)
parser.add_argument("--sgd-momentum", default=0.9, type=float)
parser.add_argument("--mode", default='LMC', const='LMC', nargs='?', choices=["LMC", "MC", "MLMC", "TSCNN"], type=str)

def get_summary_writer_log_dir(args: argparse.Namespace) -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    tb_log_dir_prefix = f'CNN_bn_bs={args.batch_size}_lr={args.learning_rate}_momentum={args.sgd_momentum}_run_'
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)

def calculate_weights(dataset):
    classes = 10
    class_counts = [0] * classes
    # compute occurrences of each class
    for (feature, label, filename) in dataset:
        class_counts[label] += 1

    # work out weight per class, favouring those with less occurrences
    per_class_weights = [1 / float(class_counts[i]) for i in range(classes)]
    
    # attach weight to each sample
    return per_class_weights, [per_class_weights[label] for (feature, label, filename) in dataset]


def main(args):
    print(f"Running in {args.mode} mode")

    # Device and log dir config
    DEVICE = torch.device("cpu")
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")

    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(
            str(log_dir),
            flush_secs=5
    )

    train_dataset = UrbanSound8KDataset('data/UrbanSound8K_train.pkl', args.mode)
    # class_weights, sample_weights = calculate_weights(train_dataset)
    # weighted_sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(train_dataset))

    # Configure data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.worker_count,
        # sampler=weighted_sampler,
    )

    test_loader = torch.utils.data.DataLoader(
        UrbanSound8KDataset('data/UrbanSound8K_test.pkl', args.mode),
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.worker_count,
        pin_memory=True,
    )

    criterion = nn.CrossEntropyLoss() # weight=torch.Tensor(class_weights).to(DEVICE))
    if args.mode == "TSCNN":
        # Run LMC and MC in parallel
        lmc_model = CNN(height=85, width=41, channels=1, class_count=10)
        mc_model = CNN(height=85, width=41, channels=1, class_count=10)
        lmc_optimizer = optim.Adam(lmc_model.parameters(), lr=args.learning_rate, weight_decay=0.0001)
        mc_optimizer = optim.Adam(mc_model.parameters(), lr=args.learning_rate, weight_decay=0.0001)
        trainer = FusionTrainer(
            lmc_model, mc_model, train_loader, test_loader, criterion, lmc_optimizer, mc_optimizer, summary_writer, DEVICE
        )
    elif args.mode == "MLMC":
        model = MLMC_CNN(height=85, width=41, channels=1, class_count=10)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.0001)
        trainer = Trainer(
            model, train_loader, test_loader, criterion, optimizer, summary_writer, DEVICE
        )
    elif args.mode == "MC" or args.mode == "LMC":
        model = CNN(height=85, width=41, channels=1, class_count=10)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.0001)
        trainer = Trainer(
            model, train_loader, test_loader, criterion, optimizer, summary_writer, DEVICE
        )

    trainer.train(
        args.epochs,
        args.val_frequency,
        print_frequency=args.print_frequency,
        log_frequency=args.log_frequency,
    )

    summary_writer.close()

if __name__ == "__main__":
    main(parser.parse_args())
