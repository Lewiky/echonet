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
from fusion_validator import FusionValidator
from trainer import Trainer
from dataset import UrbanSound8KDataset

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(
    description="Train a network for performing intelligent sound recognition using the UrbanSound8K dataset",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--dataset-root", default=Path("data"))
parser.add_argument("--log-dir", default=Path("logs"), type=Path)
parser.add_argument("--qual-results", help="File path to store qualitative results output")
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
parser.add_argument("--mode", default='LMC', const='LMC', nargs='?', choices=["LMC", "MC", "MLMC", "TSCNN"], type=str)
parser.add_argument('--dropout', default=0.5, const=0.5, nargs='?', type=float, help="Dropout probability propagated to all networks")
parser.add_argument('--weight-decay', default=1e-4, const=1e-4, nargs='?', type=float, help="Weight decay to apply to Adam optimiser")
parser.add_argument('--augmentation-length', default=1, const=1, nargs='?', type=int, help="Total iterations through dataset - >1 will include augmentation")

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
    tb_log_dir_prefix = f'CNN_bn_lr={args.learning_rate}_dropout={args.dropout}_mode={args.mode}_run_'
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)

def calculate_weights(dataset):
    """Compute inverted weighting of each class in the dataset.
    This ensures classes with more occurences have less weighting

    Args:
        dataset: the dataset the compute the weight from

    Returns:
        A tuple of inverted class weightings for each class in the dataset, as well
        as a list of the weights for each sample in the dataset, according to the class
        label associated with it.
    """
    classes = 10
    class_counts = [0] * classes
    # compute occurrences of each class
    for (_, label, _, _) in dataset:
        class_counts[label] += 1

    # work out weight per class, favouring those with less occurrences
    per_class_weights = [1 / float(class_counts[i]) for i in range(classes)]
    
    # attach weight to each sample
    return per_class_weights, [per_class_weights[label] for (_, label, _, _) in dataset]


def main(args):
    print(f"Running in {args.mode} mode")
    print(f"Running with augmentation length {args.augmentation_length}")

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

    if args.mode == "TSCNN":
        # Train LMC model independently
        args.mode = "LMC"
        lmc_train_loader, lmc_test_loader, class_weights = build_dataloader(args)
        criterion = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights).to(DEVICE))
        lmc_model = CNN(height=85, width=41, channels=1, class_count=10, dropout=args.dropout)
        lmc_optimizer = optim.Adam(lmc_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        lmc_trainer = Trainer(
            lmc_model, lmc_train_loader, lmc_test_loader, criterion, lmc_optimizer, summary_writer, DEVICE, args.qual_results
        )
        lmc_trainer.train(
            args.epochs,
            args.val_frequency,
            print_frequency=args.print_frequency,
            log_frequency=args.log_frequency,
        )

        # Train MC model independently
        args.mode = "MC"
        mc_train_loader, mc_test_loader, _ = build_dataloader(args)
        mc_model = CNN(height=85, width=41, channels=1, class_count=10, dropout=args.dropout)
        mc_optimizer = optim.Adam(mc_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        mc_trainer = Trainer(
            mc_model, mc_train_loader, mc_test_loader, criterion, mc_optimizer, summary_writer, DEVICE, args.qual_results
        )
        mc_trainer.train(
            args.epochs,
            args.val_frequency,
            print_frequency=args.print_frequency,
            log_frequency=args.log_frequency,
        )

        # Perform fusion and validate both networks using "late fusion"
        print("Validating")
        args.mode = "TSCNN"
        _, fusion_test_loader, _ = build_dataloader(args)
        f_validator = FusionValidator(lmc_model, 
            mc_model, 
            fusion_test_loader,
            criterion,
            summary_writer,
            DEVICE,
            args.qual_results)
        f_validator.validate()

    else:
        # Train the network according to the CLI Mode arg
        train_loader, test_loader, class_weights = build_dataloader(args)

        # Define the loss function
        criterion = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights).to(DEVICE))

        if args.mode == "MLMC":
            # Define the MLMC model and trainer
            model = MLMC_CNN(height=85, width=41, channels=1, class_count=10, dropout=args.dropout)
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            trainer = Trainer(
                model, train_loader, test_loader, criterion, optimizer, summary_writer, DEVICE, args.qual_results
            )
        elif args.mode == "MC" or args.mode == "LMC":
            # Define the model and trainer for MC or LMC
            model = CNN(height=85, width=41, channels=1, class_count=10, dropout=args.dropout)
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            trainer = Trainer(
                model, train_loader, test_loader, criterion, optimizer, summary_writer, DEVICE, args.qual_results
            )

        # Train the MLMC, MC or LMC model
        trainer.train(
            args.epochs,
            args.val_frequency,
            print_frequency=args.print_frequency,
            log_frequency=args.log_frequency,
        )

    summary_writer.close()

def build_dataloader(args):
    """ Build a training and test dataloader for the UrbanSound8K dataset

    Args:
        args: the CLI args

    Returns:
        A tuple of training loader, testing loader and individual inverted class weights
    """

    # Load data set and compute class weightings
    train_dataset = UrbanSound8KDataset(args.dataset_root / 'UrbanSound8K_train.pkl', args.mode, augmentation_length = args.augmentation_length)
    class_weights, sample_weights = calculate_weights(train_dataset)

    # Define a weighted sampler for the dataset
    weighted_sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(train_dataset))

    # Configure data loaders for testing and training
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.worker_count,
        sampler=weighted_sampler,
    )

    test_loader = torch.utils.data.DataLoader(
        UrbanSound8KDataset(args.dataset_root / 'UrbanSound8K_test.pkl', args.mode),
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.worker_count,
        pin_memory=True,
    )
    
    return train_loader, test_loader, class_weights

if __name__ == "__main__":
    main(parser.parse_args())
