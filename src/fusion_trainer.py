import time
import torch
import torch.backends.cudnn
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from base_trainer import BaseTrainer

class FusionTrainer(BaseTrainer):
    def __init__(
            self,
            lmc_model: nn.Module,
            mc_model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            criterion: nn.Module,
            lmc_optimizer: Optimizer,
            mc_optimizer: Optimizer,
            summary_writer: SummaryWriter,
            device: torch.device,
        ):
        BaseTrainer.__init__(self, train_loader, summary_writer)
        self.lmc_model = lmc_model.to(device)
        self.mc_model = mc_model.to(device)
        self.device = device
        self.val_loader = val_loader
        self.criterion = criterion
        self.lmc_optimizer = lmc_optimizer
        self.mc_optimizer = mc_optimizer

    def train(
            self,
            epochs: int,
            val_frequency: int,
            print_frequency: int = 20,
            log_frequency: int = 5,
            start_epoch: int = 0
        ):
        # TODO
        print("training")

    def validate(self):
        print("validating")
