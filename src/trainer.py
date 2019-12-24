import time
from typing import Union
import torch
import torch.backends.cudnn
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

def get_indices(l, x):
    '''Get all of the indices of l with value x'''
    return [i for i in range(len(l)) if l[i] == x]

class Trainer:
    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            criterion: nn.Module,
            optimizer: Optimizer,
            summary_writer: SummaryWriter,
            device: torch.device,
        ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.step = 0

    def train(
            self,
            epochs: int,
            val_frequency: int,
            print_frequency: int = 20,
            log_frequency: int = 5,
            start_epoch: int = 0
        ):
        self.model.train()
        for epoch in range(start_epoch, epochs):
            self.model.train()
            data_load_start_time = time.time()
            for i, (batch, labels, filename) in enumerate(self.train_loader):
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                data_load_end_time = time.time()

                logits = self.model.forward(batch)

                loss = self.criterion(logits, labels)

                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                with torch.no_grad():
                    preds = logits.argmax(-1)
                    accuracy = self.compute_accuracy(labels, preds)

                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time
                if ((self.step + 1) % log_frequency) == 0:
                    self.log_metrics(epoch, accuracy, loss, data_load_time, step_time)
                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, accuracy, loss, data_load_time, step_time)

                self.step += 1
                data_load_start_time = time.time()

            self.summary_writer.add_scalar("epoch", epoch, self.step)
            if ((epoch + 1) % val_frequency) == 0:
                self.validate()
                # self.validate() will put the model in validation mode,
                # so we have to switch back to train mode afterwards
                self.model.train()

    def compute_accuracy(self, labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]) -> float:
        """
        Args:
            labels: ``(batch_size, class_count)`` tensor or array containing example labels
            preds: ``(batch_size, class_count)`` tensor or array containing model prediction
        """
        assert len(labels) == len(preds)
        return float((labels == preds).sum()) / len(labels)

    def compute_class_accuracy(self, labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]) -> [float]:
        assert len(labels) == len(preds)

        classes = []
        for c in range(10):
            total = 0
            count = 0
            for label, pred in zip(labels, preds):
                if label == c:
                    if pred == c:
                        count += 1
                    total += 1
            classes.append(float(count) / float(total))
        return classes 

    def print_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        epoch_step = self.step % len(self.train_loader)
        print(
                f"epoch: [{epoch}], "
                f"step: [{epoch_step}/{len(self.train_loader)}], "
                f"batch loss: {loss:.5f}, "
                f"batch accuracy: {accuracy * 100:2.2f}, "
                f"data load time: "
                f"{data_load_time:.5f}, "
                f"step time: {step_time:.5f}"
        )

    def log_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalars(
                "accuracy",
                {"train": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"train": float(loss.item())},
                self.step
        )
        self.summary_writer.add_scalar(
                "time/data", data_load_time, self.step
        )
        self.summary_writer.add_scalar(
                "time/data", step_time, self.step
        )

    def compute_average_prediction(self, logits, mode='mode'):
        '''
        logits: batch_size x 10 tensor 
        returns 1 x 10 tensor
        '''
        if mode == 'mode':
            argmaxs = logits.argmax(dim=-1)
            return argmaxs.mode().values.item()
        elif mode == 'mean':
            means = logits.mean(dim=0)
            return means.argmax(dim=0).item()
        else:
            raise NotImplementedError
        print("Error, returning 0")
        return logits[0]

    def validate(self):
        results = {"preds": [], "labels": []}
        segment_results = {'logits': [], "labels": [], "fname": []}
        total_loss = 0
        self.model.eval()

        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for batch, labels, fnames in self.val_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(batch)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                # Collect all results to merge
                segment_results['logits'].extend(list(logits))
                segment_results['labels'].extend(list(labels))
                segment_results['fname'].extend(fnames)

            # For each unique file
            for fname in set(segment_results['fname']):
                # Get logits and labels from this file
                indices = get_indices(segment_results['fname'], fname)
                file_logits = torch.Tensor([list(segment_results['logits'][i]) for i in indices])
                file_labels = [segment_results['labels'][i] for i in indices]
                # All labels should be the same in a file
                assert(all(file_labels[0] == label for label in file_labels))
                label = file_labels[0]
                # Average the logits from this file
                prediction = self.compute_average_prediction(file_logits)
                results['preds'].append(prediction)
                results['labels'].append(label)

        accuracy = self.compute_accuracy(
            np.array(results["labels"]), np.array(results["preds"])
        )
        class_accuracy = self.compute_class_accuracy(
            np.array(results["labels"]), np.array(results["preds"])
        )
        average_loss = total_loss / len(self.val_loader)

        self.summary_writer.add_scalars(
                "accuracy",
                {"test": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"test": average_loss},
                self.step
        )
        print(f"validation loss: {average_loss:.5f}, accuracy: {accuracy * 100:2.2f}")
        for i, acc in enumerate(class_accuracy):
            print(f"class {i} accuracy: {acc * 100:22.3f}")

