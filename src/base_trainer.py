from typing import Union
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Some common functionality shared between trainers
class BaseTrainer:
    def __init__(
            self, 
            train_loader: DataLoader, 
            summary_writer: SummaryWriter,
        ):
        self.train_loader = train_loader
        self.summary_writer = summary_writer
        self.step = 0

    def get_indices(self, l, x):
        '''Get all of the indices of l with value x'''
        return [i for i in range(len(l)) if l[i] == x]

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

    def print_metrics(self, epoch, accuracy, loss, data_load_time, step_time, model=None):
        epoch_step = self.step % len(self.train_loader)
        if model != None:
            print(f"model: {model}, ", end="")

        print(
                f"epoch: [{epoch}], "
                f"step: [{epoch_step}/{len(self.train_loader)}], "
                f"batch loss: {loss:.5f}, "
                f"batch accuracy: {accuracy * 100:2.2f}, "
                f"data load time: "
                f"{data_load_time:.5f}, "
                f"step time: {step_time:.5f}"
        )

    def log_metrics(self, epoch, accuracy, loss, data_load_time, step_time, model=None):
        self.summary_writer.add_scalar("epoch", epoch, self.step)

        train_key = "train" if model == None else "train_" + model
        self.summary_writer.add_scalars(
                "accuracy",
                {train_key: accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {train_key: float(loss.item())},
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
