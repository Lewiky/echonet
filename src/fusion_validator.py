import time
import torch
import torch.backends.cudnn
import numpy as np
from typing import Union
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from base_trainer import BaseTrainer

class FusionValidator():
    def __init__(
            self,
            lmc_model: nn.Module,
            mc_model: nn.Module,
            val_loader: DataLoader,
            criterion: nn.Module,
            summary_writer: SummaryWriter,
            device: torch.device,
            qualitative_results_file: str = None
        ):
        self.lmc_model = lmc_model.to(device)
        self.mc_model = mc_model.to(device)
        self.val_loader = val_loader
        self.criterion = criterion
        self.summary_writer = summary_writer
        self.device = device
        self.qual_results_file = qualitative_results_file

    def get_indices(self, l, x):
        '''Get all of the indices of l with value x'''
        return [i for i in range(len(l)) if l[i] == x]

    def gather_qualitative_results(self):
        # Once we've finished, work out what is predicted correctly
        print("Collecting results...")
        with torch.no_grad():
            for (lmc_batch, mc_batch), labels, fnames, indices in self.val_loader:
                lmc_batch = lmc_batch.to(self.device)
                mc_batch = mc_batch.to(self.device)
                labels = labels.to(self.device)

                #compute the models predictions for each segment
                lmc_logits = self.lmc_model(lmc_batch)
                mc_logits = self.mc_model(mc_batch)

                # Take mean of lmc and mc prediction of segment
                logits = torch.mean(torch.stack((lmc_logits, mc_logits), dim=2), dim=2)

                is_correct = (labels == logits.argmax(-1))
                with open(self.qual_results_file,'a+') as f:
                    f.write("".join([f"{x},{y}\n" for (x,y) in zip(indices, is_correct)]))

    def validate(self):
        results = {"preds": [], "labels": []}
        segment_results = {'logits': [], "labels": [], "fname": []}
        total_loss = 0
        self.lmc_model.eval()
        self.mc_model.eval()

        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for ((lmc_batch, mc_batch), labels, fnames, indicies) in self.val_loader:
                lmc_batch = lmc_batch.to(self.device)
                mc_batch = mc_batch.to(self.device)
                labels = labels.to(self.device)

                #compute the models predictions for each segment
                lmc_logits = self.lmc_model(lmc_batch)
                mc_logits = self.mc_model(mc_batch)

                # Take mean of lmc and mc prediction of segment
                logits = torch.mean(torch.stack((lmc_logits, mc_logits), dim=2), dim=2)

                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                # Collect all results to merge
                segment_results['logits'].extend(list(logits))
                segment_results['labels'].extend(list(labels))
                segment_results['fname'].extend(fnames)

            # For each unique file
            for fname in set(segment_results['fname']):
                # Get logits and labels from this file
                indices = self.get_indices(segment_results['fname'], fname)
                file_logits = torch.Tensor([list(segment_results['logits'][i]) for i in indices])
                file_labels = [segment_results['labels'][i] for i in indices]
                # All labels should be the same in a file
                assert(all(file_labels[0] == label for label in file_labels))
                label = file_labels[0]
                # Average the logits from this file
                prediction = self.compute_average_prediction(file_logits)
                results['preds'].append(prediction)
                results['labels'].append(label)

        class_accuracy = self.compute_class_accuracy(
            np.array(results["labels"]), np.array(results["preds"])
        )

        # accuracy = self.compute_accuracy(
        #     np.array(results["labels"]), np.array(results["preds"])
        # )
        accuracy = np.sum(class_accuracy) / len(class_accuracy)

        average_loss = total_loss / len(self.val_loader)

        print(f"validation loss: {average_loss:.5f}, accuracy: {accuracy * 100:2.2f}")
        for i, acc in enumerate(class_accuracy):
            print(f"class {i} accuracy: {acc * 100:22.3f}")

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
