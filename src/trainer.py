import time
import torch
import torch.backends.cudnn
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from base_trainer import BaseTrainer

class Trainer(BaseTrainer):
    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            criterion: nn.Module,
            optimizer: Optimizer,
            summary_writer: SummaryWriter,
            device: torch.device,
            patience: int = 10
        ):
        BaseTrainer.__init__(self, train_loader, summary_writer)
        self.model = model.to(device)
        self.device = device
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.counter = 0
        self.best_loss = None
        self.patience = patience

    def train(
            self,
            epochs: int,
            val_frequency: int,
            print_frequency: int = 20,
            log_frequency: int = 5,
            start_epoch: int = 0
        ):
        self.model.train()
        best_loss = None
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
                current_loss = self.validate()
                if self.early_stop(current_loss):
                    break
                # self.validate() will put the model in validation mode,
                # so we have to switch back to train mode afterwards
                self.model.train()
        self.model.load_state_dict(torch.load('best_model.pt'))


    def validate(self) -> float:
        results = {"preds": [], "labels": []}
        segment_results = {'logits': [], "labels": [], "fname": []}
        total_loss = 0
        self.model.eval()

        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for batch, labels, fnames in self.val_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                #compute the models predictions for each segment
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
        accuracy = np.sum(class_accuracy) / len(class_accuracy)

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
        return average_loss

    def early_stop(self, loss: float) -> bool:
        '''
        Keep track of the loss and if it starts increasing, stop training the model
        https://www.researchgate.net/profile/Lutz_Prechelt/publication/2874749_Early_Stopping_-_But_When/links/551bc1650cf2fe6cbf75e533.pdf
        '''
        if self.best_loss is None:
            self.best_loss = loss
            # Save the best seen model to use again later
            torch.save(self.model.state_dict(), 'best_model.pt')
        elif loss > self.best_loss:
            self.counter += 1
            if self.counter > self.patience:
                print("Early stopping - loss increase detected")
                return True
        else:
            self.best_loss = loss
            self.counter -= 1
            torch.save(self.model.state_dict(), 'best_model.pt')
        return False
