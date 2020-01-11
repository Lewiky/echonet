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
            qualitative_results_file: str = None
        ):
        BaseTrainer.__init__(self, train_loader, summary_writer)
        self.lmc_model = lmc_model.to(device)
        self.mc_model = mc_model.to(device)
        self.device = device
        self.val_loader = val_loader
        self.criterion = criterion
        self.lmc_optimizer = lmc_optimizer
        self.mc_optimizer = mc_optimizer
        self.qual_results_file = qualitative_results_file

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
                    
    def train(
            self,
            epochs: int,
            val_frequency: int,
            print_frequency: int = 20,
            log_frequency: int = 5,
            start_epoch: int = 0
        ):
        self.lmc_model.train()
        self.mc_model.train()

        for epoch in range(start_epoch, epochs):
            self.lmc_model.train()
            self.mc_model.train()
            data_load_start_time = time.time()
            for i, ((lmc_batch, mc_batch), labels, filenames, indices) in enumerate(self.train_loader):
                lmc_batch = lmc_batch.to(self.device)
                mc_batch = mc_batch.to(self.device)
                labels = labels.to(self.device)
                data_load_end_time = time.time()

                lmc_logits = self.lmc_model.forward(lmc_batch)
                mc_logits = self.mc_model.forward(mc_batch)

                lmc_loss = self.criterion(lmc_logits, labels)
                lmc_loss.backward()

                mc_loss = self.criterion(mc_logits, labels)
                mc_loss.backward()

                self.lmc_optimizer.step()
                self.lmc_optimizer.zero_grad()

                self.mc_optimizer.step()
                self.mc_optimizer.zero_grad()

                with torch.no_grad():
                    lmc_preds = lmc_logits.argmax(-1)
                    mc_preds = mc_logits.argmax(-1)
                    
                    lmc_accuracy = self.compute_accuracy(labels, lmc_preds)
                    mc_accuracy = self.compute_accuracy(labels, mc_preds)

                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time
                if ((self.step + 1) % log_frequency) == 0:
                    self.log_metrics(epoch, lmc_accuracy, lmc_loss, data_load_time, step_time, model="lmc")
                    self.log_metrics(epoch, mc_accuracy, mc_loss, data_load_time, step_time, model="mc")
                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, lmc_accuracy, lmc_loss, data_load_time, step_time, model="lmc")
                    self.print_metrics(epoch, mc_accuracy, mc_loss, data_load_time, step_time, model="mc")

                self.step += 1
                data_load_start_time = time.time()

            self.summary_writer.add_scalar("epoch", epoch, self.step)
            if ((epoch + 1) % val_frequency) == 0:
                self.validate()
                # self.validate() will put the model in validation mode,
                # so we have to switch back to train mode afterwards
                self.lmc_model.train()
                self.mc_model.train()

        # Once the model has been trained, gather qualitative results 
        if self.qual_results_file is not None:
            self.gather_qualitative_results()



    def validate(self):
        results = {"preds": [], "labels": []}
        segment_results = {'logits': [], "labels": [], "fname": []}
        total_loss = 0
        self.lmc_model.eval()
        self.mc_model.eval()

        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for ((lmc_batch, mc_batch), labels, fnames) in self.val_loader:
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
