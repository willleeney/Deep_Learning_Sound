## Libraries
import time
from multiprocessing import cpu_count
from typing import Union, NamedTuple

import torch
import torch.backends.cudnn
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
import torchvision.datasets
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import argparse
import os
from pathlib import Path

from torch.utils import data
from dataset import UrbanSound8KDataset

torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                        )
## Arguments 
default_dataset_dir = Path.home() / ".cache" / "torch" / "datasets"
parser.add_argument("--log-dir", default=Path("logs"), type=Path)
parser.add_argument("--dataset-root", default=default_dataset_dir)
parser.add_argument("--learning_rate",default = 1e-3, type=float, help="Learning rate")
parser.add_argument("--sgd_momentum",default =  0.9, type=float)
parser.add_argument("--dropout", default = 0.5, type = float)
parser.add_argument("--data_aug_hflip", action='store_true')
parser.add_argument("--data_aug_vflip", action='store_true')
parser.add_argument("--data_aug_brightness", default = 0, type = float)
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

class ImageShape(NamedTuple):
    height: int
    width: int
    channels: int

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

def main(args):
    torchvision.transforms.ColorJitter(brightness=args.data_aug_brightness)
    transform = transforms.ToTensor()

    args.dataset_root.mkdir(parents=True, exist_ok=True)
    ## Choose which features to learn ( this can be an argument/ choose which net) 
    mode = 'LMC'
    ## load data
    train_loader = torch.utils.data.DataLoader(
        UrbanSound8KDataset('UrbanSound8K_train.pkl', mode),
          batch_size=args.batch_size, shuffle=True,
          num_workers=args.worker_count, pin_memory=True
          )

    val_loader = torch.utils.data.DataLoader(
        UrbanSound8KDataset('UrbanSound8K_test.pkl', mode),
          batch_size=1, shuffle=False,
          num_workers=args.worker_count, pin_memory=True)
    
    model = CNN(height=85, width=41, channels=1, class_count=10, dropout = args.dropout)
    criterion = lambda logits, labels : nn.CrossEntropyLoss()(logits, labels)
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, args.sgd_momentum)

    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(
            str(log_dir),
            flush_secs=5
    )
    trainer = Trainer(
        model, train_loader, val_loader, criterion, optimizer, summary_writer, DEVICE
    )

    trainer.train(
        args.epochs,
        args.val_frequency,
        print_frequency=args.print_frequency,
        log_frequency=args.log_frequency,
    )

    summary_writer.close()

class CNN(nn.Module):
    def __init__(self, height: int, width: int, channels: int, class_count: int, dropout: float):
        super().__init__()
        self.input_shape = ImageShape(height=height, width=width, channels=channels)
        self.class_count = class_count
        ## Convolution layer 1 
        self.conv1 = nn.Conv2d(
            in_channels=self.input_shape.channels,
            out_channels=32, 
            kernel_size=(3,3),
            padding=(2,2),
            stride=(1,1),
        )
        # batch normalisation
        self.conv1_BN = nn.BatchNorm2d(32)
        self.initialise_layer(self.conv1)

        ## Convolution layer 2
        self.conv2 = nn.Conv2d(
            in_channels=self.conv1.out_channels, 
            out_channels=64,
            kernel_size=(3,3),
            padding=(2,2),
            stride=(1,1), 
            )
        # dropout 
        self.dropout = nn.Dropout(dropout)
        self.conv2_BN = nn.BatchNorm2d(64)
        self.initialise_layer(self.conv2)
        # pooling layer
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(1, 1))

        ## Convolution layer 3
        self.conv3 = nn.Conv2d(
            in_channels=self.conv2.out_channels,
            out_channels=64,
            kernel_size=(3,3),
            padding=(2,2),
            stride=(1,1),
        )
        self.conv3_BN = nn.BatchNorm2d(64)
        self.initialise_layer(self.conv3)

        ## Convolution layer 4
        self.conv4 = nn.Conv2d(
            in_channels=self.conv3.out_channels,
            out_channels=64,
            kernel_size=(3,3),
            padding=(2,2),
            stride=(1,1),
            )
        self.conv4_BN = nn.BatchNorm2d(64)
        self.initialise_layer(self.conv4)
        
        ## Fully-Connected layer 
        self.fc1 = nn.Linear(22400, 1024)
        self.fc1_BN = nn.BatchNorm1d(1024)
        self.initialise_layer(self.fc1)

        ## Output layer
        self.fc2 = nn.Linear(1024, 10)
        self.initialise_layer(self.fc2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Conv 1 -- relu-- batch norm
        x = F.relu(self.conv1_BN(self.conv1(images)))
        # Conv 2 -- relu -- batch norm -- dropout -- pooling
        x = F.relu(self.conv2_BN(self.conv2(self.dropout(x))))
        x = self.pool(x)
        # Conv 3 -- relu -- batch norm -- dropout 
        x = F.relu(self.conv3_BN(self.conv3(self.dropout(x))))
        # Conv 4 -- relu -- batch norm -- dropout -- pooling
        x = F.relu(self.conv4_BN(self.conv4(self.dropout(x))))
        x = self.pool(x)
        # Flatten output of pooling layer
        x = torch.flatten(x, 1)
        # FC layer 1 -- sigmoid
        x = torch.sigmoid(self.fc1_BN(self.fc1(x)))
        # Output layer
        x = torch.nn.functional.softmax(self.fc2(x))
        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)


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
            for batch, labels, filename in self.train_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                data_load_end_time = time.time()
                # Forward pass
                logits = self.model.forward(batch)
                # Compute loss
                loss = self.criterion(logits, labels)
                # Compute the backward pass
                loss.backward()
                # Step optimizer 
                self.optimizer.step()
                # Zero gradient buffers
                self.optimizer.zero_grad()
                # Compute accurracy 
                with torch.no_grad():
                    preds = logits.argmax(-1)
                    accuracy, per_class_accuracy = compute_accuracy(labels, preds)

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

         # end validation code can go here
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

    def validate(self):
        results = {"preds": [], "labels": []}
        total_loss = 0
        self.model.eval()
        current_filename = ''
        file_scores = torch.zeros([1,10])
        file_scores = file_scores.to(self.device)

        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for batch, labels, filename in self.val_loader:
                if current_filename != filename and current_filename != '': 
                    file_scores = torch.sum(file_scores, 0)/(file_scores.size()[0]-1)
                    preds = file_scores.argmax(dim=-1).cpu().numpy()
                    results["preds"].extend([preds])
                    results["labels"].extend(list(current_label.cpu().numpy()))
                    file_scores = torch.zeros([1,10])
                    file_scores = file_scores.to(self.device)
                
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(batch)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                file_scores = torch.cat((file_scores, logits), 0)
                current_filename = filename
                current_label = labels
                    
        accuracy, per_class_accuracy = compute_accuracy(
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
        class_labels = ["ac", "ch", "cp", "db", "dr", "ei", "gs", "jh", "si", "sm"]
        print(f"validation loss: {average_loss:.5f}, accuracy: {accuracy * 100:2.2f}")
        for i in range(0,10):
            print(f"class_{class_labels[i]}_accuracy: {per_class_accuracy[i] * 100:2.2f}")


def compute_accuracy(
    labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Args:
        labels: ``(batch_size, class_count)`` tensor or array containing example labels
        preds: ``(batch_size, class_count)`` tensor or array containing model prediction
    """
    assert len(labels) == len(preds)
    class_acc = torch.zeros(10)
    for i in range(0,10):
        c = (preds[labels == i] == i).sum()
        d = sum(labels == i)
        if d != 0:
            class_acc[i] = c/d 
        elif c > 0:
            class_acc[i] = 0
        else: 
            class_acc[i] = 100  
         
    return float((labels == preds).sum()) / len(labels), class_acc


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
    tb_log_dir_prefix = (
      f"Kern33_stride11"
      f"CNN_bn_"
      #f"dropout={args.dropout}_"
      f"bs={args.batch_size}_"
      f"lr={args.learning_rate}_"
      #f"momentum={args.sgd_momentum}_" +
      f"run_"
    )
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)


if __name__ == "__main__":
    main(parser.parse_args())

