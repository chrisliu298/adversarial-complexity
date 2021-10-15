from math import floor
from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)
from torchmetrics.functional.classification import accuracy
from torchvision.models import resnet18, resnet34, resnet50


class BaseModel(pl.LightningModule):
    def __init__(self, lr: float):
        super().__init__()
        self.train_hist = []
        self.val_hist = []
        self.lr = lr
        # self.adv_train = adv_train
        # self.eps = eps
        # self.eps_iter = eps_iter
        # self.num_iter = num_iter
        # self.norm = norm

    def training_step(self, batch, batch_idx) -> Dict:
        x, y = batch
        # if self.adv_train:
        #     x = projected_gradient_descent(
        #         model_fn=self(),
        #         x=x,
        #         eps=self.eps,
        #         eps_iter=self.eps_iter,
        #         nb_iter=self.num_iter,
        #         norm=self.norm,
        #     )

        output = self(x)
        loss = F.cross_entropy(output, y)
        acc = accuracy(torch.argmax(output, dim=1), y)
        self.log("train_loss", loss, logger=True)
        self.log("train_acc", acc, logger=True)
        return {"loss": loss, "train_acc": acc}

    def training_epoch_end(self, outputs) -> None:
        avg_loss = torch.stack([i["loss"] for i in outputs]).mean()
        avg_acc = torch.stack([i["train_acc"] for i in outputs]).mean()
        self.train_hist.append(
            {
                "epoch": self.current_epoch,
                "avg_train_loss": avg_loss.item(),
                "avg_train_acc": avg_acc.item(),
            }
        )

    def validation_step(self, batch, batch_idx) -> Dict:
        x, y = batch
        output = self(x)
        loss = F.cross_entropy(output, y)
        acc = accuracy(torch.argmax(output, dim=1), y)
        self.log("val_loss", loss, logger=True)
        self.log("val_acc", acc, logger=True)
        return {"val_loss": loss, "val_acc": acc}

    def validation_epoch_end(self, outputs) -> None:
        avg_loss = torch.stack([i["val_loss"] for i in outputs]).mean()
        avg_acc = torch.stack([i["val_acc"] for i in outputs]).mean()
        self.log("avg_val_acc", avg_acc, logger=True)
        self.val_hist.append(
            {
                "epoch": self.current_epoch,
                "avg_val_loss": avg_loss.item(),
                "avg_val_acc": avg_acc.item(),
            }
        )

    def test_step(self, batch, batch_idx) -> Dict:
        x, y = batch
        output = self(x)
        loss = F.cross_entropy(output, y)
        acc = accuracy(torch.argmax(output, dim=1), y)
        return {"test_loss": loss, "test_acc": acc}

    def test_epoch_end(self, outputs) -> None:
        avg_loss = torch.stack([i["test_loss"] for i in outputs]).mean()
        avg_acc = torch.stack([i["test_acc"] for i in outputs]).mean()
        self.results = {
            "avg_test_loss": avg_loss.item(),
            "avg_test_acc": avg_acc.item(),
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=None) -> torch.Tensor:
        x, _ = batch
        return self(x)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)


class MLP(BaseModel):
    def __init__(
        self,
        height: int,
        width: int,
        in_channels: int,
        output_dim: int,
        model_size: str,
        lr: float = 1e-3,
    ):
        super().__init__(lr)
        model_sizes = {
            "small": 1024,
            "medium": 2048,
            "large": 4096,
        }
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * height * width, model_sizes[model_size]),
            nn.ReLU(),
            nn.Linear(model_sizes[model_size], output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.fc_block(x)
        return output


class SimpleCNN(BaseModel):
    def __init__(
        self,
        dataset: str,
        in_channels: int,
        height: int,
        width: int,
        output_dim: int,
        model_size: str,
        lr: float = 1e-3,
    ):
        super().__init__(lr)
        model_sizes = {
            "small": [32, 64, 1024],
            "medium": [64, 128, 2048],
            "large": [128, 256, 4096],
        }
        self.conv1_block = nn.Sequential(
            nn.Conv2d(in_channels, model_sizes[model_size][0], 5, 1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )
        self.conv2_block = nn.Sequential(
            nn.Conv2d(
                model_sizes[model_size][0],
                model_sizes[model_size][1],
                5,
                1,
                padding="same",
            ),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                model_sizes[model_size][1]
                * self.compute_wh(height)
                * self.compute_wh(width),
                model_sizes[model_size][2],
            ),
            nn.ReLU(),
            nn.Linear(model_sizes[model_size][2], output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1_block(x)
        x = self.conv2_block(x)
        output = self.fc_block(x)
        return output

    def compute_wh(self, in_wh):
        return floor(floor(in_wh / 2) / 2)


class ResNet(BaseModel):
    def __init__(
        self, in_channels: int, output_dim: int, layers: int, lr: float = 1e-3
    ):
        super().__init__(lr)
        resnets = {
            18: resnet18,
            34: resnet34,
            50: resnet50,
        }
        self.resnet = resnets[layers](pretrained=True)
        self.resnet.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.resnet.fc = (
            nn.Linear(2048, output_dim) if layers == 50 else nn.Linear(512, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)
