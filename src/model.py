from math import floor
from typing import Dict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)
from easydict import EasyDict
from torchmetrics.functional.classification import accuracy
from torchvision.models import resnet18, resnet34, resnet50


class BaseModel(pl.LightningModule):
    def __init__(self, lr: float, adv_config: EasyDict):
        super().__init__()
        self.train_hist = []
        self.val_hist = []
        self.lr = lr
        self.automatic_optimization = False

        self.attack_type = adv_config.attack_type
        self.adv_train_mode = adv_config.adv_train_mode
        self.adv_test_mode = adv_config.adv_test_mode
        self.eps = adv_config.eps
        self.eps_iter = adv_config.eps_iter
        self.nb_iter = adv_config.nb_iter

    def training_step(self, batch, batch_idx) -> Dict:
        x, y = batch
        if self.adv_train_mode:
            assert self.attack_type in ["fgsm", "pgd"]
            if self.attack_type == "fgsm":
                x = fast_gradient_method(model_fn=self, x=x, eps=self.eps, norm=np.inf)
            elif self.attack_type == "pgd":
                x = projected_gradient_descent(
                    model_fn=self,
                    x=x,
                    eps=self.eps,
                    eps_iter=self.eps_iter,
                    nb_iter=self.nb_iter,
                    norm=np.inf,
                )
        optimizer = self.optimizers()
        optimizer.zero_grad()
        output = self(x)
        loss = F.cross_entropy(output, y)
        acc = accuracy(torch.argmax(output, dim=1), y)
        self.manual_backward(loss)
        optimizer.step()
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
        torch.set_grad_enabled(True)
        x, y = batch
        if self.adv_train_mode:
            if self.attack_type == "fgsm":
                x = fast_gradient_method(model_fn=self, x=x, eps=self.eps, norm=np.inf)
            elif self.attack_type == "pgd":
                x = projected_gradient_descent(
                    model_fn=self,
                    x=x,
                    eps=self.eps,
                    eps_iter=self.eps_iter,
                    nb_iter=self.nb_iter,
                    norm=np.inf,
                )
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
        torch.set_grad_enabled(True)
        x, y = batch
        if self.adv_test_mode:
            if self.attack_type == "fgsm":
                x = fast_gradient_method(model_fn=self, x=x, eps=self.eps, norm=np.inf)
            elif self.attack_type == "pgd":
                x = projected_gradient_descent(
                    model_fn=self,
                    x=x,
                    eps=self.eps,
                    eps_iter=self.eps_iter,
                    nb_iter=self.nb_iter,
                    norm=np.inf,
                )
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
        adv_config,
        lr: float = 1e-3,
    ):
        super().__init__(lr, adv_config)
        model_sizes = {
            "small": 512,
            "medium": 1024,
            "large": 2048,
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
        in_channels: int,
        height: int,
        width: int,
        output_dim: int,
        model_size: str,
        adv_config: EasyDict,
        lr: float = 1e-3,
    ):
        super().__init__(lr, adv_config)
        model_sizes = {
            "small": [32, 64, 512],
            "medium": [64, 128, 1024],
            "large": [128, 256, 2048],
        }
        self.conv1_block = nn.Sequential(
            nn.Conv2d(in_channels, model_sizes[model_size][0], 5, 1, padding="same"),
            nn.BatchNorm2d(model_sizes[model_size][0]),
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
            nn.BatchNorm2d(model_sizes[model_size][1]),
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
        self,
        in_channels: int,
        output_dim: int,
        layers: int,
        adv_config: EasyDict,
        lr: float = 1e-3,
    ):
        super().__init__(lr, adv_config)
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
