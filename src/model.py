import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)
from torchmetrics.functional.classification import accuracy


class BaseModel(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.train_hist = []
        self.val_hist = []
        self.lr = lr
        # self.adv_train = adv_train
        # self.eps = eps
        # self.eps_iter = eps_iter
        # self.num_iter = num_iter
        # self.norm = norm

    def training_step(self, batch, batch_idx):
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

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([i["loss"] for i in outputs]).mean()
        avg_acc = torch.stack([i["train_acc"] for i in outputs]).mean()
        self.train_hist.append(
            {
                "epoch": self.current_epoch,
                "avg_train_loss": avg_loss.item(),
                "avg_train_acc": avg_acc.item(),
            }
        )

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = F.cross_entropy(output, y)
        acc = accuracy(torch.argmax(output, dim=1), y)
        self.log("val_loss", loss, logger=True)
        self.log("val_acc", acc, logger=True)
        return {"val_loss": loss, "val_acc": acc}

    def validation_epoch_end(self, outputs):
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

    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = F.cross_entropy(output, y)
        acc = accuracy(torch.argmax(output, dim=1), y)
        return {"test_loss": loss, "test_acc": acc}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([i["test_loss"] for i in outputs]).mean()
        avg_acc = torch.stack([i["test_acc"] for i in outputs]).mean()
        self.results = {
            "avg_test_loss": avg_loss.item(),
            "avg_test_acc": avg_acc.item(),
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, _ = batch
        return self(x)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)


class MLP(BaseModel):
    def __init__(self, height, width, in_channels, lr=1e-3):
        super().__init__(lr)
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * height * width, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10),
        )

    def forward(self, x):
        output = self.fc_block(x)
        return output


class SimpleCNN(BaseModel):
    def __init__(self, dataset, in_channels, lr=1e-3):
        super().__init__(lr)
        self.conv1_block = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, 1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )
        self.conv2_block = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1, padding="same"), nn.ReLU(), nn.MaxPool2d((2, 2))
        )
        wh = {
            "cifar10": 8,
            "fashion-mnist": 7,
            "mnist": 7,
        }
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * wh[dataset] * wh[dataset], 1024),
            nn.ReLU(),
            nn.Linear(1024, 10),
        )

    def forward(self, x):
        x = self.conv1_block(x)
        x = self.conv2_block(x)
        output = self.fc_block(x)
        return output
