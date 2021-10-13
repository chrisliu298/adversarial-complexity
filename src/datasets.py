import torch
import numpy as np
from numpy.random import choice
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import Compose, ToTensor


class ImageDataModule(LightningDataModule):
    def __init__(self, dataset_fn, batch_size=128, num_workers=2):
        super().__init__()
        self.save_hyperparameters()
        self.transforms = Compose([ToTensor()])
        self.dataset_fn = dataset_fn
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self, train_size):
        # Load raw train and test datasets
        raw_train_dataset = self.dataset_fn(
            root="/tmp/data", train=True, transform=self.transforms, download=True
        )
        raw_test_dataset = self.dataset_fn(
            root="/tmp/data", train=False, transform=self.transforms, download=True
        )
        # Define random sample indices and train/val splits
        sample_idx = choice(len(raw_train_dataset), train_size, replace=False)
        train_split_idx, val_split_idx = train_test_split(
            np.arange(len(sample_idx)), test_size=0.1, shuffle=False
        )
        assert len(np.intersect1d(train_split_idx, val_split_idx)) == 0
        # Convert to tensor dataset for indexing
        train_dataset_sample = TensorDataset(
            torch.unsqueeze(raw_train_dataset.data, dim=1), raw_train_dataset.targets
        )[sample_idx]
        self.train_dataset = TensorDataset(
            train_dataset_sample[0][train_split_idx],
            train_dataset_sample[1][train_split_idx],
        )
        self.val_dataset = TensorDataset(
            train_dataset_sample[0][val_split_idx],
            train_dataset_sample[1][val_split_idx],
        )
        self.test_dataset = TensorDataset(
            torch.unsqueeze(raw_test_dataset.data, dim=1), raw_test_dataset.targets
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


# def load_mnist(batch_size):
#     """Load training and test data."""
#     train_transforms = Compose([ToTensor()])
#     test_transforms = Compose([ToTensor()])
#     train_dataset = MNIST(
#         root="/tmp/data", train=True, transform=train_transforms, download=True
#     )
#     test_dataset = MNIST(
#         root="/tmp/data", train=False, transform=test_transforms, download=True
#     )
#     train_loader = DataLoader(
#         train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
#     )
#     test_loader = DataLoader(
#         test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
#     )
#     return EasyDict(train=train_loader, test=test_loader)


# def load_cifar10(batch_size):
#     """Load training and test data."""
#     train_transforms = Compose([ToTensor()])
#     test_transforms = Compose([ToTensor()])
#     train_dataset = CIFAR10(
#         root="/tmp/data", train=True, transform=train_transforms, download=True
#     )
#     test_dataset = CIFAR10(
#         root="/tmp/data", train=False, transform=test_transforms, download=True
#     )
#     train_loader = DataLoader(
#         train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
#     )
#     test_loader = DataLoader(
#         test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
#     )
#     return EasyDict(train=train_loader, test=test_loader)
