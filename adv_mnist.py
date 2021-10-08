import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)
from easydict import EasyDict
from torchinfo import summary

from datasets import MNISTDataset


class CNN(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 5 * 5, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def load_mnist(batch_size):
    train_transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]
    )
    test_transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]
    )
    train_dataset = MNISTDataset(root="/tmp/data", transform=train_transforms)
    test_dataset = MNISTDataset(
        root="/tmp/data", train=False, transform=test_transforms
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    return EasyDict(train=train_loader, test=test_loader)


CONFIG = EasyDict(
    max_epochs=8, adv_train=True, eps=0.3, learning_rate=1e-3, batch_size=128
)
data = load_mnist(CONFIG.batch_size)
model = CNN(in_channels=1)
print(summary(model, input_size=(CONFIG.batch_size, 1, 28, 28)))

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    model = model.cuda()
loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG.learning_rate)

model.train()
for epoch in range(1, CONFIG.max_epochs + 1):
    train_loss = 0.0
    for x, y in data.train:
        x, y = x.to(device), y.to(device)
        if CONFIG.adv_train:
            x = projected_gradient_descent(model, x, CONFIG.eps, 0.01, 40, np.inf)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print(
        "epoch: {}/{}, train loss: {:.3f}".format(epoch, CONFIG.max_epochs, train_loss)
    )

model.eval()
report = EasyDict(nb_test=0, correct=0, correct_fgm=0, correct_pgd=0)
for x, y in data.test:
    x, y = x.to(device), y.to(device)
    x_fgm = fast_gradient_method(model, x, CONFIG.eps, np.inf)
    x_pgd = projected_gradient_descent(model, x, CONFIG.eps, 0.01, 40, np.inf)
    _, y_pred = model(x).max(1)
    _, y_pred_fgm = model(x_fgm).max(1)
    _, y_pred_pgd = model(x_pgd).max(1)
    report.nb_test += y.size(0)
    report.correct += y_pred.eq(y).sum().item()
    report.correct_fgm += y_pred_fgm.eq(y).sum().item()
    report.correct_pgd += y_pred_pgd.eq(y).sum().item()
print(
    "test acc on clean examples (%): {:.3f}".format(
        report.correct / report.nb_test * 100.0
    )
)
print(
    "test acc on FGM adversarial examples (%): {:.3f}".format(
        report.correct_fgm / report.nb_test * 100.0
    )
)
print(
    "test acc on PGD adversarial examples (%): {:.3f}".format(
        report.correct_pgd / report.nb_test * 100.0
    )
)
