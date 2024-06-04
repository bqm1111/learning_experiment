from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet34, resnet18
from torchvision.transforms import transforms
from torchvision.datasets import CIFAR10
import torch.optim as optim
from torch import Tensor
from typing import Iterator, Tuple
import torchmetrics
from tqdm import tqdm

def prepare_const() -> dict:
    """Data and model directory + Training hyperparameters"""
    data_root = Path("data")
    trained_models = Path("trained_models")

    if not data_root.exists():
        data_root.mkdir()

    if not trained_models.exists():
        trained_models.mkdir()

    const = dict(
        data_root=data_root,
        trained_models=trained_models,
        total_epochs=15,
        batch_size=256,
        lr=0.1,  # learning rate
        momentum=0.9,
        lr_step_size=5,
        save_every=3,
    )

    return const


def cifar_model() -> nn.Module:
    model = resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                            stride=2, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def cifar_dataset(data_root: Path) -> Tuple[Dataset, Dataset]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.49139968, 0.48215827, 0.44653124),
                std=(0.24703233, 0.24348505, 0.26158768),
            ),
        ]
    )

    trainset = CIFAR10(root=data_root, train=True,
                       transform=transform, download=True)
    testset = CIFAR10(root=data_root, train=False,
                      transform=transform, download=True)

    return trainset, testset


def cifar_dataloader_single(
    trainset: Dataset, testset: Dataset, bs: int
) -> Tuple[DataLoader, DataLoader]:
    trainloader = DataLoader(trainset, batch_size=bs,
                             shuffle=True, num_workers=8)
    testloader = DataLoader(testset, batch_size=bs,
                            shuffle=False, num_workers=8)

    return trainloader, testloader


class TrainerSingle:
    def __init__(
        self,
        gpu_id: int,
        model: nn.Module,
        trainloader: DataLoader,
        testloader: DataLoader,
    ):
        self.gpu_id = gpu_id

        self.const = prepare_const()
        self.model = model.to(self.gpu_id)
        self.trainloader = trainloader
        self.testloader = testloader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.const["lr"],
            momentum=self.const["momentum"],
        )
        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, self.const["lr_step_size"]
        )
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=10, average="micro"
        ).to(self.gpu_id)

        self.valid_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=10, average="micro"
        ).to(self.gpu_id)

    def _run_batch(self, src: Tensor, tgt: Tensor) -> float:
        self.optimizer.zero_grad()

        out = self.model(src)
        loss = self.criterion(out, tgt)
        loss.backward()
        self.optimizer.step()

        self.train_acc.update(out, tgt)
        return loss.item()

    def _run_epoch(self, epoch: int):
        loss = 0.0
        for src, tgt in self.trainloader:
            src = src.to(self.gpu_id)
            tgt = tgt.to(self.gpu_id)
            loss_batch = self._run_batch(src, tgt)
            loss += loss_batch
        self.lr_scheduler.step()

        print(
            f"{'-' * 90}\n[GPU{self.gpu_id}] Epoch {epoch:2d} | Batchsize: {self.const['batch_size']} | Steps: {len(self.trainloader)} | LR: {self.optimizer.param_groups[0]['lr']:.4f} | Loss: {loss / len(self.trainloader):.4f} | Acc: {100 * self.train_acc.compute().item():.2f}%",
            flush=True,
        )

        self.train_acc.reset()

    def _save_checkpoint(self, epoch: int):
        ckp = self.model.state_dict()
        model_path = self.const["trained_models"] / \
            f"CIFAR10_single_epoch{epoch}.pt"
        torch.save(ckp, model_path)

    def train(self, max_epochs: int):
        self.model.train()
        for epoch in tqdm(range(max_epochs)):
            self._run_epoch(epoch)
            if epoch % self.const["save_every"] == 0:
                self._save_checkpoint(epoch)
        # save last epoch
        self._save_checkpoint(max_epochs - 1)

    def test(self, final_model_path: str):
        self.model.load_state_dict(torch.load(final_model_path))
        self.model.eval()
        with torch.no_grad():
            for src, tgt in self.testloader:
                src = src.to(self.gpu_id)
                tgt = tgt.to(self.gpu_id)
                out = self.model(src)
                self.valid_acc.update(out, tgt)
        print(
            f"[GPU{self.gpu_id}] Test Acc: {100 * self.valid_acc.compute().item():.4f}%"
        )


def main_single(gpu_id: int, final_model_path: str):
    const = prepare_const()
    train_dataset, test_dataset = cifar_dataset(const["data_root"])
    train_dataloader, test_dataloader = cifar_dataloader_single(
        train_dataset, test_dataset, const["batch_size"]
    )
    model = cifar_model()
    trainer = TrainerSingle(
        gpu_id=gpu_id,
        model=model,
        trainloader=train_dataloader,
        testloader=test_dataloader,
    )
    trainer.train(const["total_epochs"])
    trainer.test(final_model_path)


if __name__ == "__main__":
    gpu_id = 0
    final_model_path = Path("./trained_models/CIFAR10_single_epoch14.pt")
    main_single(gpu_id, final_model_path)
