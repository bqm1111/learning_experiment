import os
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from typing import Tuple
from torch.utils.data import DataLoader
from train_single import TrainerSingle, prepare_const, cifar_dataset, cifar_model
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel


def run(rank, size):
    pass


def init_process(rank, size, backend='gloo'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend=backend, rank=rank, world_size=size)


def cifar_dataloader_ddp(trainset, testset, bs) -> Tuple[DataLoader, DataLoader, DistributedSampler]:
    sampler_train = DistributedSampler(trainset)
    trainloader = DataLoader(
        trainset, batch_size=bs, shuffle=False, sampler=sampler_train, num_workers=8)
    testloader = DataLoader(testset, batch_size=bs, shuffle=False,
                            sampler=DistributedSampler(testset, shuffle=False), num_workers=8)
    return trainloader, testloader, sampler_train


class TrainerDDP(TrainerSingle):
    def __init__(self, gpu_id: int, model: nn.Module, trainloader: DataLoader, testloader: DataLoader, sampler_train: DistributedSampler):
        super().__init__(gpu_id, model, trainloader, testloader)
        torch.cuda.set_device(gpu_id)
        torch.cuda.empty_cache()
        self.model = DistributedDataParallel(self.model, device_ids=[gpu_id])
        self.sampler_train = sampler_train

    def _save_checkpoint(self, epoch: int):
        ckpt = self.model.state_dict()
        model_path = self.const["trained_models"] / \
            f"CIFAR10_ddp_epoch{epoch}.pt"
        torch.save(ckpt, model_path)

    def train(self, max_epochs: int):
        self.model.train()
        for epoch in range(max_epochs):
            self.sampler_train.set_epoch(epoch)

            self._run_epoch(epoch)
            # only save once on master gpu
            if self.gpu_id == 0 and epoch % self.const["save_every"] == 0:
                self._save_checkpoint(epoch)
        # save last epoch
        self._save_checkpoint(max_epochs - 1)

    def test(self, final_model_path: str):
        self.model.load_state_dict(
            torch.load(final_model_path, map_location="cpu")
        )
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


def main_ddp(
    rank: int,
    world_size: int,
    final_model_path: str,
):
    init_process(rank, world_size)  # initialize ddp
    const = prepare_const()
    train_dataset, test_dataset = cifar_dataset(const["data_root"])
    train_dataloader, test_dataloader, train_sampler = cifar_dataloader_ddp(
        train_dataset, test_dataset, const["batch_size"]
    )
    model = cifar_model()
    trainer = TrainerDDP(
        gpu_id=rank,
        model=model,
        trainloader=train_dataloader,
        testloader=test_dataloader,
        sampler_train=train_sampler,
    )
    trainer.train(const["total_epochs"])
    trainer.test(final_model_path)

    dist.destroy_process_group()  # clean up


if __name__ == "__main__":
    from pathlib import Path
    world_size = torch.cuda.device_count()
    final_model_path = Path("./trained_models/CIFAR10_ddp_epoch14.pt")
    mp.spawn(
        main_ddp,
        args=(world_size, final_model_path),
        nprocs=world_size,
    )  # nprocs - total number of processes - # gpus
