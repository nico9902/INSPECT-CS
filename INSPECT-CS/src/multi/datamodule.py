import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from .datasets import PEDataset
from .collator import Collator

def _worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.init_hdf5()

class PEDataModule(LightningDataModule):
    def __init__(self, cfg, device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.PEcollator = Collator(modalities=self.cfg.modalities)
        self.dataset = PEDataset

    def train_dataloader(self):
        dataset = self.dataset(self.cfg, split="train")
        if self.cfg.data.weighted_sample:
            return DataLoader(
                dataset,
                sampler = dataset.get_sampler(),
                batch_size=self.cfg.trainer.batch_size,
                pin_memory=True,
                drop_last=True,
                shuffle=False,
                num_workers=self.cfg.device.gpu_num_workers,
                collate_fn=self.PEcollator,
                worker_init_fn=_worker_init_fn
            )
        else:
            return DataLoader(
                dataset,
                batch_size=self.cfg.trainer.batch_size,
                pin_memory=True,
                drop_last=True,
                shuffle=True,
                num_workers=self.cfg.device.gpu_num_workers,
                collate_fn=self.PEcollator,
                worker_init_fn=_worker_init_fn
            )

    def val_dataloader(self):
        dataset = self.dataset(self.cfg, split="valid")
        return DataLoader(
            dataset,
            batch_size=self.cfg.trainer.batch_size,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
            num_workers=self.cfg.device.gpu_num_workers,
            collate_fn=self.PEcollator,
            worker_init_fn=_worker_init_fn
        )

    def test_dataloader(self):
        dataset = self.dataset(self.cfg, split="test")
        return DataLoader(
            dataset,
            batch_size=self.cfg.trainer.batch_size,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
            num_workers=self.cfg.device.gpu_num_workers,
            collate_fn=self.PEcollator,
            worker_init_fn=_worker_init_fn
        )