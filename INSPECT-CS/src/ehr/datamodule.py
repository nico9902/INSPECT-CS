import pandas as pd
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from ehr.datasets import PEDataset

class PEDataModule(LightningDataModule):
    def __init__(self, cfg, task, device):
        super().__init__()
        self.cfg = cfg
        self.task = task
        self.device = device

    def train_dataloader(self):
        dataset = PEDataset(self.cfg, self.task, 'train')
        if self.cfg.data.weighted_sample:
            sampler = dataset.get_sampler()
            return DataLoader(
                dataset,
                shuffle=False,
                sampler=sampler,
                batch_size=self.cfg.trainer.batch_size,
                num_workers=self.cfg.device.gpu_num_workers,
            )
        else:
            return DataLoader(
                dataset,
                batch_size=self.cfg.trainer.batch_size,
                shuffle=True,
                num_workers=self.cfg.device.gpu_num_workers,
            )

    def val_dataloader(self):
        dataset = PEDataset(self.cfg, self.task, 'valid')
        return DataLoader(
            dataset,
            batch_size=self.cfg.trainer.batch_size,
            shuffle=False,
            num_workers=self.cfg.device.gpu_num_workers,
        )

    def test_dataloader(self):
        dataset = PEDataset(self.cfg, self.task, 'test')
        return DataLoader(
            dataset,
            batch_size=self.cfg.trainer.batch_size,
            shuffle=False,
            num_workers=self.cfg.device.gpu_num_workers,
        )
    
    def all_dataloader(self):
        dataset = PEDataset(self.cfg, self.task, 'all')
        return DataLoader(
            dataset,
            batch_size=self.cfg.trainer.batch_size,
            shuffle=False,
            num_workers=self.cfg.device.gpu_num_workers,
        )