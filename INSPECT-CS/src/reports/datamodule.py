import pandas as pd
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from reports.datasets import PEDataset
from reports.collator import Collator

def _worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.init_hdf5()

class PEDataModule(LightningDataModule):
    def __init__(self, cfg, task, device):
        super().__init__()
        self.cfg = cfg
        self.task = task
        self.device = device
        self.PEcollator = Collator()

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
                collate_fn=self.PEcollator,
                worker_init_fn=_worker_init_fn
            )
        else:
            return DataLoader(
                dataset,
                batch_size=self.cfg.trainer.batch_size,
                shuffle=True,
                num_workers=self.cfg.device.gpu_num_workers,
                collate_fn=self.PEcollator,
                worker_init_fn=_worker_init_fn
            )

    def val_dataloader(self):
        dataset = PEDataset(self.cfg, self.task, 'valid')
        return DataLoader(
            dataset,
            batch_size=self.cfg.trainer.batch_size,
            shuffle=False,
            num_workers=self.cfg.device.gpu_num_workers,
            collate_fn=self.PEcollator,
            worker_init_fn=_worker_init_fn
        )

    def test_dataloader(self):
        dataset = PEDataset(self.cfg, self.task, 'test')
        return DataLoader(
            dataset,
            batch_size=self.cfg.trainer.batch_size,
            shuffle=False,
            num_workers=self.cfg.device.gpu_num_workers,
            collate_fn=self.PEcollator,
            worker_init_fn=_worker_init_fn
        )