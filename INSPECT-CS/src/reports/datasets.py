import torch
import numpy as np  
import pandas as pd
from torch.utils.data import Dataset
import h5py

from .utils_data import read_features_from_hdf5

class PEDataset(Dataset):
    def __init__(self, cfg, task, split, transform=None):
        # Load data 
        self.cfg = cfg
        self.cohort = pd.read_csv(cfg.data.cohort_csv.format(task=task))

        # Split the cohort based on the specified split
        filtered_cohort = self.cohort[self.cohort['split'] == split]

        # Get impression IDs and patient IDs
        self.impression_ids = filtered_cohort["ImpressionID"].tolist()
        self.pids = filtered_cohort["PatientID"].tolist()

        # Load labels
        self.labels = filtered_cohort[task].tolist()
        print(f"Unique labels in {task}: {set(self.labels)}")

        # Discard labels and ids with label 'Censored'
        censored_indices = [i for i, label in enumerate(self.labels) if label == 'Censored']
        self.labels = [label for i, label in enumerate(self.labels) if i not in censored_indices]
        self.impression_ids = [impression_id for i, impression_id in enumerate(self.impression_ids) if i not in censored_indices]
        self.pids = [pid for i, pid in enumerate(self.pids) if i not in censored_indices]

        # Convert labels to binary (0 and 1)
        self.labels = [1 if label == True else 0 for label in self.labels]
        print(f"Unique labels in {task}: {set(self.labels)}")
        print(
            f"Pos: {len([t for t in self.labels if t == 1])} ; Neg: {len([t for t in self.labels if t == 0])}"
        )

        # Apply transform
        self.transform = transform

    def init_hdf5(self):
        # called once per worker
        self.images_h5 = h5py.File(self.cfg.data.hdf5_path, 'r', libver='latest', swmr=True)
    
    def __del__(self):
        if hasattr(self, 'images_h5'):
            self.images_h5.close()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = read_features_from_hdf5(self.images_h5, self.impression_ids[idx])
        
        label = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(-1)

        return sample, label, self.pids[idx]
    
    def get_sampler(self):
        neg_class_count = (np.array(self.labels) == 0).sum()
        pos_class_count = (np.array(self.labels) == 1).sum()
        class_weight = [1 / neg_class_count, 1 / pos_class_count]
        weights = [class_weight[i] for i in self.labels]

        weights = torch.Tensor(weights).double()
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True
        )

        return sampler