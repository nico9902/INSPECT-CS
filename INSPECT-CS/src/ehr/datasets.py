import torch
import numpy as np  
import pandas as pd
import pickle
import os
from torch.utils.data import Dataset

class PEDataset(Dataset):
    def __init__(self, cfg, task, split, transform=None):
        # Load data 
        self.cfg = cfg
        self.cohort = pd.read_csv(cfg.data.cohort_csv.format(task=task))

        # Split the cohort based on the specified split
        filtered_cohort = self.cohort
        if split != "all":
            filtered_cohort = self.cohort[self.cohort['split'] == split]

        # Get impression IDs and patient IDs
        self.impression_ids = filtered_cohort["ImpressionID"].tolist()
        self.pids = filtered_cohort["PatientID"].tolist()
        self.cohort_label_times = filtered_cohort["StudyTime"].tolist()

        # Load features matrix
        with open(os.path.join(self.cfg.data.pkl, self.cfg.task, "filtered_featurized_patients.pkl"), "rb") as f:
            self.feature_matrix, self.patient_ids, _, self.label_times = pickle.load(f)
        if not isinstance(self.feature_matrix, np.ndarray):
            if hasattr(self.feature_matrix, "toarray"):
                self.feature_matrix = self.feature_matrix.toarray()  # Convert to dense array if needed

        # Create mapping from (impression_id, label_time) to matrix row index
        self.matrix_index_map = {}
        for i, (patient_id, label_time) in enumerate(zip(self.patient_ids, self.label_times)):
            label_time = pd.to_datetime(label_time)
            self.matrix_index_map[(patient_id, label_time)] = i

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

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Get the correct row from the pre-loaded feature matrix
        # impression_id = self.impression_ids[idx]
        patient_id = self.pids[idx]
        label_time = self.cohort_label_times[idx]
        
        # Find the matrix row index using the mapping
        matrix_idx = self.matrix_index_map[(patient_id, pd.to_datetime(label_time))]

        # Get features from the pre-loaded matrix
        sample = torch.tensor(self.feature_matrix[matrix_idx], dtype=torch.float32)

        if self.transform:
            sample = self.transform(sample)
        
        label = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(-1)

        return sample, label, self.pids[idx], label_time
    
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