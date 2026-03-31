import torch
import pandas as pd
import numpy as np
import pickle
import os

from .dataset_base import DatasetBase

class PEDataset(DatasetBase):
    def __init__(self, cfg, split="train", transform=None):
        super().__init__(cfg, split, transform)
        # Load data 
        self.cfg = cfg
        self.cohort = pd.read_csv(cfg.data.cohort_csv.format(task=self.cfg.task))
        self.modalities = cfg.modalities
        assert len(self.modalities) in [2,3], "Modalities length must be 2 or 3"

        # Split the cohort based on the specified split
        filtered_cohort = self.cohort[self.cohort['split'] == split]

        # Get impression IDs, patient IDs and label times
        self.impression_ids = filtered_cohort["ImpressionID"].tolist()
        self.pids = filtered_cohort["PatientID"].tolist()
        self.cohort_label_times = filtered_cohort["StudyTime"].tolist()

        # Load labels
        self.labels = filtered_cohort[cfg.dataset.target].tolist()

        # Discard labels and ids with label 'Censored'
        censored_indices = [i for i, label in enumerate(self.labels) if label == 'Censored']
        self.labels = [label for i, label in enumerate(self.labels) if i not in censored_indices]
        self.impression_ids = [impression_id for i, impression_id in enumerate(self.impression_ids) if i not in censored_indices]
        self.pids = [pid for i, pid in enumerate(self.pids) if i not in censored_indices]

        # Convert labels to binary (0 and 1)
        self.labels = [1 if label == True else 0 for label in self.labels]
        print(
            f"Pos: {len([t for t in self.labels if t == 1])} ; Neg: {len([t for t in self.labels if t == 0])}"
        )

        # Get feture matrix (if ehr modality)
        if "ehr" in self.modalities:
             # Load features matrix
            with open(os.path.join(self.cfg.data.ehr_features_path, self.cfg.task, "ehr_features.pkl"), "rb") as f:
                self.feature_matrix, self.patient_ids, _, self.label_times = pickle.load(f)
            #self.feature_matrix = self.feature_matrix.toarray()  # Convert to dense array if needed

            # Create mapping from (impression_id, label_time) to matrix row index
            self.matrix_index_map = {}
            for i, (patient_id, label_time) in enumerate(zip(self.patient_ids, self.label_times)):
                label_time = pd.to_datetime(label_time)
                self.matrix_index_map[(patient_id, label_time)] = i

        # Apply transform
        self.transform = transform

    def __len__(self):
        return len(self.impression_ids)

    def __getitem__(self, index):
        study = self.impression_ids[index]
        patient_id = self.pids[index]
        label_time = self.cohort_label_times[index]

        x1, x2, y, mask = None, None, None, None

        # Modalità 1
        if self.modalities[0] == "report":
            x1 = self.read_reports_from_hdf5(study)
        elif self.modalities[0] == "image":
            x1 = self.read_images_from_hdf5(study)
            x1, mask = self.fix_series_slice_number(x1)
            if self.cfg.dataset.contextualize_slice:
                x1 = self.contextualize_slice(x1)
            x1 = torch.from_numpy(x1).float()
            mask = torch.tensor(mask).float()
        elif self.modalities[0] == "ehr":
            matrix_idx = self.matrix_index_map[(patient_id, pd.to_datetime(label_time))]
            x1 = torch.tensor(self.feature_matrix[matrix_idx], dtype=torch.float32)

        # Modalità 2
        if self.modalities[1] == "report":
            x2 = self.read_reports_from_hdf5(study)
        elif self.modalities[1] == "image":
            x2 = self.read_images_from_hdf5(study)
            x2, mask = self.fix_series_slice_number(x2)
            if self.cfg.dataset.contextualize_slice:
                x2 = self.contextualize_slice(x2)
            x2 = torch.from_numpy(x2).float()
            mask = torch.tensor(mask).float()
        elif self.modalities[1] == "ehr":
            matrix_idx = self.matrix_index_map[(patient_id, pd.to_datetime(label_time))]
            x2 = torch.tensor(self.feature_matrix[matrix_idx], dtype=torch.float32)
        
        # Target
        y = torch.tensor([self.labels[index]], dtype=torch.float32)
        
        # Modalità 3
        if len(self.modalities) == 3:
            if self.modalities[2] == "report":
                x3 = self.read_reports_from_hdf5(study)
            elif self.modalities[2] == "image":
                x3 = self.read_images_from_hdf5(study)
                x3, mask = self.fix_series_slice_number(x3)
                if self.cfg.dataset.contextualize_slice:
                    x3 = self.contextualize_slice(x3)
                x3 = torch.from_numpy(x3).float()
                mask = torch.tensor(mask).float()
            elif self.modalities[2] == "ehr":
                matrix_idx = self.matrix_index_map[(patient_id, pd.to_datetime(label_time))]
                x3 = torch.tensor(self.feature_matrix[matrix_idx], dtype=torch.float32)
            return x1, x2, x3, y, mask, patient_id, study

        return x1, x2, y, mask, patient_id, study
    
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
    
    def contextualize_slice(self, arr):
        # Make new empty array
        new_arr = np.zeros((arr.shape[0], arr.shape[1] * 3), dtype=np.float32)

        # Fill first third of new array with original features
        for i in range(len(arr)):
            new_arr[i, : arr.shape[1]] = arr[i]

        # Difference between previous neighbor
        new_arr[1:, arr.shape[1] : arr.shape[1] * 2] = (
            new_arr[1:, : arr.shape[1]] - new_arr[:-1, : arr.shape[1]]
        )

        # Difference between next neighbor
        new_arr[:-1, arr.shape[1] * 2 :] = (
            new_arr[:-1, : arr.shape[1]] - new_arr[1:, : arr.shape[1]]
        )

        return new_arr