import torch
import numpy as np
import pandas as pd
# import cv2
# import h5py

from ..constants import *
from .dataset_base import DatasetBase
# from omegaconf import OmegaConf
# from PIL import Image
# from pathlib import Path
# import os
import pickle


class Dataset1D(DatasetBase):
    def __init__(self, cfg, split="train", transform=None):
        super().__init__(cfg, split)
        # Load data 
        self.cfg = cfg
        self.cohort = pd.read_csv(cfg.dataset.cohort_file.format(task=cfg.dataset.target))

        # Split the cohort based on the specified split
        filtered_cohort = self.cohort[self.cohort['split'] == split]

        # Get impression IDs and patient IDs
        self.impression_ids = filtered_cohort["ImpressionID"].tolist()
        self.pids = filtered_cohort["PatientID"].tolist()

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

        if split == "test":
            self.cfg.dataset.sample_strategy = "fix"

        if "rsna" not in cfg.dataset.csv_path:
            self.study = self.impression_ids
        else:
            self.study = self.df["SeriesInstanceUID"].tolist()

    def __getitem__(self, index):
        # read featurized series
        study = self.study[index]
        x = self.read_from_hdf5(study)

        # fix number of slices
        x, mask = self.fix_series_slice_number(x)

        # contextualize slices
        if self.cfg.dataset.contextualize_slice:
            x = self.contextualize_slice(x)

        # create torch tensor
        x = torch.from_numpy(x).float()

        mask = torch.tensor(mask).float()

        # get traget
        y = [self.labels[index]]
        # y = self.pe_labels[index]
        y = torch.tensor(y).float()

        return x, y, mask, self.pids[index]

    def __len__(self):
        return len(self.study)

    def contextualize_slice(self, arr):
        # make new empty array
        new_arr = np.zeros((arr.shape[0], arr.shape[1] * 3), dtype=np.float32)

        # fill first third of new array with original features
        for i in range(len(arr)):
            new_arr[i, : arr.shape[1]] = arr[i]

        # difference between previous neighbor
        new_arr[1:, arr.shape[1] : arr.shape[1] * 2] = (
            new_arr[1:, : arr.shape[1]] - new_arr[:-1, : arr.shape[1]]
        )

        # difference between next neighbor
        new_arr[:-1, arr.shape[1] * 2 :] = (
            new_arr[:-1, : arr.shape[1]] - new_arr[1:, : arr.shape[1]]
        )

        return new_arr

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


class RSNADataset1D(DatasetBase):
    def __init__(self, cfg, split="test", transform=None):
        super().__init__(cfg, split)

        self.cfg = cfg
        self.df = pd.read_csv(cfg.dataset.csv_path)
        if "rsna" not in cfg.dataset.csv_path:
            self.df["patient_datetime"] = self.df.apply(
                lambda x: f"{x.patient_id}_{x.procedure_time}", axis=1
            )
            # duplicate patient_datetime remove
            self.df = self.df.drop_duplicates(subset=["patient_datetime"])

            if split != "all":
                self.df = self.df[self.df["split"] == split]
        elif "rsna" in cfg.dataset.csv_path:
            if split == "test":
                path = "/share/pi/nigam/projects/zphuo/data/PE/inspect/image_modality/anon_pe_features/rsna_hdf5_keys_testsplit.pkl"
                with open(path, "rb") as f:
                    keys = pickle.load(f)
                self.df = self.df[self.df["SeriesInstanceUID"].isin(keys)]

            elif split != "all":
                self.df = self.df[self.df["Split"] == split]

        if split == "test":
            self.cfg.dataset.sample_strategy = "fix"

        # hdf5 path
        model_type = self.cfg.dataset.pretrain_args.model_type
        input_size = self.cfg.dataset.pretrain_args.input_size
        channel_type = self.cfg.dataset.pretrain_args.channel_type
        # self.hdf5_path = os.path.join(
        #     self.cfg.exp.base_dir,
        #     f"{model_type}_{input_size}_{channel_type}_features/"
        #     + f"{model_type}_{input_size}_{channel_type}_features.hdf5",
        # )
        # self.hdf5_path = "/share/pi/nigam/projects/zphuo/data/PE/inspect/image_modality/anon_pe_features_full/features.hdf5"
        self.hdf5_path = self.cfg.dataset.hdf5_path

        # self.cfg.dataset.hdf5_path
        if self.hdf5_path is None:
            raise Exception("Encoded slice HDF5 required")

        if "rsna" not in cfg.dataset.csv_path:
            self.df = self.df[~self.df[cfg.dataset.target].isin(["Censored", "Censor"])]

            self.study = (
                self.df["patient_datetime"]
                .apply(lambda x: x.replace("T", " "))
                .tolist()
            )
        else:
            self.study = self.df["SeriesInstanceUID"].tolist()

        self.df[cfg.dataset.target] = self.df[cfg.dataset.target].astype(str)
        self.labels = [1 if t == "1" else 0 for t in self.df[cfg.dataset.target]]
        print(
            f"Pos: {len([t for t in self.labels if t == 1])} ; Neg: {len([t for t in self.labels if t == 0])}"
        )

    def __getitem__(self, index):
        # read featurized series
        study = self.study[index]
        x = self.read_from_hdf5(study, hdf5_path=self.hdf5_path)

        # fix number of slices
        x, mask = self.fix_series_slice_number(x)

        # contextualize slices
        if self.cfg.dataset.contextualize_slice:
            x = self.contextualize_slice(x)

        # create torch tensor
        x = torch.from_numpy(x).float()

        mask = torch.tensor(mask).float()

        # get traget
        y = [self.labels[index]]
        # y = self.pe_labels[index]
        y = torch.tensor(y).float()

        return x, y, mask, study

    def __len__(self):
        return len(self.study)

    def contextualize_slice(self, arr):
        # make new empty array
        new_arr = np.zeros((arr.shape[0], arr.shape[1] * 3), dtype=np.float32)

        # fill first third of new array with original features
        for i in range(len(arr)):
            new_arr[i, : arr.shape[1]] = arr[i]

        # difference between previous neighbor
        new_arr[1:, arr.shape[1] : arr.shape[1] * 2] = (
            new_arr[1:, : arr.shape[1]] - new_arr[:-1, : arr.shape[1]]
        )

        # difference between next neighbor
        new_arr[:-1, arr.shape[1] * 2 :] = (
            new_arr[:-1, : arr.shape[1]] - new_arr[1:, : arr.shape[1]]
        )

        return new_arr

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
