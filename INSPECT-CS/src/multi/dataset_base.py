import torch
import pickle
import numpy as np
import pandas as pd
import h5py
from torch.utils.data import Dataset
from hydra.utils import to_absolute_path


class DatasetBase(Dataset):
    def __init__(self, cfg, split="train", transform=None):
        self.cfg = cfg
        self.transform = transform
        self.split = split

        path = to_absolute_path("data/dict_slice_thickness.pkl")
        self.dict_slice_thickness = pickle.load(open(path, "rb"))

    def init_hdf5(self):
        # called once per worker
        self.reports_h5 = h5py.File(self.cfg.data.reports_hdf5_path, 'r', libver='latest', swmr=True)
        self.images_h5  = h5py.File(self.cfg.data.image_hdf5_path,  'r', libver='latest', swmr=True)

    def __del__(self):
        if hasattr(self, 'images_h5'): self.images_h5.close()
        if hasattr(self, 'reports_h5'): self.reports_h5.close()

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def read_images_from_hdf5(self, key, slice_idx=None):
        key = str(key)
        f = self.images_h5
        if slice_idx is None:
            arr = f[key][:]
        else:
            arr = f[key][slice_idx]

        thickness_ls = []
        for idx_th in range(arr.shape[0]):
            try:
                thickness_ls.append(self.dict_slice_thickness[key] * idx_th)
            except:
                print(
                    key,
                    idx_th,
                    "=========no thickness info=============================",
                )
                thickness_ls.append(0)
        thickness_ls = np.array(thickness_ls)
        arr = np.concatenate([arr, thickness_ls[:, None]], axis=1)
        return arr
    

    def read_reports_from_hdf5(self, key):
        """
        Reads features for a given impression_id from an HDF5 file.
        """
        key = str(key)
        f = self.reports_h5
        if key in f:
            features = f[key][:]
        else:
            raise KeyError(f"Impression ID {key} not found in the HDF5 file.")
        return features

    def fix_slice_number(self, df: pd.DataFrame):
        num_slices = min(self.cfg.dataset.num_slices, df.shape[0])
        if self.cfg.dataset.sample_strategy == "random":
            slice_idx = np.random.choice(
                np.arange(df.shape[0]), replace=False, size=num_slices
            )
            slice_idx = list(np.sort(slice_idx))
            df = df.iloc[slice_idx, :]
        elif self.cfg.dataset.sample_strategy == "fix":
            df = df.iloc[:num_slices, :]
        else:
            raise Exception("Sampling strategy either 'random' or 'fix'")
        return df

    def fix_series_slice_number(self, series):
        num_slices = min(self.cfg.dataset.num_slices, series.shape[0])
        if num_slices == self.cfg.dataset.num_slices:
            if self.cfg.dataset.sample_strategy == "random":
                slice_idx = np.random.choice(
                    np.arange(series.shape[0]), replace=False, size=num_slices
                )
                slice_idx = list(np.sort(slice_idx))
                features = series[slice_idx, :]
            elif self.cfg.dataset.sample_strategy == "fix":
                pad = int((series.shape[0] - num_slices) / 2)  # select middle slices
                start = pad
                end = pad + num_slices
                features = series[start:end, :]
            else:
                raise Exception("Sampling strategy either 'random' or 'fix'")
            mask = np.ones(num_slices)
        else:
            mask = np.zeros(self.cfg.dataset.num_slices)
            mask[:num_slices] = 1
            shape = [self.cfg.dataset.num_slices] + list(series.shape[1:])
            features = np.zeros(shape)

            features[:num_slices] = series

        return features, mask

    def fill_series_to_num_slicess(self, series, num_slices):
        x = torch.zeros(()).new_full((num_slices, *series.shape[1:]), 0.0)
        x[: series.shape[0]] = series
        return x