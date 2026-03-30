import torch
import numpy as np
import pandas as pd
import tqdm
import os
import nibabel as nib
import cv2
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
from pathlib import Path
from ..constants import *
from .dataset_base import DatasetBase

class InferenceDataset2D(DatasetBase):  
    def __init__(self, cfg, split="train", transform=None):
        super().__init__(cfg, split)

        self.cfg = cfg
        self.split = split
        self.transform = transform

        self.df = pd.read_csv(cfg.dataset.csv_path)

        # Read splits if split_path is provided
        if hasattr(cfg.dataset, 'split_path') and cfg.dataset.split_path:
            splits_df = pd.read_csv(cfg.dataset.split_path)
            # Merge splits with main dataframe
            self.df = self.df.merge(splits_df, on='impression_id', how='left')

        # Filter by split
        if self.split != "all" and 'split' in self.df.columns:
            self.df = self.df[self.df["split"] == self.split]

        # get all nifti files for each impression_id
        self.all_instances = []
        for _, row in tqdm.tqdm(self.df.iterrows(), total=len(self.df)):
            nifti_path = os.path.join(self.cfg.dataset.dicom_dir, f"{row['impression_id']}.nii.gz")  # image_id already has .nii.gz
            self.all_instances.append([row["impression_id"], nifti_path])  # Store slice-level info
    
    def __getitem__(self, index):
        # get impression row
        pdt, nifti_path = self.all_instances[index]
        # Read image based on format
        if nifti_path.endswith('.nii.gz'):
            if str(pdt) == "PE452890e":
                print(f"Skip {nifti_path}")
                return None, None, pdt
            # Read NIfTI
            nifti_img = nib.load(nifti_path)
            pixel_array = nifti_img.get_fdata()
            # print(f"Pixel array shape: {pixel_array.shape} and path {nifti_path}")

            # Process all slices in parallel
            with ThreadPoolExecutor(max_workers=16) as executor:
                slices = list(executor.map(
                    lambda idx: self.process_and_transform_slice(pixel_array, idx),
                    range(pixel_array.shape[2])
                ))

            # Convert all slices to a tensor
            ct_array = torch.stack(slices)

            # get labels
            y = torch.tensor([0])
        else:
            raise ValueError(f"Unsupported file format: {nifti_path}")

        return ct_array, y, f"{pdt}"   
    
    def __len__(self):
        return len(self.all_instances)
    
    def process_and_transform_slice(self, pixel_array, slice_idx):
        """Process and transform a single slice."""
        # Process slice
        ct_slice = self.process_slice(pixel_array=pixel_array, slice_idx=slice_idx)

        # Transform
        if ct_slice.shape[0] == 3:
            ct_slice = np.transpose(ct_slice, (1, 2, 0))
            ct_slice = Image.fromarray(np.uint8(ct_slice * 255))
        else:
            ct_slice = Image.fromarray(np.uint8(ct_slice * 255)).convert("RGB")
        x = self.transform(ct_slice) if self.transform else ct_slice

        # Check dimension
        if x.shape[0] == 1:  # for repeat
            c, w, h = list(x.shape)
            x = x.expand(3, w, h)
        x = x.type(torch.FloatTensor)

        return x
    
    def process_slice(self, slice_info: pd.Series = None, dicom_dir: Path = None, pixel_array = None, slice_idx: int = None):
        """process slice with windowing, resize and transforms"""
        resize_size = self.cfg.dataset.transform.resize_size
        # Read image
        if len(pixel_array.shape) == 3:
            slice_array = pixel_array[:, :, slice_idx]
        elif len(pixel_array.shape) == 4:
            slice_array = pixel_array[:, :, slice_idx, 0]
        # Resize image
        if resize_size != slice_array.shape[-1]:
            slice_array = cv2.resize(
                slice_array, (resize_size, resize_size), interpolation=cv2.INTER_AREA
            )
        # window
        if self.cfg.dataset.transform.channels == "repeat":
            ct_slice = self.windowing(slice_array, 400, 1000)
        else:
            ct_slice = [
                self.windowing(slice_array, -600, 1500),  # LUNG window
                self.windowing(slice_array, 400, 1000),   # PE window
                self.windowing(slice_array, 40, 400),     # MEDIASTINAL window
            ]
            ct_slice = np.stack(ct_slice)
        
        return ct_slice
    
class Dataset2D(DatasetBase):
    def __init__(self, cfg, split="train", transform=None):
        super().__init__(cfg, split)

        self.cfg = cfg
        self.split = split
        self.transform = transform

        self.df = pd.read_csv(cfg.dataset.csv_path)

        # Read splits if split_path is provided
        if hasattr(cfg.dataset, 'split_path') and cfg.dataset.split_path:
            splits_df = pd.read_csv(cfg.dataset.split_path)
            # Merge splits with main dataframe
            self.df = self.df.merge(splits_df, on='impression_id', how='left')

        # Filter by split
        if self.split != "all" and 'split' in self.df.columns:
            self.df = self.df[self.df["split"] == self.split]

        # if self.split != "all":
        #     self.df = self.df[self.df["split"] == self.split]

        if self.split == "train":
            if cfg.dataset.sample_frac < 1.0:
                all_ids = list(self.df["impression_id"].unique())
                num_sample = int(len(all_ids) * cfg.dataset.sample_frac)
                sampled_ids = np.random.choice(all_ids, num_sample, replace=False)
                self.df = self.df[self.df["impression_id"].isin(sampled_ids)]
        
        if self.split == "valid":
            all_ids = list(self.df["impression_id"].unique())
            num_sample = int(len(all_ids) * 0.1)
            sampled_ids = np.random.choice(all_ids, num_sample, replace=False)
            self.df = self.df[self.df["impression_id"].isin(sampled_ids)]

        # get all nifti files for each impression_id
        self.all_instances = []
        for idx, row in tqdm.tqdm(self.df.iterrows(), total=len(self.df)):
            nifti_path = os.path.join(self.cfg.dataset.dicom_dir, f"{row['impression_id']}.nii.gz")  # image_id already has .nii.gz
            num_slices = self.get_num_slices(nifti_path)  # Function to get the number of slices in the NIfTI file
            for slice_idx in range(num_slices):
                self.all_instances.append([row["impression_id"], slice_idx, nifti_path])  # Store slice-level info
    
    def __getitem__(self, index):
        # get slice row
        pdt, slice_idx, nifti_path = self.all_instances[index]

        # Process slice on demand
        ct_slice = self.process_slice(slice_path=nifti_path, slice_idx=slice_idx)

        # transform
        if ct_slice.shape[0] == 3:
            ct_slice = np.transpose(ct_slice, (1, 2, 0))
            ct_slice = Image.fromarray(np.uint8(ct_slice * 255))
        else:
            ct_slice = Image.fromarray(np.uint8(ct_slice * 255)).convert("RGB")
        x = self.transform(ct_slice)

        # check dimension
        if x.shape[0] == 1:  # for repeat
            c, w, h = list(x.shape)
            x = x.expand(3, w, h)
        x = x.type(torch.FloatTensor)

        # get labels
        y = torch.tensor([0])

        return x, y, f"{pdt}@{slice_idx}"

    def __len__(self):
        return len(self.all_instances)
    
    def process_slice(self, slice_info: pd.Series = None, dicom_dir: Path = None, slice_path: str = None, slice_idx: int = None):
        """process slice with windowing, resize and transforms"""
        if slice_path is None:
            slice_path = dicom_dir / slice_info[INSTANCE_PATH_COL]
        slice_array = self.read_image(slice_path, slice_idx)  # Use read_image instead of read_dicom

        # window
        if self.cfg.dataset.transform.channels == "repeat":
            ct_slice = self.windowing(slice_array, 400, 1000)  # use PE window by default
        else:
            ct_slice = [
                self.windowing(slice_array, -600, 1500),  # LUNG window
                self.windowing(slice_array, 400, 1000),  # PE window
                self.windowing(slice_array, 40, 400),  # MEDIASTINAL window
            ]
            ct_slice = np.stack(ct_slice)

        return ct_slice


class RSNADataset2D(DatasetBase):
    def __init__(self, cfg, split="train", transform=None):
        super().__init__(cfg, split)

        self.df = pd.read_csv(cfg.dataset.csv_path)
        self.transform = transform
        self.cfg = cfg
        self.label_name = "pe_present_on_image"

        # only use positive CTs
        self.df = self.df[self.df.negative_exam_for_pe == 0]

        if self.split != "all":
            self.df = self.df[self.df["Split"] == self.split]

        if self.split == "train":
            if cfg.dataset.sample_frac < 1.0:
                study = list(self.df["patient_datetime"].unique())
                num_study = len(study)
                num_sample = int(num_study * cfg.dataset.sample_frac)
                sampled_study = np.random.choice(study, num_sample, replace=False)
                self.df = self.df[self.df["patient_datetime"].isin(sampled_study)]

        print(self.df[self.label_name].value_counts())
        self.labels = self.df[self.label_name].to_list()

    def __getitem__(self, index):
        # get slice row
        instance_info = self.df.iloc[index]

        # get slice info
        study_id = instance_info[STUDY_COL]
        series_id = instance_info[SERIES_COL]
        instance_id = instance_info[INSTANCE_COL]

        slice_path = (
            Path(self.cfg.dataset.dicom_dir)
            / study_id
            / series_id
            / f"{instance_id}.dcm"
        )
        ct_slice = self.process_slice(slice_path=slice_path)

        if ct_slice.sum() == 0:
            print(study_id)
            instance_id = "skip"

        if len(ct_slice.shape) == 4:
            print(study_id, series_id, instance_id)
            ct_slice = np.squeeze(ct_slice[0])

        # transform
        if ct_slice.shape[0] == 3:
            try:
                ct_slice = np.transpose(ct_slice, (1, 2, 0))
                ct_slice = Image.fromarray(np.uint8(ct_slice * 255))
            except:
                print(ct_slice.shape)

        else:
            ct_slice = Image.fromarray(np.uint8(ct_slice * 255)).convert("RGB")

        x = self.transform(ct_slice)

        # check dimention
        if x.shape[0] == 1:  # for repeat
            c, w, h = list(x.shape)
            x = x.expand(3, w, h)
        x = x.type(torch.FloatTensor)

        # get labels
        y = torch.tensor([instance_info[self.label_name]]).float()

        return x, y, 0, instance_id

    def __len__(self):
        return len(self.df)

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
