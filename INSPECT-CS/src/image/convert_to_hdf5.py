import os
import h5py
import numpy as np
from pathlib import Path
from collections import defaultdict

def convert_npy_to_hdf5(input_dir, output_path):
    # Create HDF5 file
    with h5py.File(output_path, 'w') as hdf5_file:
        # Get all NPY files
        npy_files = list(Path(input_dir).glob('*.npy'))

        # Group slices by impression_id
        impression_slices = defaultdict(list)

        for npy_file in npy_files:
            try:
                # Extract impression_id and slice_idx from filename
                filename = npy_file.stem
                impression_id, slice_idx = filename.split('_')
                slice_idx = int(slice_idx)

                # Load slice features
                slice_features = np.load(npy_file)

                # Ensure slice features are 2D
                if len(slice_features.shape) == 1:
                    slice_features = slice_features.reshape(1, -1)

                # Append slice to the corresponding impression_id
                impression_slices[impression_id].append((slice_idx, slice_features))
            except Exception as e:
                print(f"Error processing {npy_file}: {str(e)}")
                continue

        # Save each impression as a dataset in the HDF5 file
        for impression_id, slices in impression_slices.items():
            # Sort slices by slice_idx
            slices = sorted(slices, key=lambda x: x[0])
            stacked_slices = np.stack([slice_features for _, slice_features in slices], axis=0)

            # Save to HDF5
            hdf5_file.create_dataset(impression_id, data=stacked_slices, dtype='float32')
            print(f"Added impression {impression_id} to HDF5 file with shape {stacked_slices.shape}")

if __name__ == "__main__":
    input_dir = "/mimer/NOBACKUP/groups/naiss2023-6-336/multimodal_os/PE-Insight/datasets/inspect2/anon_pe_features_full_new"
    output_path = "/mimer/NOBACKUP/groups/naiss2023-6-336/multimodal_os/PE-Insight/datasets/inspect2/features.hdf5"

    convert_npy_to_hdf5(input_dir, output_path)
    print(f"HDF5 file created at: {output_path}")