import os
import pickle
import nibabel as nib

def build_dict_slice_thickness(nii_dir):
    dict_slice_thickness = {}

    for fname in os.listdir(nii_dir):
        if fname.endswith(".nii.gz"):
            path = os.path.join(nii_dir, fname)
            img = nib.load(path)
            header = img.header
            voxel_dims = header.get_zooms()  # (x, y, z)

            # Assuming z-dimension corresponds to slice thickness
            slice_thickness = voxel_dims[2]

            # Get impression_id from filename (you may need to customize this)
            impression_id = os.path.splitext(os.path.splitext(fname)[0])[0]
            dict_slice_thickness[impression_id] = slice_thickness

    return dict_slice_thickness

thickness_dict = build_dict_slice_thickness("/mimer/NOBACKUP/groups/naiss2023-6-336/dataset_shared/inspect2/CTPA/")
with open("/mimer/NOBACKUP/groups/naiss2023-6-336/dataset_shared/inspect2/dict_slice_thickness.pkl", "wb") as f:
    pickle.dump(thickness_dict, f)