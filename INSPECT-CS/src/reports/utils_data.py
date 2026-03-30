def read_features_from_hdf5(hdf5_dataset, impression_id):
    """
    Reads features for a given impression_id from an HDF5 file.
    """
    impression_id = str(impression_id)
    if impression_id in hdf5_dataset:
        features = hdf5_dataset[impression_id][:]
    else:
        raise KeyError(f"Impression ID {impression_id} not found in the HDF5 file.")
    return features