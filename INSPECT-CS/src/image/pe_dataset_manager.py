import pickle
import pandas as pd

def rename_pkl_datasets(pkl_file, mapping_file):
    """
    Renames keys in a pickle file based on a mapping provided in a TSV file.

    Parameters:
        pkl_file (str): Path to the pickle file.
        mapping_file (str): Path to the TSV file containing old and new names.
    """
    # Load the mapping from the TSV file
    mapping = pd.read_csv(mapping_file, sep='\t', header=None)

    # Load the pickle file
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    # Rename keys based on the mapping
    for _, row in mapping.iterrows():
        old_name = row[6]
        new_name = row[0]

        if old_name in data:  # Check if old_name exists in the pickle data
            data[new_name] = data.pop(old_name)
            print(f"Renamed '{old_name}' to '{new_name}'")
        else:
            print(f"Key '{old_name}' not found in the pickle file.")

    # Save the updated pickle file
    with open(pkl_file, 'wb') as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    # Example usage
    pkl_file = "/mimer/NOBACKUP/groups/naiss2023-6-336/dataset_shared/inspect2/dict_slice_thickness.pkl"  # Replace with the path to your pickle file
    mapping_file = "/mimer/NOBACKUP/groups/naiss2023-6-336/multimodal_os/PE-Insight/data/study_mapping_20250611.tsv"  # Replace with the path to your TSV file
    rename_pkl_datasets(pkl_file, mapping_file)