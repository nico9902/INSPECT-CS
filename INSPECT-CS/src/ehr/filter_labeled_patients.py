import pickle
import numpy as np
import pandas as pd
import os

# --- Configuration and Paths ---
base_path = 'data/ehr'
cohort_path = os.path.join(base_path, 'cohort_0.2.0_master_file_anon.csv')
tasks = ['1_month_mortality', '6_month_mortality', '12_month_mortality']

# Load the master cohort file containing 'Censored' status information
cohort_df = pd.read_csv(cohort_path)

for task in tasks:
    print(f"\n--- Processing Task: {task} ---")
    
    # Define directories and file paths for the specific task
    task_dir = os.path.join(base_path, f'output/labels_and_features/{task}')
    input_pkl = os.path.join(task_dir, 'featurized_patients.pkl')
    output_pkl = os.path.join(task_dir, 'filtered_featurized_patients.pkl')
    labeled_patients_csv = os.path.join(task_dir, 'labeled_patients.csv')
    output_csv = os.path.join(task_dir, 'filtered_cohort.csv')

    # 1. Load the original featurized data (Pickle)
    with open(input_pkl, 'rb') as f:
        sparse_matrix, ids, labels, timestamps = pickle.load(f)
    
    print(f"Initial: {len(ids)} patients, Matrix shape: {sparse_matrix.shape}")

    # 2. Filter out 'Censored' patients based on the master cohort file
    # This step ensures we only keep patients with a definitive outcome for the specific task
    censored_ids = cohort_df[cohort_df[task] == 'Censored']['PersonID'].tolist()
    mask_censored = ~np.isin(ids, censored_ids)

    # Apply the censorship mask to all data structures
    sparse_matrix = sparse_matrix[mask_censored]
    ids = np.array(ids)[mask_censored]
    labels = np.array(labels)[mask_censored]
    timestamps = np.array(timestamps)[mask_censored]

    # 3. Align with the labeled_patients CSV file
    # Ensure the CSV version matches the patients remaining after the censorship filter
    labeled_df = pd.read_csv(labeled_patients_csv)
    filtered_labeled_df = labeled_df[labeled_df['patient_id'].isin(ids)]
    
    # 4. Deduplication
    # Remove duplicates based on the combination of patient_id and timestamp
    temp_df = pd.DataFrame({'patient_id': ids, 'timestamp': timestamps})
    unique_mask = ~temp_df.duplicated(subset=['patient_id', 'timestamp'])
    
    # Apply the final deduplication mask
    sparse_matrix = sparse_matrix[unique_mask.values]
    ids = ids[unique_mask.values]
    labels = labels[unique_mask.values]
    timestamps = timestamps[unique_mask.values]

    # Synchronize the final CSV to match the deduplicated Pickle data
    final_csv_df = filtered_labeled_df[filtered_labeled_df['patient_id'].isin(ids)]

    # 5. Save the filtered and deduplicated results
    # Save the updated Tuple to a new Pickle file
    with open(output_pkl, 'wb') as f_out:
        pickle.dump((sparse_matrix, ids, labels, timestamps), f_out)
    
    # Save the synchronized CSV file
    final_csv_df.to_csv(output_csv, index=False)

    # Final Statistics
    print(f"Final: {len(ids)} patients, Matrix shape: {sparse_matrix.shape}")
    print(f"Files saved: \n - {output_pkl}\n - {output_csv}")

print("\nProcessing completed for all tasks.")