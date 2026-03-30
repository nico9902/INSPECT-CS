import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import pickle

# Load the feature matrix and labels
feature_matrix_path = "/Users/domenicopaolo/Documents/PhD AI/Projects/PE-Insight/data/ehr/output/labels_and_features/6_month_mortality/featurized_patients.pkl"  # Update with actual path

# Load data
with open(feature_matrix_path, "rb") as f:
    feature_matrix, patient_ids, label_values, label_times = pickle.load(f)

print("Feature matrix shape:", feature_matrix.shape)
print("Number of patient IDs:", len(patient_ids))
print("Number of label values:", len(label_values)) 
print("Number of label times:", len(label_times))

count = 0
for features, patient_id, label_value, label_time in zip(feature_matrix, patient_ids, label_values, label_times):
    if count < 10:
        print(f"Patient ID: {patient_id}, Label Value: {label_value}, Label Time: {label_time}")
        print("Features:", features.shape)
    count += 1

# # Find the index of the patient with ID 115968504
# patient_id_to_find = 127926385
# if patient_id_to_find in patient_ids:
#     target_indices = [i for i, pid in enumerate(patient_ids) if pid == patient_id_to_find]
#     for idx in target_indices:
#         print(f"Patient ID: {patient_id_to_find}, Label: {label_values[idx]}, Label Time: {label_times[idx]}")
# else:
#     print(f"Patient ID {patient_id_to_find} not found.")

# import os
# path_to_cohort = '/Users/domenicopaolo/Documents/PhD AI/Projects/PE-Insight/outputs'
# tasks = ['1_month_mortality', '6_month_mortality', '12_month_mortality']

# for task in tasks:
#     ehr_csv = os.path.join(path_to_cohort, f'ehr_{task}', 'test_preds.csv')
#     impression_csv = os.path.join(path_to_cohort, f'image_{task}_0', 'test_preds.csv')    

#     # Read the CSV files
#     ehr_data = pd.read_csv(ehr_csv)['label'].to_list()
#     impression_data = pd.read_csv(impression_csv)['label'].to_list()

#     # Find indices where the labels differ in the two lists
#     differing_indices = [i for i in range(len(ehr_data)) if ehr_data[i] != impression_data[i]]
#     print(f"Task: {task}, Number of differing labels: {len(differing_indices)}")
#     print("indices:", differing_indices)