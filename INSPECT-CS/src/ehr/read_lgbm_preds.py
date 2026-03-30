import pickle
import os
import numpy as np
import collections
import csv
import logging
from datetime import datetime

path_to_cohort = 'data/ehr/output/labels_and_features'
root = 'data/ehr/output/gbm_models'
tasks = ['1_month_mortality', '6_month_mortality', '12_month_mortality']
seeds = [1, 2, 3, 4]
for task in tasks:
    for seed in seeds:
        path_to_output_dir = f'{root}/{task}_{seed}'
        path_to_predictions = f'{path_to_output_dir}/predictions.pkl'

        with open(path_to_predictions, 'rb') as f:
            proba, patient_ids, label_values, label_times = pickle.load(f)

        print(f"Task: {task}")
        print(f"Number of patients: {len(patient_ids)}")
        print(f"Number of predictions: {len(proba)}")


        pid_split_assignment = collections.defaultdict(set)
        PATIENT_ID_COLUMN = 'patient_id'
        with open(os.path.join(path_to_cohort, task, 'filtered_cohort.csv')) as f:
            reader = csv.DictReader(f)
            cohort_rows = list(reader)
            for row in cohort_rows:
                pid_split_assignment[row["split"]].add(int(row[PATIENT_ID_COLUMN]))

        total_size = 0
        total_set = set()
        for split in pid_split_assignment.values():
            total_size += len(split)
            total_set |= split

        assert total_size == len(total_set)

        train_mask = np.isin(patient_ids, list(pid_split_assignment["train"]))
        valid_mask = np.isin(patient_ids, list(pid_split_assignment["valid"]))
        test_mask = np.isin(patient_ids, list(pid_split_assignment["test"]))

        train_proba = proba[train_mask]
        valid_proba = proba[valid_mask]
        test_proba = proba[test_mask]

        train_patient_ids = np.array(patient_ids)[train_mask]
        valid_patient_ids = np.array(patient_ids)[valid_mask]
        test_patient_ids = np.array(patient_ids)[test_mask]
        test_label_times = np.array(label_times)[test_mask]
        test_labels = np.array(label_values)[test_mask]

        # Find the label and label time for a specific patient ID
        target_id = 124953145
        if target_id in test_patient_ids:
            target_indices = [i for i, pid in enumerate(test_patient_ids) if pid == target_id]
            for idx in target_indices:
                print(f"Patient ID: {target_id}, Label: {test_labels[idx]}, Label Time: {test_label_times[idx]}")
        else:
            print(f"Patient ID {target_id} not found in the test cohort.")

        # Sort test_patient_ids and test_proba according to the order in filtered_cohort.csv
        test_cohort_patient_ids = [row['patient_id'] for row in cohort_rows if row["split"] == "test"]
        
        # Get the corresponding indices, probabilities, and label times for test cohort
        sorted_indices = []
        tmp_ids = []
        for pid in test_cohort_patient_ids:
            if int(pid) not in tmp_ids:
                indices = [i for i, p in enumerate(test_patient_ids) if p == int(pid)]
                sorted_indices.extend(indices)
                tmp_ids.append(int(pid))
        # print(len(sorted_indices)) #2935
        
        sorted_test_patient_ids = np.array(test_cohort_patient_ids, dtype=int)
        sorted_test_patient_ids = test_patient_ids[sorted_indices]
        sorted_test_proba = test_proba[sorted_indices]
        sorted_test_labels = test_labels[sorted_indices]
        sorted_test_label_times = test_label_times[sorted_indices]

        # Find the label and label time for a specific patient ID
        target_id = 124953145
        if target_id in sorted_test_patient_ids:
            target_indices = [i for i, pid in enumerate(sorted_test_patient_ids) if pid == target_id]
            for idx in target_indices:
                print(f"Patient ID: {target_id}, Label: {sorted_test_labels[idx]}, Label Time: {sorted_test_label_times[idx]}")
        else:
            print(f"Patient ID {target_id} not found in the test cohort.")

        # Find patient IDs with multiple instances
        from collections import Counter
        pid_counts = Counter(sorted_test_patient_ids)
        multiple_instance_pids = {pid for pid, count in pid_counts.items() if count > 1}
        
        if multiple_instance_pids:
            print(f"Found {len(multiple_instance_pids)} patient IDs with multiple instances")
            
            # Create lists to store final sorted results
            final_patient_ids = []
            final_proba = []
            final_label_times = []
            final_labels = []
            
            # Track which indices we've already processed
            processed_indices = set()
            
            for i, pid in enumerate(sorted_test_patient_ids):
                if i in processed_indices:
                    continue
                    
                if pid in multiple_instance_pids:
                    # Find all instances of this patient ID
                    pid_indices = [j for j, p in enumerate(sorted_test_patient_ids) if p == pid]
                    
                    # Get the label times for these instances
                    pid_label_times = sorted_test_label_times[pid_indices]
                    pid_probas = sorted_test_proba[pid_indices]
                    pid_labels = sorted_test_labels[pid_indices]
                    
                    # Sort by label time
                    time_sort_indices = np.argsort(pid_label_times)
                    
                    # Add sorted instances to final lists
                    for idx in time_sort_indices:
                        actual_idx = pid_indices[idx]
                        final_patient_ids.append(sorted_test_patient_ids[actual_idx])
                        final_proba.append(sorted_test_proba[actual_idx])
                        final_label_times.append(sorted_test_label_times[actual_idx])
                        final_labels.append(sorted_test_labels[actual_idx])
                    
                    # Mark all these indices as processed
                    processed_indices.update(pid_indices)
                else:
                    # Single instance - keep in original order
                    final_patient_ids.append(sorted_test_patient_ids[i])
                    final_proba.append(sorted_test_proba[i])
                    final_label_times.append(sorted_test_label_times[i])
                    final_labels.append(sorted_test_labels[i])
                    processed_indices.add(i)
            
            # Convert back to numpy arrays
            sorted_test_patient_ids = np.array(final_patient_ids)
            sorted_test_proba = np.array(final_proba)
            sorted_test_label_times = np.array(final_label_times)
            sorted_test_labels = np.array(final_labels)
            
            print(f"Sorted {len([pid for pid in multiple_instance_pids])} unique patient IDs with multiple instances by label time")
        
        print(f"Test cohort patient IDs: {sorted_test_patient_ids[:5]}")
        print(f"Sorted test probabilities: {sorted_test_proba[:5]}")
        print(f"Sorted test label times: {sorted_test_label_times[:5]}")
        print(f"Sorted test labels: {sorted_test_labels[:5]}")

        # Save the sorted predictions in a csv with columns patient_id, label, prob
        dir = f'/Users/domenicopaolo/Documents/PhD AI/Projects/PE-Insight/outputs/ehr_{task}_{seed}'
        if not os.path.exists(dir):
            os.makedirs(dir)
        output_file = os.path.join(f'/Users/domenicopaolo/Documents/PhD AI/Projects/PE-Insight/outputs/ehr_{task}_{seed}', "test_preds.csv")
        with open(output_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['patient_id', 'label', 'prob'])
            for pid, label, prob in zip(sorted_test_patient_ids, sorted_test_labels, sorted_test_proba):
                writer.writerow([pid, float(label), prob])