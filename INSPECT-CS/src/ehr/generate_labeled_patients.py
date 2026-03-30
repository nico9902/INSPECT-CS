import os
import pandas as pd

# Define the output directory and report log path
output_dir = 'data/ehr/output/labels_and_features'
cohort_csv = 'data/cohort_0.2.0_master_file_anon.csv'

cohort_columns = ['PatientID', 'StudyTime', 'split']

# Define the tasks to process
tasks = ['1_month_mortality', '6_month_mortality', '12_month_mortality']

# Modify the loop to map patient_id and process predictions
for task in tasks:
    print(f"Processing task: {task}")
    
    labeled_csv_path = os.path.join(output_dir, f'{task}', 'labeled_patients.csv')
    if not os.path.exists(labeled_csv_path):
        print(f"Missing files in task {task}. Skipping...")
        continue

    labeled_patients = pd.read_csv(labeled_csv_path)

    # Load the labels file
    cohort = pd.read_csv(cohort_csv)

    # Select the required columns from the cohort
    cohort_selected = cohort[cohort_columns]
    cohort_selected.rename(columns={'StudyTime': 'prediction_time'}, inplace=True)
    cohort_selected['label_type'] = 'boolean'
    # Map the label values and discard rows with 'Censored' values
    cohort_selected['value'] = cohort['PatientID'].map(
        cohort.set_index('PatientID')[task]
    )
    cohort_selected.rename(columns={'PatientID': 'patient_id'}, inplace=True)
    cohort_selected = cohort_selected[cohort_selected['value'] != 'Censored']

    # Save the labeled patients to a new CSV file
    cohort_selected.to_csv(os.path.join(output_dir, f'{task}', 'labeled_patients.csv'), index=False)
