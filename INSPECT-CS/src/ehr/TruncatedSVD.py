import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

tasks = ['1_month_mortality', '6_month_mortality', '12_month_mortality']
for task in tasks:
    # Load the feature matrix and labels
    feature_matrix_path = f"data/ehr/output/labels_and_features/{task}/filtered_featurized_patients.pkl"  # Update with actual path
    cohort = pd.read_csv(f"data/ehr/output/labels_and_features/{task}/filtered_cohort.csv")

    # Load data
    filtered_cohort = cohort #[cohort['split'] == 'train']

    # Get impression IDs, patient IDs and label times
    impression_ids = filtered_cohort["ImpressionID"].tolist()
    pids = filtered_cohort["PatientID"].tolist()
    label_times = filtered_cohort["StudyTime"].tolist()
    with open(feature_matrix_path, "rb") as f:
        feature_matrix, patient_ids, label_values, label_times = pickle.load(f)

    # Filter the feature matrix to include only rows related to pids and label_times in filtered_cohort
    filtered_indices = [i for i, (pid, ltime) in enumerate(zip(patient_ids, label_times)) if pid in pids and ltime in label_times]
    feature_matrix = feature_matrix[filtered_indices]
    patient_ids = [patient_ids[i] for i in filtered_indices]
    label_values = [label_values[i] for i in filtered_indices]
    label_times = [label_times[i] for i in filtered_indices]

    print("Filtered feature matrix shape:", feature_matrix.shape)

    svd = TruncatedSVD(n_components=768, random_state=42)
    X_dense = svd.fit_transform(feature_matrix)  # shape: (n, 512)

    explained = np.cumsum(svd.explained_variance_ratio_)

    plt.plot(explained)
    plt.axhline(y=0.90, color='r', linestyle='--', label='90% varianza')
    plt.axhline(y=0.95, color='g', linestyle='--', label='95% varianza')
    plt.xlabel("Numero di componenti")
    plt.ylabel("Varianza spiegata cumulativa")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("Reduced feature matrix shape:", X_dense.shape)

    # Save data
    output_path = f"data/ehr/output/labels_and_features/{task}/truncatedSVD_features.pkl"
    with open(output_path, "wb") as f:
        pickle.dump((X_dense, patient_ids, label_values, label_times), f)

    print(f"Reduced feature matrix saved to {output_path}")