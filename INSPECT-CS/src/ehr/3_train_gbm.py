#python3 src/ehr/3_train_gbm.py --path_to_cohort "/Users/domenicopaolo/Documents/PhD AI/Projects/PE-Insight/data/ehr/output/labels_and_features/1_month_mortality/filtered_cohort.csv" --path_to_label_features "/Users/domenicopaolo/Documents/PhD AI/Projects/PE-Insight/data/ehr/output/labels_and_features/1_month_mortality" --path_to_output_dir "/Users/domenicopaolo/Documents/PhD AI/Projects/PE-Insight/data/ehr/output/gbm_models/1_month_mortality_1" --target "1_month_mortality" --seed 0
import argparse
import os
import pickle
from typing import List, Tuple, Optional
from collections import Counter
import collections

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    LogisticRegressionCV,
)
from sklearn.preprocessing import MaxAbsScaler
from loguru import logger
# from utils import load_data, save_data
import matplotlib.pyplot as plt
import collections
from sklearn.model_selection import GridSearchCV, PredefinedSplit, ParameterGrid
from scipy.sparse import issparse
import scipy
import csv
import lightgbm as ltb

# import femr
# import femr.datasets


XGB_PARAMS = {
    "max_depth": [3, 6, -1],
    "learning_rate": [0.02, 0.1, 0.5],
    "num_leaves": [10, 25, 100],
}

PATIENT_ID_COLUMN = "PatientID"  # "patient_id"
TIME_COLUMN = "StudyTime"  # "procedure_time"

def tune_hyperparams(
    X_train, y_train, X_val, y_val, model, params, num_threads: int = 1
):
    # In `test_fold`, -1 indicates that the corresponding sample is used for training, and a value >=0 indicates the test set.
    # We use `PredefinedSplit` to specify our custom validation split
    if issparse(X_train):
        # Need to concatenate sparse matrices differently
        X = scipy.sparse.vstack([X_train, X_val])
    else:
        X = np.concatenate((X_train, X_val), axis=0)
    y = np.concatenate((y_train, y_val), axis=0)
    test_fold = -np.ones(X.shape[0])
    test_fold[X_train.shape[0] :] = 1
    clf = GridSearchCV(
        model,
        params,
        n_jobs=6,
        verbose=1,
        cv=PredefinedSplit(test_fold=test_fold),
        refit=False,
    )
    clf.fit(X, y)
    best_model = model.__class__(**clf.best_params_)
    best_model.fit(X_train, y_train)
    return best_model


def main(args):
    # Imposta TUTTI i seed possibili
    import random
    import os
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    pid_split_assignment = collections.defaultdict(set)
    pid_label_assignment = collections.defaultdict(set)
    with open(args.path_to_cohort) as f:
        if args.path_to_cohort.endswith(".csv"):
            reader = csv.DictReader(f)
        elif args.path_to_cohort.endswith(".tsv"):
            reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            # pid_split_assignment[row["split"]].add(int(row["patient_id"]))
            pid_split_assignment[row["split"]].add(int(row[PATIENT_ID_COLUMN]))
            row[TIME_COLUMN] = pd.to_datetime(row[TIME_COLUMN])
            pid_label_assignment[f'{int(row[PATIENT_ID_COLUMN])}_{row[TIME_COLUMN]}'].add(
                row[args.target]  # e.g., '1_month_mortality'
            )
        
        unique_values = set()
        for labels in pid_label_assignment.values():
            unique_values.update(labels)
        print(f"Unique values in pid_label_assignment: {unique_values}")

    total_size = 0
    total_set = set()
    for split in pid_split_assignment.values():
        total_size += len(split)
        total_set |= split

    assert total_size == len(total_set)

    with open(
        os.path.join(args.path_to_label_features, "filtered_featurized_patients.pkl"), "rb"
    ) as f:
        features, patient_ids, label_values, label_times = pickle.load(f)

    train_mask = np.isin(patient_ids, list(pid_split_assignment["train"]))
    valid_mask = np.isin(patient_ids, list(pid_split_assignment["valid"]))
    test_mask = np.isin(patient_ids, list(pid_split_assignment["test"]))

    logger.info(f"Task: {args.path_to_output_dir.split('/')[-1]}")
    logger.info(f"Num train: {sum(train_mask)}")
    logger.info(f"Num valid: {sum(valid_mask)}")
    logger.info(f"Num test: {sum(test_mask)}")

    X_train = features[train_mask, :]
    X_valid = features[valid_mask, :]
    X_test = features[test_mask, :]

    # Sort pid_label_assignment according to the patient IDs
    label_time_map = {f'{pid}_{pd.to_datetime(label_time)}': pid_label_assignment[f'{pid}_{pd.to_datetime(label_time)}']
                      for pid, label_time in zip(patient_ids, label_times)
                      if f'{pid}_{pd.to_datetime(label_time)}' in pid_label_assignment}
    sorted_labels = [next(iter(label_time_map[f'{pid}_{pd.to_datetime(label_time)}']))
                     for pid, label_time in zip(patient_ids, label_times)
                     if f'{pid}_{pd.to_datetime(label_time)}' in label_time_map]
    
    print(f"Total number of labels found: {len(sorted_labels)}")
    sorted_labels = np.array([
        float(label) if label.replace('.', '', 1).isdigit() else (1.0 if label.upper() == 'TRUE' else 0.0 if label.upper() == 'FALSE' else np.nan)
        for label in sorted_labels
    ])
    unique_sorted_labels = np.unique(sorted_labels)
    print(f"Unique values in sorted_labels: {unique_sorted_labels}")
    sorted_labels = sorted_labels[~np.isnan(sorted_labels)]  # Remove NaN values if any
    label_values = sorted_labels

    y_train = label_values[train_mask]
    y_valid = label_values[valid_mask]
    y_test = label_values[test_mask]

    os.makedirs(args.path_to_output_dir, exist_ok=True)

    if not args.test_GBM:
        model = tune_hyperparams(
            X_train,
            y_train,
            X_valid,
            y_valid,
            ltb.LGBMClassifier(random_state=args.seed, deterministic=True, force_row_wise=True, feature_fraction= 0.8, bagging_fraction= 0.8, bagging_freq= 1, data_random_seed=args.seed),
            XGB_PARAMS,
            num_threads=args.num_threads,
        )

        with open(os.path.join(args.path_to_output_dir, "model.pkl"), "wb") as f:
            pickle.dump(model, f)

        proba = model.predict_proba(features)[:, 1]

        with open(os.path.join(args.path_to_output_dir, "predictions.pkl"), "wb") as f:
            pickle.dump([proba, patient_ids, label_values, label_times], f)

        y_train_proba = proba[train_mask]
        y_valid_proba = proba[valid_mask]
        y_test_proba = proba[test_mask]

        train_auroc = metrics.roc_auc_score(y_train, y_train_proba)
        val_auroc = metrics.roc_auc_score(y_valid, y_valid_proba)
        test_auroc = metrics.roc_auc_score(y_test, y_test_proba)
        test_auprc = metrics.average_precision_score(y_test, y_test_proba)
        test_mcc = metrics.matthews_corrcoef(y_test, (y_test_proba >= 0.5).astype(int))
        logger.info(f"Train AUROC: {train_auroc}")
        logger.info(f"Val AUROC: {val_auroc}")
        logger.info(f"Test AUROC: {test_auroc}")
        logger.info(f"Test AUPRC: {test_auprc}")
        logger.info(f"Test MCC: {test_mcc}")
    else:
        with open(os.path.join(args.path_to_output_dir, "model.pkl"), "rb") as f:
            model = pickle.load(f)

        proba = model.predict_proba(features)[:, 1]
        y_test_proba = proba[test_mask]
        test_auroc = metrics.roc_auc_score(y_test, y_test_proba)
        test_auprc = metrics.average_precision_score(y_test, y_test_proba)
        test_mcc = metrics.matthews_corrcoef(y_test, (y_test_proba >= 0.5).astype(int))
        
        logger.info(f"Test AUROC: {test_auroc}")
        logger.info(f"Test AUPRC: {test_auprc}")
        logger.info(f"Test MCC: {test_mcc}")

    logger.success("DONE!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate CLMBR patient representations"
    )
    parser.add_argument(
        "--path_to_cohort", required=True, type=str, help="Path to femr database"
    )
    # parser.add_argument(
    #     "--path_to_database", required=True, type=str, help="Path to femr database"
    # )
    parser.add_argument(
        "--path_to_label_features",
        required=True,
        type=str,
        help="Path to save labeles and featurizers",
    )
    parser.add_argument(
        "--test_GBM",
        default=False,
        action="store_true",
        help="if you want to run GBM for test split only",
    )
    parser.add_argument(
        "--path_to_output_dir",
        required=True,
        type=str,
        help="Path to save labeles and featurizers",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        help="The number of threads to use",
        default=1,
    )
    parser.add_argument(
        "--target",
        type=str,
        default='1_month_mortality',
        help="Target task to train the model on (1_month_mortality, 6_month_mortality, 12_month_mortality)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    main(args)