import pandas as pd
import os
from sklearn.metrics import roc_auc_score, matthews_corrcoef, average_precision_score
import numpy as np
import itertools
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Average unimodal prediction probabilities for late fusion.")
    parser.add_argument("--output_dir", default="outputs", help="Directory containing unimodal prediction folders.")
    parser.add_argument("--report_log_path", default="fusion_metrics_report.log", help="Path to write the metrics report.")
    parser.add_argument("--tasks", nargs="+", default=["1_month_mortality", "6_month_mortality", "12_month_mortality"], help="Prediction tasks to evaluate.")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds to scan, starting from zero.")
    return parser.parse_args()


def main():
    args = parse_args()
    modalities = ['reports', 'image', 'ehr', 'ehr_autoencoder']

    with open(args.report_log_path, 'w') as report_log:
        for task in args.tasks:
            report_log.write(f"=== Processing task: {task} ===\n")
            print(f"\n=== Processing task: {task} ===")

            # Evaluate every multimodal combination while avoiding duplicate EHR
            # representations in the same late-fusion ensemble.
            for r in range(2, len(modalities) + 1):
                for combo in itertools.combinations(modalities, r):
                    if 'ehr' in combo and 'ehr_autoencoder' in combo:
                        continue
                    combo_name = '+'.join(combo)
                    report_log.write(f"\n--- Combination: {combo_name} ---\n")
                    print(f"\n--- Combination: {combo_name} ---")

                    aucs, mccs, auprcs = [], [], []
                    survived_low_risk_list, died_high_risk_list = [], []

                    for seed in range(args.seeds):
                        try:
                            csv_paths = {
                                'reports': os.path.join(args.output_dir, f'reports_{task}_{seed}', 'test_preds.csv'),
                                'image': os.path.join(args.output_dir, f'image_{task}_{seed}', 'test_preds.csv'),
                                'ehr': os.path.join(args.output_dir, f'ehr_{task}_{seed}', 'test_preds.csv'),
                                'ehr_autoencoder': os.path.join(args.output_dir, f'sae_ehr_{task}_{seed}', 'test_preds.csv')
                            }

                            dfs = []
                            for m in combo:
                                if not os.path.exists(csv_paths[m]):
                                    raise FileNotFoundError(f"{m} missing for seed {seed}: {csv_paths[m]}")
                                dfs.append(pd.read_csv(csv_paths[m]))

                            y_true = dfs[0]['label']
                            probs = [df['prob'] for df in dfs]

                            avg_probs = np.mean(probs, axis=0)
                            y_pred = (avg_probs >= 0.5).astype(int)

                            aucs.append(roc_auc_score(y_true, avg_probs))
                            mccs.append(matthews_corrcoef(y_true, y_pred))
                            auprcs.append(average_precision_score(y_true, avg_probs))

                            survivors_mask = (y_true == 0)
                            survivors_low_risk_mask = survivors_mask & (avg_probs <= 0.5)
                            pct_survived_low_risk = (survivors_low_risk_mask.sum() / survivors_mask.sum()) * 100 if survivors_mask.sum() > 0 else 0
                            survived_low_risk_list.append(pct_survived_low_risk)

                            deceased_mask = (y_true == 1)
                            deceased_high_risk_mask = deceased_mask & (avg_probs >= 0.5)
                            pct_died_high_risk = (deceased_high_risk_mask.sum() / deceased_mask.sum()) * 100 if deceased_mask.sum() > 0 else 0
                            died_high_risk_list.append(pct_died_high_risk)

                        except Exception as e:
                            report_log.write(f"Skipping seed {seed}: {e}\n")
                            print(f"Skipping seed {seed}: {e}")
                            continue

                    if aucs:
                        mean_auc, std_auc = np.mean(aucs), np.std(aucs)
                        mean_mcc, std_mcc = np.mean(mccs), np.std(mccs)
                        mean_auprc, std_auprc = np.mean(auprcs), np.std(auprcs)
                        mean_surv_low, std_surv_low = np.mean(survived_low_risk_list), np.std(survived_low_risk_list)
                        mean_died_high, std_died_high = np.mean(died_high_risk_list), np.std(died_high_risk_list)

                        report_log.write(f"AUC: mean={mean_auc:.4f}, std={std_auc:.4f}\n")
                        report_log.write(f"MCC: mean={mean_mcc:.4f}, std={std_mcc:.4f}\n")
                        report_log.write(f"AUPRC: mean={mean_auprc:.4f}, std={std_auprc:.4f}\n")
                        report_log.write(f"Survived <= 50% Risk: mean={mean_surv_low:.2f}%, std={std_surv_low:.2f}%\n")
                        report_log.write(f"Died >= 50% Risk: mean={mean_died_high:.2f}%, std={std_died_high:.2f}%\n")

                        print(f"AUC: mean={mean_auc:.4f}, std={std_auc:.4f}")
                        print(f"MCC: mean={mean_mcc:.4f}, std={std_mcc:.4f}")
                        print(f"AUPRC: mean={mean_auprc:.4f}, std={std_auprc:.4f}")
                        print(f"Survived <= 50% Risk: mean={mean_surv_low:.2f}%, std={std_surv_low:.2f}%")
                        print(f"Died >= 50% Risk: mean={mean_died_high:.2f}%, std={std_died_high:.2f}%")
                    else:
                        report_log.write("No valid seeds for this combination.\n")
                        print("No valid seeds for this combination.")


if __name__ == "__main__":
    main()
