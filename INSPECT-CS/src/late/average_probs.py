import pandas as pd
import os
from sklearn.metrics import roc_auc_score, matthews_corrcoef, average_precision_score
import numpy as np
import itertools

# Directory di output e tasks
output_dir = '/Users/domenicopaolo/Documents/PhD AI/Projects/PE-Insight/outputs'
tasks = ['1_month_mortality', '6_month_mortality', '12_month_mortality']
report_log_path = os.path.join('fusion_metrics_report.log')

modalities = ['reports', 'image', 'ehr', 'ehr_autoencoder']

with open(report_log_path, 'w') as report_log:
    for task in tasks:
        report_log.write(f"=== Processing task: {task} ===\n")
        print(f"\n=== Processing task: {task} ===")

        # Genera tutte le combinazioni non vuote di modalità
        for r in range(2, len(modalities) + 1):
            for combo in itertools.combinations(modalities, r):
                if 'ehr' in combo and 'ehr_autoencoder' in combo:
                    continue  # Skip combinations with both 'ehr' and 'ehr_autoencoder'
                combo_name = '+'.join(combo)
                report_log.write(f"\n--- Combination: {combo_name} ---\n")
                print(f"\n--- Combination: {combo_name} ---")

                aucs, mccs, auprcs = [], [], []
                survived_low_risk_list, died_high_risk_list = [], []

                for seed in range(5):
                    try:
                        # Percorsi CSV per ogni modalità
                        csv_paths = {
                            'reports': os.path.join(output_dir, f'reports_{task}_{seed}', 'test_preds.csv'),
                            'image': os.path.join(output_dir, f'image_{task}_{seed}', 'test_preds.csv'),
                            'ehr': os.path.join(output_dir, f'ehr_{task}_{seed}', 'test_preds.csv'),
                            'ehr_autoencoder': os.path.join(output_dir, f'sae_ehr_{task}_{seed}', 'test_preds.csv')
                        }
                        
                        # Carica solo le modalità presenti nella combinazione
                        dfs = []
                        for m in combo:
                            if not os.path.exists(csv_paths[m]):
                                raise FileNotFoundError(f"{m} missing for seed {seed}: {csv_paths[m]}")
                            dfs.append(pd.read_csv(csv_paths[m]))

                        # Estrai etichette (tutte devono avere stessi pazienti)
                        y_true = dfs[0]['label']
                        probs = [df['prob'] for df in dfs]

                        # Media delle probabilità
                        avg_probs = np.mean(probs, axis=0)
                        y_pred = (avg_probs >= 0.5).astype(int)

                        # Calcola metriche
                        auc = roc_auc_score(y_true, avg_probs)
                        mcc = matthews_corrcoef(y_true, y_pred)
                        auprc = average_precision_score(y_true, avg_probs)

                        aucs.append(auc)
                        mccs.append(mcc)
                        auprcs.append(auprc)

                        # Calcola percentuali richieste
                        # Pazienti sopravvissuti (label 0) con rischio <= 50%
                        survivors_mask = (y_true == 0)
                        survivors_low_risk_mask = survivors_mask & (avg_probs <= 0.5)
                        pct_survived_low_risk = (survivors_low_risk_mask.sum() / survivors_mask.sum()) * 100 if survivors_mask.sum() > 0 else 0
                        survived_low_risk_list.append(pct_survived_low_risk)

                        # Pazienti deceduti (label 1) con rischio >= 50%
                        deceased_mask = (y_true == 1)
                        deceased_high_risk_mask = deceased_mask & (avg_probs >= 0.5)
                        pct_died_high_risk = (deceased_high_risk_mask.sum() / deceased_mask.sum()) * 100 if deceased_mask.sum() > 0 else 0
                        died_high_risk_list.append(pct_died_high_risk)

                    except Exception as e:
                        report_log.write(f"Skipping seed {seed}: {e}\n")
                        print(f"Skipping seed {seed}: {e}")
                        continue

                # Calcola media e std
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