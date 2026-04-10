import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import os
import pickle
import pandas as pd
from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef

from multi import networks
# from multi.utils_general import build_loss

class PEModel(pl.LightningModule):
    def __init__(self, cfg, device, exp_name=None):
        super().__init__()
        self.cfg = cfg
        self.device_type = device.type
        self.model = networks.init_model(cfg)
        self.criterion = nn.BCEWithLogitsLoss() #build_loss(cfg)
        self.exp_name = exp_name
        self.modalities = cfg.modalities
        self.save_dir = os.path.join(cfg.base_dir, exp_name)
        self.alpha = cfg.trainer.alpha if hasattr(cfg.trainer, 'alpha') else 0.5

        # Store outputs and labels for AUC computation
        self.step_outputs = {"train": {"outputs": [], "labels": [], "ids": [], "impressions": []},
                             "val": {"outputs": [], "labels": [], "ids": [], "impressions": []},
                             "test": {"outputs": [], "labels": [], "ids": [], "impressions": []}}

        # Initialize lists to track losses and metrics
        self.train_losses = []
        self.valid_losses = []
        self.train_metrics = []
        self.valid_metrics = []
        self.epochs = 0
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.trainer.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.cfg.trainer.lr_step_size, gamma=self.cfg.trainer.lr_gamma)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def on_train_epoch_end(self):
        return self.shared_epoch_end("train")

    def on_validation_epoch_end(self):
        return self.shared_epoch_end("val")
    
    def on_test_epoch_end(self):
        return self.shared_epoch_end("test")
    
    def on_train_end(self):
        self.plot_metrics()

    def shared_step(self, batch, split):
        if split == "train":
            if self.cfg.model.fusion.add_contrast and self.cfg.model.name == "armour":
                if hasattr(self.model.fusion, 'train_stage'): self.model.fusion.train_stage=True
            
        if split == "val" and self.cfg.model.name == "armour":
            if self.cfg.model.fusion.add_contrast:
                if hasattr(self.model.fusion, 'train_stage'): self.model.fusion.train_stage=False

        # Forward pass
        *xs, labels, mask, pids, impressions = batch   
        if self.cfg.model.name == "early":
            outputs, contrastive_loss = self.model(*xs)  
        else:
            if 'ehr' in self.modalities:
                ehr_index = self.modalities.index('ehr')
                xs[ehr_index] = xs[ehr_index].float().unsqueeze(1)
            outputs, contrastive_loss = self.model(*xs, *mask)  

        loss = self.alpha * contrastive_loss + (1 - self.alpha) * self.criterion(outputs, labels)

        # Log loss
        if split == "train" or split == "val":
            self.log(f"{split}_loss", loss, on_step=True, on_epoch=True)
        

        # Collect outputs and labels for AUC computation
        self.step_outputs[split]["outputs"].append(outputs)
        self.step_outputs[split]["labels"].append(labels)
        self.step_outputs[split]["ids"].append(pids)
        self.step_outputs[split]["impressions"].append(impressions)

        return loss
    
    def shared_epoch_end(self, split):  
        all_outputs = torch.cat([f for x in self.step_outputs[split]["outputs"] for f in x])
        all_labels = torch.cat([f for x in self.step_outputs[split]["labels"] for f in x])
        all_ids = [f for x in self.step_outputs[split]["ids"] for f in x]
        all_impressions = [f for x in self.step_outputs[split]["impressions"] for f in x]
        prob = torch.sigmoid(all_outputs)
        
        if type(all_labels) == torch.Tensor:
            all_labels = all_labels.detach().cpu().numpy()
        if type(all_outputs) == torch.Tensor:
            prob_ = torch.sigmoid(all_outputs)
            prob_ = prob_.detach().cpu().numpy()
        
        # Compute AUC
        auc = roc_auc_score(all_labels, prob_)
        self.log(f"{split}_auc", auc, on_epoch=True, logger=True, prog_bar=True)
        
        # Compute AUPRC
        auprc = average_precision_score(all_labels, prob_)
        self.log(f"{split}_auprc", auprc, on_epoch=True, logger=True, prog_bar=True)
        
        # Compute MCC
        binary_preds = (prob_ >= 0.5).astype(int)
        mcc = matthews_corrcoef(all_labels, binary_preds)
        self.log(f"{split}_mcc", mcc, on_epoch=True, logger=True, prog_bar=True)

        if split == "test":
            config_out_dir = os.path.join(self.save_dir, "config.pkl")
            pickle.dump(self.cfg, open(config_out_dir, "wb"))

            out_dir = os.path.join(self.save_dir, "test_preds.csv")
            all_p = prob.cpu().detach().tolist()
            all_label = all_labels.tolist()
            all_ids = [f for x in self.step_outputs[split]["ids"] for f in x]
            outfile = defaultdict(list)
            for ids, impressions, label, p in zip(all_ids, all_impressions, all_label, all_p):
                pid = ids
                impression = impressions
                outfile["patient_id"].append(pid)
                outfile["impression_id"].append(impression)
                outfile["label"].append(label)
                outfile["prob"].append(p)

            df = pd.DataFrame.from_dict(outfile)
            df.to_csv(out_dir, index=False)
            print("=" * 80)
            print(f"Config saved at: {out_dir})")
            print(f"Predictions saved at: {out_dir})")
            print("AUC:", auc)
            print("AUPRC:", auprc)
            print("MCC:", mcc)
            print("=" * 80)

        # Track losses and metrics
        if self.trainer.global_rank == 0:
            if split == "train":
                self.train_losses.append(self.trainer.callback_metrics[f"{split}_loss"].item())
                self.train_metrics.append(auc * 100)
            elif split == "val":
                self.valid_losses.append(self.trainer.callback_metrics[f"{split}_loss"].item())
                self.valid_metrics.append(auc * 100)

        # Clear stored outputs and labels
        self.step_outputs[split]["outputs"].clear()
        self.step_outputs[split]["labels"].clear()

        # Increment epoch count and plot if validation epoch ends
        if split == "train" and self.trainer.global_rank == 0:
            self.epochs += 1

    def plot_metrics(self):
        print("Plotting metrics...")
        ticks = list(range(1, self.epochs + 1))

        # Create report directory if it doesn't exist
        os.makedirs(os.path.join(self.save_dir, self.exp_name), exist_ok=True)

        # Plot Loss
        fig1, ax1 = plt.subplots(figsize=(12, 8), num=1)
        ax1.set_xticks(np.arange(0, self.epochs + 1, step=max(1, self.epochs // 10)))
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss Function', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_yscale('log')
        ax1.plot(ticks, self.train_losses, 'b-', linewidth=1.0, label='Training (best %.2f at ep. %d)' % ( min(self.train_losses), ticks[np.argmin(self.train_losses)]))
        ax1.plot(ticks, self.valid_losses, 'b--', linewidth=1.0, label='Validation (best %.2f at ep. %d)' % ( min(self.valid_losses), ticks[np.argmin(self.valid_losses)]))
        ax1.legend(loc="lower left")
        plt.savefig(os.path.join(self.save_dir, f"loss.png"))
        plt.close(fig1)

        # Plot Metric
        fig2, ax2 = plt.subplots(figsize=(12, 8), num=1)
        ax2.set_xticks(np.arange(0, self.epochs + 1, step=max(1, self.epochs // 10)))
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Metric %', color='red')
        ax2.set_ylim(0, 100)
        ax2.set_yticks(np.arange(0, 101, step=10))
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.plot(ticks, self.train_metrics, 'r-', linewidth=1.0, label='Training (best %.2f at ep. %d)' % ( max(self.train_metrics), ticks[np.argmax(self.train_metrics)]))
        ax2.plot(ticks, self.valid_metrics, 'r--', linewidth=1.0, label='Validation (best %.2f at ep. %d)' % ( max(self.valid_metrics), ticks[np.argmax(self.valid_metrics)]))
        ax2.legend(loc="lower left")
        plt.xlim(0, self.epochs + 1)
        plt.savefig(os.path.join(self.save_dir, f"metric.png"))
        plt.close(fig2)