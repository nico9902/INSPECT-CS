# INSPECT-CS
This is a code implemention of the framework proposed in the paper "Multimodal Clinical Data Integration for Prognosis of Pulmonary Embolism: A Comparative Study".

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  
[![Paper](https://img.shields.io/badge/Paper-ResearchGate-green)](https://www.researchgate.net/publication/403120230_Multimodal_Clinical_Data_Integration_for_Prognosis_of_Pulmonary_Embolism_A_Comparative_Study)

## 📌 Overview
This repository contains the official implementation of the paper:

**"Multimodal Clinical Data Integration for Prognosis of Pulmonary Embolism: A Comparative Study"**  
Authors: Domenico Paolo, Paolo Soda,
Matteo Tortora, Alessandro Bria, Rosa Sicilia.

We combine structured EHR data, clinical notes, and imaging features to improve risk prediction performance.

---

## ⚙️ Installation
```bash
git clone https://github.com/nico9902/INSPECT-CS.git
cd INSPECT-CS
pip install -r requirements.txt
```
---

## 🚀 Usage

The project is modular: you can train unimodal models (EHR-only, Report-only) or multimodal fusion models. We use Hydra, so you can override any parameter directly from the command line.

* **Unimodal Reports:**
  
```
sbatch bash/reports/run_classify_1m_mort.sh
```
---

## 🏗 Model Architecture
The framework integrates three distinct clinical data modalities using specialized encoders and various fusion strategies to optimize prognostic accuracy.

### Modality Encoders
* **CT Imaging:** Slices are processed using a **ResNetV2-101** backbone (pretrained with BigTransfer). Slice-level features are aggregated via **bidirectional GRU** and a **Hybrid Attention-and-Max Pooling** mechanism.
* **Radiology Reports:** Encoded using **Clinical-Longformer** to handle long-form clinical text. It employs a **two-level hierarchical attention mechanism** (token-level and sentence-level) to generate a 768-dimensional report embedding.
* **Structured EHR:** Processed through a **Supervised Autoencoder (EHR-AE)** with two layers to learn task-adaptive representations. For tree-based baselines, a **LightGBM** model is also supported.
![Proposed Method](figures/unimodal_approaches_page-0001.jpg)

### Fusion Strategies
1.  **Late Fusion (MEAN):** A robust strategy that averages the predicted probabilities from independent unimodal models. This approach demonstrated the most stable and highest performance (MCC) across different time horizons.
   ![Proposed Method](figures/late_fusion_page-0001.jpg)
3.  **Early Fusion:** Features from all three modalities are concatenated into a single vector before being passed to a Multi-Layer Perceptron (MLP) classifier.
   ![Proposed Method](figures/early_fusion_page-0001.jpg)
5.  **Intermediate Fusion:**
    * **ARMOUR:** Employs cross-attention and contrastive alignment to ensure robustness against missing modalities.
      ![Proposed Method](figures/armour_page-0001.jpg)
    * **CROSS:** Uses a hierarchy of Multi-Head Cross-Attention (MHCA) blocks to model complex inter-modality interactions.ù
      ![Proposed Method](figures/cross_fusion_page-0001.jpg)

---

## 📊 Dataset & Preprocessing

The study utilizes the **INSPECT dataset**, the first large-scale, public multimodal cohort for PE.

### Dataset Statistics
* **Scope:** 23,248 CTPA studies from 19,402 unique patients.
* **Targets:** All-cause mortality at 1-month, 6-month, and 12-month intervals.
* **Splitting:** Strict patient-level splits (train/val/test) are implemented to prevent data leakage.

### Preprocessing Pipelines
* **CT Imaging:**
    * Intensity values converted to **Hounsfield Units (HU)**.
    * Three standard clinical windows (**Lung, PE, and Mediastinum**) are applied and stacked into 3-channel images.
    * Slices are resized to 256×256 and center-cropped to **224×224**.
* **Radiology Reports:**
    * Text is segmented into sentences using a custom clinical-aware algorithm.
    * Tokenization is performed using the **Clinical-Longformer** tokenizer.
* **Structured EHR:**
    * Data represented as a sparse count matrix of clinical codes (ICD-10, Labs, Medications) prior to the scan date.
    * Dimensionality is reduced using **Truncated SVD (TSVD)** to 128 dimensions or via the task-specific **Supervised Autoencoder**.

---

## 🎓 Citation

If you use this code, please cite our work:
```
@article{paolomultimodal,
  title={Multimodal Clinical Data Integration for Prognosis of Pulmonary Embolism: A Comparative Study},
  author={Paolo, Domenico and Soda, Paolo and Tortora, Matteo and Bria, Alessandro and Sicilia, Rosa}
}
```
---

## 📜 License

This project is licensed. Please review the [LICENSE](LICENSE) file for more information.
