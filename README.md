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

python train.py

---

## 🏗 Model Architecture
The pipeline consists of three main components:
1.  **Feature Extractor**: An EfficientNetB0 backbone that processes $N$ slices per patient.
2.  **Attention Mechanism**: A dedicated layer that computes attention scores $\alpha_i$ for each slice, aggregating them into a single context vector.
3.  **Survival Predictor**: A Fully Connected network that outputs the risk distribution across 24 monthly time bins.

![Proposed Method](figures/Method.png)

---

## 📊 Dataset & Preprocessing
The model was validated on the public **INSPECT** dataset.

### Preprocessing Pipeline:
* **Resampling**: Voxel spacing standardized to $1 \times 1 \times 3$ mm.
* **Lung Masking**: Slices are filtered based on lung area (threshold > 2%) using a U-Net segmenter.
* **HU Clipping**: Hounsfield Units clipped to $[-1000, 400]$ range.
* **Normalization**: Min-Max scaling and resizing to $224 \times 224$ pixels.

![Preprocessing](figures/Preprocessing.png)

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
