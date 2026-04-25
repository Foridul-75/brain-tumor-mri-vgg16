# Brain Tumor MRI Classification 🧠

A research-grade deep learning pipeline for classifying brain tumors from MRI scans using a fine-tuned **VGG16** architecture, with full **explainability (Grad-CAM, Grad-CAM++, LIME)**, **5-fold stratified cross-validation**, and **clinical-grade statistical validation** (Bootstrap 95% CI, Cohen's Kappa, MCC, Sensitivity/Specificity/PPV/NPV).

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success)

---

## 📋 Overview

This project implements a **clinically-aware brain tumor MRI classifier** that distinguishes between four classes:

| Class | Description |
|-------|-------------|
| 🧠 **Glioma** | Tumors arising from glial cells |
| 🧠 **Meningioma** | Tumors of the meninges (brain/spinal cord membranes) |
| 🧠 **Pituitary** | Tumors of the pituitary gland |
| ✅ **No Tumor** | Healthy brain MRI scans |

Unlike basic CNN classifiers, this pipeline emphasizes **statistical rigor** and **interpretability** — making it suitable for thesis work, research papers, and real-world medical AI evaluation.

---

## 🎯 Key Features

- ✅ **VGG16 Transfer Learning** — ImageNet pretrained, last 15 conv layers fine-tuned
- ✅ **5-Fold Stratified Cross Validation** — robust performance estimation
- ✅ **Class Imbalance Handling** — per-fold computed class weights
- ✅ **Mixed Precision Training (AMP)** — faster training on GPU
- ✅ **Best-Epoch Selection by Macro F1** — tracks F1 instead of accuracy (better for imbalanced data)
- ✅ **Early Stopping + LR Scheduling** — `ReduceLROnPlateau` with patience
- ✅ **Bootstrap 95% Confidence Intervals** — for every metric
- ✅ **Cohen's Kappa + Matthews Correlation Coefficient** — robust to imbalance
- ✅ **Clinical Metrics** — Sensitivity, Specificity, PPV, NPV per class
- ✅ **Three XAI Methods** — Grad-CAM, Grad-CAM++, and LIME
- ✅ **McNemar Test** — statistical comparison across model architectures
- ✅ **Live Streamlit Demo App** — interactive image upload + prediction + Grad-CAM

---

## 🛠️ Technologies

| Stack | Tools |
|-------|-------|
| **Deep Learning** | PyTorch, torchvision |
| **Augmentation** | Albumentations |
| **Explainability** | Grad-CAM, Grad-CAM++, LIME, scikit-image |
| **Statistical Validation** | scikit-learn, statsmodels (McNemar test) |
| **Visualization** | Matplotlib, Seaborn, OpenCV |
| **Demo Frontend** | Streamlit |
| **Numerical / Image** | NumPy, Pandas, Pillow |

---

## 📊 Dataset

The model is trained on a 4-class brain tumor MRI dataset organized as:

```
Brain_MRI_Images/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

Recommended public source: [Brain Tumor MRI Dataset on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).

---

## 🏗️ Model Architecture

**Backbone:** VGG16 pretrained on ImageNet
**Custom Classifier Head:**

```
Linear(25088 → 512) → ReLU → Dropout(0.5)
Linear(512   → 256) → ReLU → Dropout(0.4)
Linear(256   → 4)
```

**Fine-tuning Strategy:** The last **15 convolutional layers** of VGG16 are unfrozen and fine-tuned on the brain MRI domain. Earlier layers (low-level features) remain frozen to preserve ImageNet representations.

**Inplace ReLU disabled** throughout — required for Grad-CAM gradient hooks.

---

## ⚙️ Training Configuration

| Hyperparameter | Value |
|---|---|
| Image size | 224 × 224 |
| Batch size | 32 |
| Max epochs | 50 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Early stopping patience | 8 epochs |
| Loss | CrossEntropyLoss with per-fold class weights |
| Selection metric | Macro F1 |
| Mixed precision | Enabled (CUDA) |
| Cross validation | 5-fold StratifiedKFold |
| Random seed | 42 |

**Augmentation pipeline (training only):**
HorizontalFlip · Rotate ±10° · RandomScale ±10% · Brightness/Contrast ±10% · CLAHE · GridDistortion · CoarseDropout

---

## 📈 Results

> 📌 *Replace these placeholder values with your actual run output from `cv_summary` and `test_result`.*

### Cross-Validation Performance (5-Fold)

| Metric | Mean ± Std | 95% Bootstrap CI |
|--------|-----------|------------------|
| Accuracy | 0.XXXX ± 0.XXXX | [0.XXXX, 0.XXXX] |
| Macro F1 | 0.XXXX ± 0.XXXX | [0.XXXX, 0.XXXX] |
| Macro AUC | 0.XXXX ± 0.XXXX | [0.XXXX, 0.XXXX] |
| Cohen's Kappa | 0.XXXX ± 0.XXXX | [0.XXXX, 0.XXXX] |
| MCC | 0.XXXX ± 0.XXXX | [0.XXXX, 0.XXXX] |

### Held-Out Test Set Performance

| Metric | Value | 95% CI |
|--------|-------|--------|
| Accuracy | 0.XXXX | [0.XXXX, 0.XXXX] |
| Macro Precision | 0.XXXX | [0.XXXX, 0.XXXX] |
| Macro Recall | 0.XXXX | [0.XXXX, 0.XXXX] |
| Macro F1 | 0.XXXX | [0.XXXX, 0.XXXX] |
| Macro AUC | 0.XXXX | [0.XXXX, 0.XXXX] |

### Per-Class Clinical Metrics

| Class | Sensitivity | Specificity | PPV | NPV |
|-------|------------|------------|-----|-----|
| Glioma | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |
| Meningioma | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |
| No Tumor | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |
| Pituitary | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |

The training pipeline automatically generates confusion matrices (raw + normalized), ROC curves, and per-fold loss/accuracy plots.

---

## 🔍 Explainable AI (XAI)

Three complementary explanation methods are implemented to verify the model attends to clinically meaningful regions rather than spurious image artifacts:

| Method | What it shows |
|--------|--------------|
| **Grad-CAM** | Class-discriminative spatial heatmap from final conv layer gradients |
| **Grad-CAM++** | Improved localization for multi-instance and small-object cases |
| **LIME** | Superpixel-level positive/negative contribution map |

Each method produces:
- Raw heatmap
- Image-overlay visualization
- Bounding-box localization at a configurable activation threshold

A per-class XAI grid is generated automatically for thesis figures.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/brain-tumor-mri-vgg16.git
cd brain-tumor-mri-vgg16
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset

Download the Brain Tumor MRI Dataset and structure it as shown in [Dataset](#-dataset). Update the paths in the notebook config:

```python
TRAIN_ROOT = "path/to/Brain_MRI_Images/Training"
TEST_ROOT  = "path/to/Brain_MRI_Images/Testing"
```

### 4. Train the model

Run the training notebook end-to-end:

```bash
jupyter notebook VGG16_Dataset1_ColabReady.ipynb
```

Training takes approximately **2–3 hours on a single GPU** (T4 / V100 / A100 / RTX 3060+). The best fold's checkpoint is saved to `checkpoints_VGG16_Dataset1/`.

### 5. Launch the live demo

```bash
streamlit run Brain_Tumor_MRI_APP.py
```

Then open [http://localhost:8501](http://localhost:8501), upload an MRI scan, and view the predicted class with confidence scores and Grad-CAM heatmap overlay.

---

## 📁 Project Structure

```
brain-tumor-mri-vgg16/
├── VGG16_Dataset1_ColabReady.ipynb   # Full training pipeline (Colab-ready)
├── Brain_Tumor_MRI_APP.py            # Streamlit demo application
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
├── checkpoints_VGG16_Dataset1/       # Saved model weights (.pt)
├── results_VGG16_Dataset1/           # Per-fold metrics, plots, JSON summary
└── artifacts_VGG16_Dataset1/         # Test predictions for McNemar comparison
```

---

## 🌐 Deployment

The Streamlit demo app can be deployed for free on:

- **[Streamlit Community Cloud](https://streamlit.io/cloud)** — connect your GitHub repo and deploy in one click
- **[Hugging Face Spaces](https://huggingface.co/spaces)** — select Streamlit SDK
- **Render / Railway / Fly.io** — for persistent hosting with a model file mounted

> **Note:** Model checkpoint files are typically too large for direct GitHub commit. Use [Git LFS](https://git-lfs.com/) or host the `.pt` file on Hugging Face Hub / Google Drive and download it on app startup.

---

## 📚 Methodology Notes

This pipeline was designed with the following research-quality principles:

1. **No data leakage** — stratified splits preserve class distribution; no test sample appears in any training fold.
2. **Reproducibility** — all randomness controlled via a single `SEED` (Python, NumPy, PyTorch, CUDA).
3. **Honest evaluation** — best epoch selected by **validation Macro F1** (not training accuracy) to avoid overfitting bias.
4. **Confidence intervals on everything** — point estimates without CIs are misleading; bootstrap resampling gives proper uncertainty bounds.
5. **Imbalance-aware metrics** — Kappa and MCC are reported alongside accuracy because medical datasets are rarely balanced.
6. **Explainability is mandatory** — a black-box medical classifier is not deployable; three independent XAI methods cross-verify model attention.

---

## 🔬 Cross-Model Statistical Comparison

The pipeline includes a **McNemar exact test** helper for comparing two trained models on identical test predictions:

```python
mcnemar_compare_from_files("VGG16", "ResNet50")
```

This produces a contingency table and exact p-value, allowing rigorous "Is model A significantly better than model B?" claims for thesis chapters.

---

## 📖 Citation

If you use this work in academic research, please cite:

```bibtex
@misc{islam2025braintumor,
  author       = {MD Foridul Islam},
  title        = {Brain Tumor MRI Classification with VGG16: A Statistically-Validated, Explainable Deep Learning Pipeline},
  year         = {2025},
  howpublished = {\url{https://github.com/yourusername/brain-tumor-mri-vgg16}}
}
```

---

## 👤 Author

**MD Foridul Islam**
Computer Science and Engineering — Daffodil International University (DIU)

- 🐙 GitHub: [@yourusername](https://github.com/yourusername)
- 💼 LinkedIn: [your-linkedin-handle](https://linkedin.com/in/your-linkedin-handle)
- 📧 Email: your.email@example.com

---

## 📜 License

This project is released under the **MIT License**. See `LICENSE` for details.

---

## 🙏 Acknowledgments

- Brain Tumor MRI Dataset by [Masoud Nickparvar](https://www.kaggle.com/masoudnickparvar) on Kaggle
- VGG16 architecture: Simonyan & Zisserman, *Very Deep Convolutional Networks for Large-Scale Image Recognition* (2014)
- Grad-CAM: Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization* (2017)
- LIME: Ribeiro et al., *"Why Should I Trust You?": Explaining the Predictions of Any Classifier* (2016)

---

<div align="center">

**⭐ If this project helped your research, please star the repository!**

</div>
