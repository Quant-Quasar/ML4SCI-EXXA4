# ML4SCI EXXA4 — Foundation Models for Exoplanet Characterization

[![GSoC 2026](https://img.shields.io/badge/GSoC-2026-blue.svg)](https://summerofcode.withgoogle.com/)
[![ML4SCI](https://img.shields.io/badge/org-ML4SCI-green.svg)](https://ml4sci.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Applicant:** Aditya Parashar  
**Institution:** Birla Institute of Technology, Mesra  
**Project:** [EXXA4 — Foundation Models for Exoplanet Characterization](https://ml4sci.org/gsoc/2026/proposal_EXXA4.html)  
**Mentors:** Katia Matcheva, Konstantin Matchev, Sergei Gleyzer, Jason Terry, Alex Roman, Emilie Panek

---

## Overview

This repository implements a self-supervised foundation model framework for exoplanet
characterization. The evaluation test combines BYOL (Bootstrap Your Own Latent)
pre-training on ALMA protoplanetary disk images with physics-informed radial profile
features, an enhanced autoencoder with accessible latent space, and a CNN + BiGRU +
Attention transit light curve classifier.

**Note on ground truth:** No labels are provided with the disk dataset.
The filename prefix `planet{N}` is a simulation configuration ID, not a planet count.
All clustering evaluation uses internal metrics (silhouette score, cluster stability)
and visual inspection.

---

## Results

### General Test — BYOL Clustering of 150 ALMA Disk Images

| Configuration | Silhouette |
|---|---|
| Baseline AE + KMeans (no UMAP) | 0.3826 |
| Baseline AE + UMAP + KMeans | 0.6182 |
| BYOL + KMeans (no UMAP) | 0.4724 |
| BYOL + UMAP + KMeans | **0.7149** |
| BYOL + UMAP + HDBSCAN | **0.7361** |
| BYOL + Radial + Gaps (Exp D, physics ablation) | 0.6823 |

Key finding from Experiment D: adding physics-informed radial profile features *reduces*
silhouette from 0.7149 → 0.6823, confirming the BYOL backbone already encodes
morphological substructure that the hand-crafted features attempt to add.

### Image-Based Test — Enhanced Autoencoder

| Metric | Value |
|---|---|
| Reconstruction MSE | 0.000456 |
| MS-SSIM (upsampled to 256px) | 0.9559 |
| Latent dimension | 64 |
| `model.encode(x)` API | ✅ |

### Sequential Test — Transit Light Curve Classifier

| Metric | Value |
|---|---|
| Test AUC | 0.9701 |
| Test Accuracy | 93.2% |
| Hard-tier Recall (500–1500 ppm) | 0.7518 |
| PR-AUC | 0.9773 |
| Optimal threshold (Youden's J) | 0.6788 |

---

## Deliverables

| Deliverable | Status |
|---|---|
| `run_disk_inference(data_dir)` — full BYOL clustering pipeline on new FITS files | ✅ |
| `run_transit_inference(X_new)` — classification with MC Dropout uncertainty | ✅ |
| `model.encode(x)` — accessible autoencoder latent space | ✅ |
| Pre-trained weights on Google Drive | ✅ |
| Held-out generalisation on 5 unseen disk morphologies | ✅ |

---

## Pre-trained Weights

**[Download Pre-trained Weights (Google Drive)](https://drive.google.com/drive/folders/1my557Ch23qR8teEVb64L0AroBJlSqYO8?usp=sharing)**

| File | Description |
|---|---|
| `byol_backbone.pt` | BYOL ResNet-18 backbone (512-dim, single-channel) |
| `enhanced_ae.pt` | Enhanced autoencoder (3.8M parameters) |
| `transit_clf.pt` | CNN + BiGRU + Attention classifier (197K parameters) |

---

## Repository Structure

```
ML4SCI-EXXA4/
│
├── exxa4_foundation_model_test.ipynb   # Complete evaluation test — all three tasks
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

Notebooks for the GSoC coding period (Stages 2–6) will be added starting May 27, 2026:

```
├── 02_spectral_byol_pretraining.ipynb     # 1D BYOL pre-training on HELA spectra
├── 03_foundation_model_finetuning.ipynb   # Retrieval head + R²/MAE vs baselines
├── 04_uncertainty_quantification.ipynb    # MC Dropout + ensemble + ECE calibration
├── 05_cross_modal_transfer.ipynb          # Disk → spectrum transfer experiment
└── 06_jwst_generalisation.ipynb           # 256-channel JWST extension + real data
```

---

## Quick Start

```bash
git clone https://github.com/Quant-Quasar/ML4SCI-EXXA4.git
cd ML4SCI-EXXA4
pip install -r requirements.txt
```

Open `exxa4_foundation_model_test.ipynb` in Google Colab, mount Drive, and run all cells.  
**Runtime:** ~40 minutes on a T4 GPU. No user intervention required after Drive mount.

### Inference APIs

```python
# Disk morphology clustering on new FITS files
assignments, embeddings = run_disk_inference('/path/to/fits/dir')

# Transit classification with uncertainty
probs, predictions, uncertainty = run_transit_inference(X_new)
# X_new: (N, 200) numpy array of normalised light curves

# Autoencoder latent encoding
latents = ae.encode(images)  # images: (B, 1, 128, 128) → latents: (B, 64)
```

---

## Architecture

### BYOL Backbone
- ResNet-18, modified for single-channel ALMA input (`Conv2d(1, 64, 7, 2, 3)`)
- Online: backbone + projector MLP(512→2048→256) + predictor MLP
- Target: backbone + projector, EMA-updated (τ cosine-annealed 0.996→1.0, 500 epochs)
- Augmentations: rotation 360°, flips, RandomResizedCrop, Gaussian blur, contrast jitter, threshold masking

### Enhanced Autoencoder
- Encoder: Conv2d(1→32→64→128→256, stride-2) + AdaptiveAvgPool(4×4) + Linear(4096→64)
- Decoder: Linear(64→4096) + 5× ConvTranspose2d → Sigmoid
- Training: Phase 1 MSE (60 epochs) → Phase 2 MSE + MS-SSIM α=0.85 (60 epochs)

### CNN + BiGRU + Attention Classifier
- CNN stem: Conv1d kernels 7→5→3 with MaxPool between stages
- BiGRU: hidden=64, 2 layers, bidirectional, dropout=0.3
- Additive (Bahdanau) attention over GRU time steps → 128-dim context
- 197,186 total parameters

### Physics Features (Experiment D)
- Azimuthally-averaged radial intensity profiles (64-dim) via median binning
- Automated gap detection using Gaussian-smoothed peak-finding (scipy)
- Combined with BYOL embeddings (radial ×2, gap count ×3 weighted)

---

## References

**Self-Supervised Learning**
- Grill et al. (2020). Bootstrap Your Own Latent. *NeurIPS 33*. [arXiv:2006.07733](https://arxiv.org/abs/2006.07733)
- He et al. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*. doi:10.1109/CVPR.2016.90
- Slijepcevic et al. (2024). Radio Galaxy Zoo BYOL. *MNRAS* 530(1):1274. doi:10.1093/mnras/stae1048
- Mohale & Lochner (2024). Unsupervised Discovery via SSL. [arXiv:2311.14157](https://arxiv.org/abs/2311.14157)

**Dimensionality Reduction & Clustering**
- McInnes et al. (2018). UMAP. [arXiv:1802.03426](https://arxiv.org/abs/1802.03426)
- Campello et al. (2013). HDBSCAN. *PAKDD 2013*. doi:10.1007/978-3-642-37456-2_14

**Image Quality**
- Wang et al. (2004). SSIM. *IEEE Trans. Image Processing* 13(4):600–612. doi:10.1109/TIP.2003.819861

**Exoplanet Science**
- Forestano et al. (2025). Supervised ML with UQ for exoplanet retrieval. [arXiv:2508.04982](https://arxiv.org/abs/2508.04982)
- Márquez-Neila et al. (2018). Supervised ML for exoplanetary spectra. *Nature Astronomy* 2:719–724.
- Terry et al. (2022). Locating Hidden Exoplanets in ALMA Data. *ApJ* 941(2):192. doi:10.3847/1538-4357/aca477
- Kanagawa et al. (2016). Planet Gap Width. *PASJ* 68:43. doi:10.1093/pasj/psw037
- Andrews et al. (2018). DSHARP. *ApJL* 869:L41. doi:10.3847/2041-8213/aaf741
- Changeat & Yip (2023). ABC Database. *RASTI* 2(1):45. [zenodo:6770103](https://doi.org/10.5281/zenodo.6770103)
- Tinetti et al. (2018). ARIEL. *Experimental Astronomy* 46:135. doi:10.1007/s10686-018-9598-x

**Transit Modelling**
- Kreidberg (2015). batman. *PASP* 127(957):1161–1165. doi:10.1086/683602

**Software**
- Paszke et al. (2019). PyTorch. *NeurIPS 32*.
- Astropy Collaboration (2022). Astropy v5.0. *ApJ* 935(2):167. doi:10.3847/1538-4357/ac7c74
- Pedregosa et al. (2011). Scikit-learn. *JMLR* 12:2825–2830.

---
*GSoC 2026 — ML4SCI EXXA4 | [adityaparashar3434@gmail.com](mailto:adityaparashar3434@gmail.com)*