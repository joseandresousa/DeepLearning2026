# Deep Learning 2026 — Chest X-ray Classification

Multi-class classification of chest X-rays into four categories: **COVID-19, Lung Opacity, Normal, and Viral Pneumonia**, using a progression of deep learning architectures from a simple baseline to a domain-pretrained Vision Transformer ensemble.

## Dataset

[COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database/data) — 21,165 chest X-ray images (PNG, 299×299, grayscale).

| Class | Images |
|---|---|
| Normal | 10,192 |
| Lung Opacity | 6,012 |
| COVID-19 | 3,616 |
| Viral Pneumonia | 1,345 |

## Targets

| Metric | Target | Best Result |
|---|---|---|
| Test Accuracy | ≥ 92% | **94.9%** (Ensemble) |
| COVID-19 Recall | ≥ 0.90 | **0.987** (Ensemble) |

Both targets met.

## Models

| # | Model | Test Acc | COVID Recall | Notes |
|---|---|---|---|---|
| 4.1 | Baseline (Dense) | 77.8% | 0.572 | Global features only |
| 4.2 | CNN | 86.2% | 0.923 | 3 conv blocks, 299×299 |
| 4.3 | CNN + Regularisation | 85.6% | 0.952 | Dropout + L2 |
| 4.4a | MobileNetV2 | 88.6% | 0.913 | ImageNet pretrained |
| 4.4b | EfficientNetB0 | 92.4% | 0.950 | Compound scaling |
| 4.5 | RAD-DINO | 94.4% | 0.970 | Chest X-ray pretrained ViT |
| 6 | Tuned CNN (75×75) | 88.6% | 0.821 | Keras Tuner Hyperband |
| 7 | Weighted Ensemble | **94.9%** | **0.987** | RAD-DINO + EfficientNetB0 + MobileNetV2 + CNN+Reg |

## Key Findings

- **Domain pretraining is decisive**: RAD-DINO (chest X-ray pretrained) outperforms EfficientNetB0 (ImageNet pretrained) despite similar architecture families
- **More parameters ≠ better results**: RAD-DINO's 198K-parameter classifier outperforms a 22.5M-parameter scratch CNN
- **Ensembles work through diversity**: the weakest model (CNN+Reg, 85.6%) drives ensemble COVID recall to 0.987
- **Shortcut learning is real**: Grad-CAM reveals 3 of 4 classes activate on border annotations and equipment markers, not lung pathology

## Structure

```
DP_JS.ipynb          — Main notebook (sections 1–10)
DP_JS.html           — Rendered HTML export
models/              — Saved model weights and training metadata
```

## Hardware

NVIDIA Quadro RTX 3000 (6GB VRAM), Intel Core i7, 32GB RAM. Training times range from 1.8 min (Baseline) to ~30 min (RAD-DINO feature extraction).

## References

- Rahman et al. (2021) — *Exploring the Effect of Image Enhancement Techniques on COVID-19 Detection using Chest X-ray Images* — Computers in Biology and Medicine
- Selvaraju et al. (2017) — *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization* — [arXiv:1610.02391](https://arxiv.org/abs/1610.02391)
- Pérez-García et al. (2024) — *RAD-DINO: Exploring Scalable Medical Image Encoders Beyond Text Supervision* — [arXiv:2401.10815](https://arxiv.org/abs/2401.10815)

