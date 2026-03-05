# FiLM-Enhanced WGAN for Conditional Password Generation

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Kaggle%20%7C%20colab%20%7C%20WSL-lightgrey)

Official implementation of the research paper:

**“Conditional Password Generation Using a FiLM-Enhanced WGAN: A Controlled Comparison Against Standard GAN Baselines.”**

Published in:
**5th IEEE International Conference on AI in Cybersecurity (ICAIC 2026)**
University of Houston, USA.

---

# Abstract

This project investigates the use of **Feature-wise Linear Modulation (FiLM)** within a conditional **Wasserstein GAN (WGAN-GP)** for modeling password distributions.

The study compares four architectures under a unified preprocessing and training pipeline:

* PassGAN
* PaC-GAN
* WGAN-CGAN
* **FiLM-WGAN-CGAN (proposed model)**

The FiLM-enhanced generator improves **conditional fidelity and structural consistency** while maintaining competitive diversity.

All experiments follow strict ethical guidelines and report **aggregate statistics only**.

---

# Repository Structure

```
film-wgan-password-generation
│
├── paper
│   └── GAN_final_paper.pdf
│
├── notebooks
│   ├── 01_preprocess.ipynb
│   ├── 02_train_model.ipynb
│   └── 03_evaluation.ipynb
│
├── src
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── model.py
│
├── figures
│   ├── architecture.png
│   ├── loss_curve.png
│
├── requirements.txt
├── README.md
├── LICENSE
└── .github/workflows/build.yml
```

---

# Dataset

This project uses the **PasswordCollection dataset**.

Source:

https://github.com/yuqian5/PasswordCollection

The dataset is **not included in this repository**.

Preprocessing converts passwords into a **fixed-length integer representation (16 tokens)**.

---

# Preprocessing Pipeline

The preprocessing pipeline performs the following steps:

1. Unicode normalization (NFKD)
2. ASCII filtering (94 printable characters)
3. Password truncation or padding to length **16**
4. Integer token encoding
5. Export dataset as:

```
train_data.npy
chars.txt
```

---

# Training Configuration

All models are trained using identical hyperparameters:

| Parameter         | Value     |
| ----------------- | --------- |
| Optimizer         | Adam      |
| β1                | 0         |
| β2                | 0.9       |
| Generator LR      | 2×10⁻⁵    |
| Critic LR         | 1×10⁻⁴    |
| Batch Size        | 96        |
| Latent Dimension  | 128       |
| Sequence Length   | 16        |
| Critic Iterations | 5         |
| Gradient Penalty  | λ = 10    |
| Training Budget   | 60 epochs |

Conditional experiments use a **balanced subset of 250k samples**.

---

# Training Environment

Experiments were executed on **Kaggle notebooks** using:

* GPU: **NVIDIA T4 (16 GB VRAM)**
* Python: **3.11**
* TensorFlow: **2.15**

---

# Evaluation Metrics

The following metrics are used for evaluation:

### Uniqueness

Fraction of unique samples among generated outputs.

### N-gram Coverage

Coverage of **2-gram, 3-gram, and 4-gram patterns** between generated and real samples.

### Character-Class Fidelity

Comparison of character class distributions between generated and real data.

### Conditional Fidelity

Accuracy of generating passwords within the requested **length bucket**.

---

# Quick Start

Install dependencies:

```
pip install -r requirements.txt
```

Preprocess dataset:

```
python src/preprocess.py
```

Train model:

```
python src/train.py
```

Evaluate results:

```
python src/evaluate.py
```

---

# Ethical Considerations

This repository does **not release raw password data or generated password samples**.

Only **aggregate statistics** are reported to prevent misuse.

---

# Citation

If you use this work, please cite:

```
@inproceedings{film_wgan_password_2026,
title={Conditional Password Generation Using a FiLM-Enhanced WGAN},
booktitle={IEEE International Conference on AI in Cybersecurity},
year={2026}
}
```

---

# License

This project is licensed under the **MIT License**.

See `LICENSE` for details.


The dataset is downloaded during preprocessing and is not included in this repository.
