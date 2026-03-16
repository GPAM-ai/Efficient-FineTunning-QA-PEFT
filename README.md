# Efficient Fine-Tuning Methods for Portuguese Question Answering

[![Paper](https://img.shields.io/badge/Paper-PROPOR%202026-blue)](PROPOR_2026_QA(3).pdf)
[![Python](https://img.shields.io/badge/Python-3.11-green)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Models-yellow)](https://huggingface.co/neuralmind/bert-base-portuguese-cased)

This repository contains the code and results for the paper **"Efficient Fine-Tuning Methods for Portuguese Question Answering: A Comparative Study of PEFT on BERTimbau and Exploratory Evaluation of Generative LLMs"**, submitted to **PROPOR 2026**.

---

## 📝 Abstract

Large Language Models (LLMs) have transformed Natural Language Processing, yet their computational costs pose accessibility barriers for low-resource languages like **Brazilian Portuguese**. This work systematically evaluates **Parameter-Efficient Fine-Tuning (PEFT)** and quantization techniques applied to **BERTimbau** for Question Answering (QA) on the **SQuAD-BR** dataset.

### Key Findings:
1.  **LoRA Efficiency:** LoRA achieves **95.8%** of baseline performance on BERTimbau-Large, reducing training time by **73.5%**.
2.  **Learning Rate Sensitivity:** Higher learning rates (2e-4) are critical for PEFT success, yielding F1 gains of up to **+19.71 points**.
3.  **Quantization Resilience:** Larger models (Large) exhibit twice the resilience to 4-bit quantization compared to Base models.
4.  **Encoder Advantage:** Encoder-based models (BERTimbau) outperform generative models (Sabiá, Tucano) in efficiency for extractive QA, requiring up to **4.2x less GPU memory**.

---

## 📊 Experimental Results

### 1. Overview: F1 Score Across All Configurations
The heatmap below consolidates results from all 40 experiments, crossing PEFT methods, architectures (Base vs. Large), learning rates, and epochs. Note that PEFT methods (especially LoRA and DoRA) maintain stability even when Full Fine-Tuning (Full FT) collapses under high learning rates in the Large model.

<img width="1569" height="610" alt="6ec131ab-1" src="https://github.com/user-attachments/assets/235bb053-9f50-422b-9663-480785397822" />


### 2. Impact of Learning Rate
This section details the direct performance comparison under two distinct learning rates. The graph highlights how PEFT benefits from more aggressive rates (2e-4) to escape local minima.

<img width="1380" height="1580" alt="160d2f49-1" src="https://github.com/user-attachments/assets/5689f666-e24c-40ab-b5e1-8f191581bfd7" />

### 3. GPU Memory Consumption
Quantization methods (QLoRA and QDoRA) enable training Large models on consumer-grade GPUs, reducing memory usage by up to **86.9%**.

<img width="797" height="606" alt="ebc7860c-1" src="https://github.com/user-attachments/assets/47060259-7926-4b91-9548-6c689fd9a453" />

---

## 🛠 Supported Methods


This study was guided by three core hypotheses:
*   **H1 (Efficiency):** PEFT methods can match full fine-tuning performance with reduced computational cost.
*   **H2 (Scale Robustness):** Larger models are more resilient to aggressive quantization.
*   **H3 (Optimization Sensitivity):** PEFT requires significantly higher learning rates for convergence.

### Evaluated Methods
| Method | Description | Key Advantage |
| :--- | :--- | :--- |
| **Full FT** | Fine-tuning all parameters | Performance baseline |
| **LoRA** | Low-Rank Adaptation | Drastically reduced trainable parameters |
| **QLoRA** | Quantized LoRA (4-bit) | Minimal GPU memory usage |
| **DoRA** | Weight-Decomposed Low-Rank Adaptation | Directional stability in learning |
| **QDoRA** | Quantized DoRA | Combines decomposition and quantization |

---

## 🧪 Experimental Configuration

### Hardware & Software
*   **GPU:** NVIDIA RTX A4500 (20GB VRAM)
*   **Frameworks:** PyTorch 2.1.0, Transformers 4.36.0, PEFT 0.7.1, bitsandbytes 0.41.0
*   **Dataset:** SQuAD-BR (v1.1 translated to Portuguese)

### PEFT Hyperparameters
*   **Rank (r):** 16
*   **Alpha (α):** 32
*   **Target Modules:** query, key, value, output projection
*   **Dropout:** 0.1
*   **Learning Rates:** 2e-4 (PEFT-optimized) vs 4.25e-5 (Standard)

---

## 📈 Comparison with Generative Models

An exploratory evaluation was conducted with **Tucano (1.1B)** and **Sabiá (7B)**. While Sabiá-7B with LoRA achieves a competitive F1 (78.10%), it demands significantly more resources than BERTimbau.

| Model | Method | F1 Score (%) | GPU Memory (MB) | Time (hh:mm:ss) |
| :--- | :--- | :---: | :---: | :---: |
| **BERTimbau-Base** | LoRA | 78.01 | 3,687 | 00:31:37 |
| **BERTimbau-Large** | LoRA | **81.32** | 9,019 | 01:23:41 |
| **Sabiá-7B** | LoRA | 78.10 | 15,642 | 01:31:15 |
| **Tucano-1.1B** | LoRA | 63.86 | 4,301 | 00:25:28 |

---

## 🚀 Quick Start

Follow these steps to reproduce the experiments, from data preparation to final evaluation.

### 1. Setup and Environment
```bash
# Clone the repository
git clone https://github.com/Caio-Veloso1/green-ai-extractive-qa-pt.git
cd green-ai-extractive-qa-pt/qa_bertimbau/bertimbau_base 

# Install dependencies
pip install -r requirements.txt
```

### 2. Execution Pipeline
The workflow is divided into four main stages:

| Step | Command | Description |
| :--- | :--- | :--- |
| **1. Prepare Data** | `python preprocessing.py` | Pre-process the SQuAD-BR dataset |
| **2. Train Model** | `python main_lora.py` | Fine-tune with LoRA (or use `main_qlora.py`) |
| **3. Predict** | `python main_predict.py` | Generate answers on the test set |
| **4. Evaluate** | `jupyter notebook analysis_bertimbau_base.ipynb` | Detailed analysis of results and metrics |

---

## 📚 Citation

If you use this work or the data presented herein, please cite the original paper:

```bibtex
@inproceedings{nina2026efficient,
  title={Efficient Fine-Tuning Methods for Portuguese Question Answering: A Comparative Study of PEFT on BERTimbau and Exploratory Evaluation of Generative LLMs},
  author={Nina, Mariela M. and Costa, Caio Veloso and Berton, Lilian and Vega-Oliveros, Didier A.},
  booktitle={Proceedings of PROPOR 2026},
  year={2026},
  address={São José dos Campos, SP, Brazil}
}
```

---
*This repository adheres to **Green AI** principles, promoting sustainable and accessible approaches to natural language processing in Portuguese.*

