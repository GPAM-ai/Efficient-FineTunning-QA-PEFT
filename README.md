# Efficient QA for Brazilian Portuguese: PEFT & Quantization on BERTimbau

This repository contains the code and findings from a systematic evaluation of **Parameter-Efficient Fine-Tuning (PEFT)** and **quantization** techniques applied to **BERTimbau** for extractive Question Answering on the **SQuAD-BR** dataset.

## 🚀 Overview
As Large Language Models (LLMs) grow, computational costs become a barrier for low-resource languages. This project explores more sustainable **"Green AI"** approaches by optimizing encoder-based models for Brazilian Portuguese, proving they can remain competitive with generative models while using significantly fewer resources.

## 📊 Key Results
*   **Performance:** LoRA on BERTimbau-Large achieves **95.8%** of the baseline F1 score.
*   **Efficiency:** Training time reduced by **73.5%**.
*   **Optimization:** Higher learning rates (**2e-4**) improved PEFT performance by up to **+19.71 F1 points**.
*   **Resilience:** Larger models (335M) are twice as resilient to quantization than smaller versions (110M).
*   **Encoder vs. Generative:** BERTimbau-Base requires **3x less training time** and **4.2x less GPU memory** than generative models like Tucano or Sabiá for the same task.

## 🛠 Supported Methods
We evaluated **40 configurations** combining:
*   **Models:** BERTimbau (Base & Large).
*   **PEFT Methods:** LoRA, DoRA, QLoRA, and QDoRA.
*   **Comparisons:** Exploratory benchmarks against generative models (Tucano, Sabiá).
