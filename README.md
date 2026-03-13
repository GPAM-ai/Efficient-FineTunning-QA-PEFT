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



-------------------------------------------------------------------

# Efficient Fine-Tuning Methods for Portuguese Question Answering

[![Paper](https://img.shields.io/badge/Paper-PROPOR%202026-blue)](PROPOR_2026_QA(3).pdf)
[![Python](https://img.shields.io/badge/Python-3.11-green)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Models-yellow)](https://huggingface.co/neuralmind/bert-base-portuguese-cased)

Este repositório contém o código e os resultados do artigo **"Efficient Fine-Tuning Methods for Portuguese Question Answering: A Comparative Study of PEFT on BERTimbau and Exploratory Evaluation of Generative LLMs"**, submetido ao **PROPOR 2026**.

## 📝 Resumo 

Embora os grandes modelos de linguagem (LLMs) tenham transformado o processamento de linguagem natural, seus custos computacionais criam barreiras de acessibilidade para línguas de baixos recursos, como o **Português Brasileiro**. Este trabalho apresenta uma avaliação sistemática de técnicas de **Parameter-Efficient Fine-Tuning (PEFT)** e quantização aplicadas ao **BERTimbau** para a tarefa de Question Answering (QA) no dataset **SQuAD-BR**.

### Principais Descobertas:
1.  **Eficiência do LoRA:** O LoRA atinge **95,8%** do desempenho do baseline no BERTimbau-Large, reduzindo o tempo de treinamento em **73,5%**.
2.  **Sensibilidade à Taxa de Aprendizado:** Taxas de aprendizado mais altas (2e-4) são críticas para o sucesso do PEFT, resultando em ganhos de até **+19,71 pontos de F1**.
3.  **Resiliência à Quantização:** Modelos maiores (Large) apresentam o dobro da resiliência à quantização de 4 bits em comparação aos modelos Base.
4.  **Vantagem do Encoder:** Modelos baseados em encoder (BERTimbau) superam modelos generativos (Sabiá, Tucano) em eficiência para QA extrativo, exigindo até **4,2x menos memória GPU**.

---

## 🚀 Uso Rápido

Siga os passos abaixo para reproduzir os experimentos do artigo, desde a preparação dos dados até a avaliação final.

### 1. Instalação e Ambiente
```bash
# Clonar o repositório
git clone https://github.com/Caio-Veloso1/green-ai-extractive-qa-pt.git
cd green-ai-extractive-qa-pt/qa_bertimbau/bertimbau_base 

# Instalar dependências
pip install -r requirements.txt
```

### 2. Pipeline de Execução
O fluxo de trabalho é dividido em quatro etapas principais:

| Passo | Comando | Descrição |
| :--- | :--- | :--- |
| **1. Preparar Dados** | `python preprocessing.py` | Pré-processamento do dataset SQuAD-BR |
| **2. Treinar Modelo** | `python main_lora.py` | Fine-tuning com LoRA (ou use `main_qlora.py`) |
| **3. Predição** | `python main_predict.py` | Geração de respostas no conjunto de teste |
| **4. Avaliação** | `jupyter notebook analysis_bertimbau_base.ipynb` | Análise detalhada dos resultados e métricas |

## 📊 Resultados Experimentais

### 1. Visão Geral: F1 Score em Todas as Configurações
O heatmap abaixo consolida os resultados de todos os 40 experimentos, cruzando métodos PEFT, arquiteturas (Base vs. Large), taxas de aprendizado e épocas. Note que os métodos PEFT (especialmente LoRA e DoRA) mantêm a estabilidade mesmo quando o Full Fine-Tuning (Full FT) colapsa sob taxas de aprendizado elevadas no modelo Large.

<img width="1609" height="610" alt="9475c0d2-1" src="https://github.com/user-attachments/assets/4446ca08-4c51-4cf9-b54b-aca224f9cb60" />


### 2. Performance (F1 Score) vs. Taxa de Aprendizado
Abaixo, comparamos o desempenho de diferentes métodos PEFT (LoRA, QLoRA, DoRA, QDoRA) sob duas taxas de aprendizado distintas. Note o colapso do Full Fine-Tuning (Full FT) no modelo Large quando utilizada uma taxa de aprendizado alta, enquanto os métodos PEFT permanecem estáveis.

<img width="1380" height="1580" alt="160d2f49-1" src="https://github.com/user-attachments/assets/5689f666-e24c-40ab-b5e1-8f191581bfd7" />


### 3. Consumo de Memória GPU
Os métodos de quantização (QLoRA e QDoRA) permitem o treinamento de modelos Large em GPUs de consumo, reduzindo o uso de memória em até **86,9%**.

<img width="797" height="606" alt="ebc7860c-1" src="https://github.com/user-attachments/assets/47060259-7926-4b91-9548-6c689fd9a453" />


---

## 💡 Hipóteses

O estudo foi guiado por três hipóteses centrais:
*   **H1 (Eficiência):** Métodos PEFT podem igualar o desempenho do fine-tuning total com custo reduzido.
*   **H2 (Robustez de Escala):** Modelos maiores são mais resilientes à quantização agressiva.
*   **H3 (Otimização):** PEFT requer taxas de aprendizado significativamente maiores para convergir.

### Métodos Avaliados
| Método | Descrição | Vantagem Principal |
| :--- | :--- | :--- |
| **Full FT** | Fine-tuning de todos os parâmetros | Baseline de performance |
| **LoRA** | Low-Rank Adaptation | Redução drástica de parâmetros treináveis |
| **QLoRA** | Quantized LoRA (4-bit) | Mínimo uso de memória GPU |
| **DoRA** | Weight-Decomposed Low-Rank Adaptation | Estabilidade direcional no aprendizado |
| **QDoRA** | Quantized DoRA | Combinação de decomposição e quantização |

---

## 🧪 Configuração Experimental

### Hardware & Software
*   **GPU:** NVIDIA RTX A4500 (20GB VRAM)
*   **Frameworks:** PyTorch 2.1.0, Transformers 4.36.0, PEFT 0.7.1, bitsandbytes 0.41.0
*   **Dataset:** SQuAD-BR (v1.1 traduzido para Português)

### Hiperparâmetros PEFT
*   **Rank (r):** 16
*   **Alpha (α):** 32
*   **Target Modules:** query, key, value, output projection
*   **Dropout:** 0.1
*   **Learning Rates:** 2e-4 (PEFT-optimized) vs 4.25e-5 (Standard)

---

## 📈 Comparação com Modelos Generativos

Realizamos uma avaliação exploratória com os modelos **Tucano (1.1B)** e **Sabiá (7B)**. Embora o Sabiá-7B com LoRA atinja um F1 competitivo (78,10%), ele exige significativamente mais recursos que o BERTimbau.

| Modelo | Método | F1 Score (%) | Memória GPU (MB) | Tempo (hh:mm:ss) |
| :--- | :--- | :---: | :---: | :---: |
| **BERTimbau-Base** | LoRA | 78,01 | 3.687 | 00:31:37 |
| **BERTimbau-Large** | LoRA | **81,32** | 9.019 | 01:23:41 |
| **Sabiá-7B** | LoRA | 78,10 | 15.642 | 01:31:15 |
| **Tucano-1.1B** | LoRA | 63,86 | 4.301 | 00:25:28 |

---

## 📚 Citação

Se você utilizar este trabalho ou os dados aqui apresentados, por favor cite o artigo original:

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
*Este repositório segue os princípios de **Green AI**, promovendo abordagens sustentáveis e acessíveis para o processamento de linguagem natural em português.*

