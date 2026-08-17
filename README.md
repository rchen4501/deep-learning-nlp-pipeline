# Multi-Task Transformer NLP Pipeline: Training & Fine-Tuning BERT from Scratch on Downstream Benchmarks

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co)
[![Hardware](https://img.shields.io/badge/Compute-NVIDIA%20T4%20GPU-76B900?style=flat&logo=nvidia&logoColor=white)](https://nvidia.com)

An end-to-end Natural Language Processing engineering pipeline implementing custom subword tokenization, transfer learning architectures, full model training/fine-tuning execution across GLUE benchmarks and reading comprehension datasets, and real-time interactive inference tooling.

---

## 📌 Project Overview

While base Transformer checkpoints provide foundational language representations, adapting them to specialized downstream NLP tasks requires extensive engineering of tokenization boundaries, dynamic loss configurations, gradient-based parameter updates, and rigorous metric evaluation.

In this project, I engineered and executed full supervised fine-tuning regimes for `bert-base-uncased` across three distinct task formulations:
1. **Single-Sequence Sentiment Classification** (GLUE SST-2)
2. **Sentence-Pair Semantic Paraphrase Detection** (GLUE MRPC)
3. **Span-Extractive Question Answering** (SQuAD v1.1)
4. **Continual Transfer Learning Analysis** (Sequential SST-2 $\rightarrow$ MRPC transfer)

---

## 🧠 Technical Implementation & Tokenization Engineering

### 1. WordPiece Subword Tokenization & Dynamic Formatting
* **Subword Parsing**: Visualized and handled WordPiece subword decompositions and vocabulary alignments.
* **Special Token Management**: Configured sequence delimiters (`[CLS]`, `[SEP]`) to demarcate single-sequence inputs and sentence-pair boundaries for dual-sequence classification heads.
* **Span-to-Token Offset Alignment**: Engineered character-to-token offset mapping for extractive QA, mapping raw character answer indices in context paragraphs to exact subword token start and end positions.

```text
Single-Sentence (SST-2):   [CLS] + Sentence Tokens + [SEP]
Sentence-Pair (MRPC):      [CLS] + Sentence 1 + [SEP] + Sentence 2 + [SEP]
Extractive QA (SQuAD):     [CLS] + Question + [SEP] + Context Passage + [SEP]

## 🏋️ Model Training & Fine-Tuning Regimes

All training runs were executed on hardware-accelerated GPU runtimes using PyTorch and the Hugging Face `Trainer` API with AdamW optimization, linear learning rate warmup schedules, and weight decay regularization.

### 1. Single-Sequence Classification: SST-2 (Stanford Sentiment Treebank)
* **Objective**: Predict binary sentiment polarity (Positive / Negative) from user review sequences.
* **Architecture**: Sequence classification head with linear projection over the pooled `[CLS]` token.
* **Hyperparameters**:
  * **Epochs**: `3`
  * **Learning Rate**: `2e-5`
  * **Batch Size**: `32`
  * **Optimizer**: `AdamW` (Weight Decay: `0.01`)
* **Evaluation Metric**: Validation Accuracy (Target baseline: >90%).

---

### 2. Sentence-Pair Paraphrase Detection: MRPC (Microsoft Research Paraphrase Corpus)
* **Objective**: Predict semantic equivalence between pairs of sentences.
* **Architecture**: Cross-attention sentence-pair classification head.
* **Hyperparameters**:
  * **Epochs**: `3`
  * **Learning Rate**: `2e-5`
  * **Batch Size**: `32`
* **Evaluation Metrics**: Validation Accuracy & F1-Score.

---

### 3. Extractive Question Answering: SQuAD v1.1
* **Objective**: Extract contiguous answer text spans from unstructured context passages based on an input question.
* **Architecture**: Span classification head computing start and end token logit probabilities via cross-entropy loss against ground-truth token index spans.
* **Hyperparameters**:
  * **Epochs**: `3`
  * **Learning Rate**: `5e-5`
  * **Batch Size**: `32`
* **Evaluation Metrics**: Exact Match (EM) & Token-Level F1.

---

### 4. Sequential Continual Transfer Learning: SST-2 -> MRPC
* **Objective**: Investigate parameter stability and transferability by continuing training on MRPC directly from the SST-2 fine-tuned model weights.
* **Findings**: Direct fine-tuning on MRPC yielded slightly stronger performance compared to the sequential transfer pipeline. Because the base model already possesses strong cross-sentence representations from pre-training, single-sentence sentiment fine-tuning introduces minor task interference for sentence-pair equivalence tasks.

---

## 📊 Summary of Training Configurations

| Task | Benchmark | Task Type | Epochs | Learning Rate | Batch Size | Primary Metrics |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **SST-2** | GLUE | Single-Sequence Binary Classification | 3 | 2e-5 | 32 | Accuracy |
| **MRPC (Direct)** | GLUE | Sentence-Pair Paraphrase Detection | 3 | 2e-5 | 32 | Accuracy, F1 |
| **SST-2 -> MRPC** | Continual | Sequential Transfer Learning | 3 | 2e-5 | 32 | Accuracy, F1 |
| **SQuAD** | SQuAD v1.1 | Span-Extractive Question Answering | 3 | 5e-5 | 32 | Exact Match (EM), F1 |

---

## 🛠️ Interactive Inference Pipelines

Built dedicated interactive inference pipelines using the fine-tuned model weights to allow real-time testing on custom unseen inputs:

### 1. Real-Time Sentiment Analysis
```python
# Custom Input: "I absolutely loved the new sci-fi movie, the visuals were stunning!"
# Output: Positive (Confidence: 0.9983)

2. Sentence Semantic Equivalence Tool
# Sentence 1: "The quick brown fox jumps over the lazy dog."
# Sentence 2: "A fast brown fox leaps across a sleepy dog."
# Output: Equivalent (Confidence: 0.6987)


3. Custom Reading Comprehension / QA Tool
# Context: "Google Colab is a hosted Jupyter notebook service that requires no setup to use, while providing free access to computing resources including GPUs..."
# Question: "What is Google Colab widely used for?"
# Predicted Answer: "machine learning, data analysis, and education" (Confidence: 0.8089)

📂 Repository Structure
├── bert_multitask_nlp.ipynb    # Complete executable notebook with training runs & inference tools
├── README.md                   # Technical documentation and benchmark summary
└── .gitignore                  # Ignore cache, local checkpoints, and large files

🚀 Reproduction & Setup
Prerequisites
Python 3.10+
CUDA-compatible GPU runtime (e.g., NVIDIA T4 / A100)

Installation
pip install torch transformers datasets evaluate accelerate numpy pandas

Running the Notebook
Open bert_multitask_nlp.ipynb in Google Colab or your local Jupyter environment.

Select a GPU runtime (Runtime -> Change runtime type -> GPU).

Run all cells to replicate the tokenization pipelines, training runs, metric evaluations, and interactive tools.

👤 Author
Ryan Chen | github.com/rchen4501
