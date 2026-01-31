# CSE256 PA1: Sentiment Analysis with Neural Networks

A comprehensive implementation of sentiment analysis using various neural network architectures including Bag-of-Words (BOW), Deep Averaging Networks (DAN), and Byte-Pair Encoding (BPE) tokenization.

## Table of Contents

- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Running Models](#running-models)
- [Running Experiments](#running-experiments)
- [Command-Line Arguments](#command-line-arguments)
- [Project Structure](#project-structure)
- [Output Files](#output-files)

## Project Overview

This project implements and compares different neural network models for sentiment analysis:

1. **BOW (Bag-of-Words)**: Simple neural networks using BOW embeddings
2. **DAN (Deep Averaging Network)**: Networks using GloVe word embeddings
3. **DAN_BPE**: DAN with Byte-Pair Encoding tokenization

## Requirements

```bash
pip install torch scikit-learn matplotlib
```

## Quick Start

Run all models with default settings:

```bash
# BOW model (2 and 3 layers)
python main.py --model BOW

# DAN with pretrained GloVe 300d embeddings
python main.py --model DAN

# DAN with BPE tokenization
python main.py --model DAN_BPE
```

## Running Models

### 1. Bag-of-Words Model

Trains both 2-layer and 3-layer neural networks with BOW features:

```bash
python main.py --model BOW
```

**Output:**

- Training and dev accuracy printed every 10 epochs
- Plots saved to `experiment_fig/train_accuracy.png` and `experiment_fig/dev_accuracy.png`

---

### 2. Deep Averaging Network (DAN)

#### Using Pretrained GloVe 300d Embeddings (Default):

```bash
python main.py --model DAN
```

#### Using Pretrained GloVe 50d Embeddings:

```bash
python main.py --model DAN --embeddings 50
```

#### Training from Scratch (Custom Embedding Dimension):

```bash
python main.py --model DAN --no-from_pretrained --embed_dim 100
```

---

### 3. DAN with BPE Tokenization

Trains a DAN model using Byte-Pair Encoding:

```bash
python main.py --model DAN_BPE
```

With custom embedding dimension:

```bash
python main.py --model DAN_BPE --embed_dim 200
```

---

## Running Experiments

Automated experiment scripts are available in the `experiment_code/` directory.

### DAN Experiments

Runs comprehensive experiments varying embeddings, layers, hidden dimensions, and dropout:

```bash
cd experiment_code
python run_DAN_experiments.py
```

**Experiments:**

1. Embedding variations (pretrained vs scratch, 50d vs 300d)
2. Layer depth (2, 3, 4 layers)
3. Hidden dimensions (100, 200, 300)
4. Dropout rates (0.0, 0.25, 0.5, 0.75)

**Output:** Plots saved to `experiment_fig/exp1_*.png` through `experiment_fig/exp4_*.png`

---

### BPE Experiments

Runs comprehensive experiments with BPE tokenization:

```bash
cd experiment_code
python run_BPE_experiments.py
```

**Experiments:**

1. Vocab size comparison (500, 1000, 1500, 2000)
2. BPE vs pretrained GloVe embeddings
3. Layer depth (2, 3, 4 layers)
4. Hidden dimensions (100, 200, 300)
5. Dropout rates (0.0, 0.25, 0.5, 0.75)

**Output:** Plots saved to `experiment_fig/bpe_exp1_*.png` through `experiment_fig/bpe_exp5_*.png`

---

## Command-Line Arguments

### `main.py` Arguments

| Argument               | Type  | Default      | Description                                           |
| ---------------------- | ----- | ------------ | ----------------------------------------------------- |
| `--model`              | str   | **Required** | Model type: `BOW`, `DAN`, or `DAN_BPE`                |
| `--from_pretrained`    | flag  | `True`       | Use pretrained embeddings (DAN only)                  |
| `--no-from_pretrained` | flag  | -            | Train embeddings from scratch                         |
| `--embeddings`         | int   | `300`        | Pretrained embedding dimension (50 or 300)            |
| `--embed_dim`          | int   | `None`       | Custom embedding dimension when training from scratch |
| `--n_hidden`           | int   | `100`        | Hidden layer size                                     |
| `--n_class`            | int   | `2`          | Number of output classes                              |
| `--n_layers`           | int   | `2`          | Number of hidden layers                               |
| `--dropout`            | float | `0`          | Dropout rate (0.0 to 1.0)                             |

### Examples

**DAN with custom architecture:**

```bash
python main.py --model DAN --n_hidden 200 --n_layers 3 --dropout 0.25
```

**Training from scratch with 100d embeddings:**

```bash
python main.py --model DAN --no-from_pretrained --embed_dim 100 --n_hidden 150 --dropout 0.3
```

**BPE with custom architecture:**

```bash
python main.py --model DAN_BPE --embed_dim 150 --n_hidden 200 --n_layers 3 --dropout 0.5
```

---

## Project Structure

```
CSE256_PA1_WI26/
├── main.py                      # Main training script
├── BOWmodels.py                 # BOW dataset and models
├── DANmodels.py                 # DAN dataset and models
├── bpe_tokenizer.py             # BPE tokenizer implementation
├── sentiment_data.py            # Data loading utilities
├── utils.py                     # Utility functions
├── data/                        # Training and dev data
│   ├── train.txt
│   ├── dev.txt
│   ├── glove.6B.50d-relativized.txt
│   └── glove.6B.300d-relativized.txt
├── experiment_code/             # Automated experiment scripts
│   ├── run_DAN_experiments.py
│   └── run_BPE_experiments.py
├── experiment_fig/              # Generated plots
├── experiment_log/              # Experiment logs
└── bpe_models/                  # Cached BPE tokenizers
```

---

## Output Files

### Experiment Figures

All plots are saved to `experiment_fig/`:

**BOW Model:**

- `train_accuracy.png` - Training accuracy comparison
- `dev_accuracy.png` - Dev accuracy comparison

**DAN Experiments:**

- `exp1_embeddings.png` - Embedding variations
- `exp2_layers.png` - Layer depth impact
- `exp3_hidden_dim.png` - Hidden dimension impact
- `exp4_dropout.png` - Dropout regularization

**BPE Experiments:**

- `bpe_exp1_vocab_size.png` - Vocab size impact
- `bpe_exp2_vs_pretrained.png` - BPE vs pretrained comparison
- `bpe_exp3_layers.png` - Layer depth impact
- `bpe_exp4_hidden_dim.png` - Hidden dimension impact
- `bpe_exp5_dropout.png` - Dropout regularization

### BPE Model Caching

Trained BPE tokenizers are cached in `bpe_models/` for reuse:

- `bpe_vocab500.pkl`
- `bpe_vocab1000.pkl`
- `bpe_vocab1500.pkl`
- `bpe_vocab2000.pkl`

---

## Training Details

- **Optimizer:** Adam (lr=0.0001)
- **Loss Function:** Negative Log Likelihood (NLLLoss)
- **Epochs:** 100
- **Batch Size:** 16
- **Sentence Length:** 512 tokens (padded/truncated)

Progress is printed every 10 epochs showing training and dev accuracy.

---

## Notes

- **BOW Model:** Uses CountVectorizer with max_features=512
- **DAN Model:** Averages word embeddings before passing through the network
- **BPE Model:** Learns subword units for better handling of rare/unseen words
- All models use log-softmax activation for binary classification

---

## Author

UCSD CSE256 Winter 2026 - Programming Assignment 1
