# Assignment 2 - Run Guide

This repository has two parts:
- `problem1`: Corpus processing + custom NumPy Word2Vec (CBOW/Skip-gram)
- `problem2`: Name generation models (Vanilla RNN, BLSTM, RNN+Attention)

## 1) Environment Setup

From the repository root (`Assignment2`), install required packages.

```bash
pip install numpy nltk pdfplumber matplotlib scikit-learn wordcloud torch
pip install wordninja
```

Notes:
- `wordninja` is optional in code, but recommended for better token splitting.
- If `pip` points to a different Python, use your active venv python explicitly.

## 2) Problem 1 - Stepwise Execution

Working directory:

```bash
cd problem1
```

### Step 1: Extract raw text from PDFs
(I have uploaded the corpus.txt itself)
```bash
python extract_text.py 
```

Input:
- `data/file1.pdf` ... `data/file6.pdf` 

Output:
- `data/raw_corpus.txt`

### Step 2: Clean corpus and create sentence corpus

```bash
python clean_corpus.py
```

Outputs:
- `cleaned data/c_corpus.txt`
- `cleaned data/clean_corpus.txt`
- `cleaned data/clean_sentences.txt`

### Step 3: Tokenize cleaned corpus

```bash
python tokenize_text.py
```

Output:
- `cleaned data/tokens.txt`

### Step 4: Train Word2Vec models

```bash
python train_word2vec.py
```

Expected output folder/files:
- `models/cbow.model`
- `models/cbow.model.vocab.json`
- `models/cbow.model.vectors.npy`
- `models/skipgram.model`
- `models/skipgram.model.vocab.json`
- `models/skipgram.model.vectors.npy`

### Step 5: Optional analysis scripts

```bash
python corpus_stats.py
python similarWords.py
python analogy_test.py
python visualize_embedding.py
python wordCloud_plot.py
```

## 3) Problem 2 - Stepwise Execution

Working directory:

```bash
cd ../problem2
```

### Step 1: Train Vanilla RNN and generate names

```bash
python vanilla_rnn.py
```

Outputs:
- `rnn_loss.json`
- `vanilla_rnn_loss_curve.png`
- `rnn.txt`
- `vanilla_rnn_model.pth`

### Step 2: Train BLSTM model and generate names

```bash
python blsmt1.py
```

Outputs:
- `blstm_loss.json`
- `blsmt1_loss_curve.png`
- `blstm.txt`
- (checkpoint save is not implemented in this script)

### Step 3: Train RNN+Attention and generate names

```bash
python rnn_attention.py
```

Outputs:
- `attention_loss.json`
- `rnn_attention_loss_curve.png`
- `attention.txt`
- `rnn_attention_model.pth`

### Step 4: Evaluate generated names

```bash
python evaluation.py
```

Reads:
- `rnn.txt`
- `blstm.txt`
- `attention.txt`
- `TrainingNames1.txt` (fallback: `TrainingNames.txt`)

### Step 5: Plot comparison charts

```bash
python plots.py
```

Outputs in `plots/`:
- `unique_names_comparison.png`
- `novel_names_comparison.png`
- `novelty_comparison.png`
- `diversity_comparison.png`
- `epochs_comparison.png`
- `novelty_diversity_comparison.png`

 names, `*_loss.json`, `*_loss_curve.png`, `plots/`, and optional `*.pth`
