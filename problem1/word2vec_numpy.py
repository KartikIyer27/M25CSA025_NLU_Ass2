"""
Pure NumPy Word2Vec implementation — drop-in replacement for gensim's Word2Vec.
Supports Skip-gram (sg=1) and CBOW (sg=0) with negative sampling.
Saves/loads model as:  <path>.vectors.npy  +  <path>.vocab.json
"""

import numpy as np
import os
import json
from collections import Counter


# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────

def _sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


# ──────────────────────────────────────────────────────────────────────────────
# WordVectors — mimics gensim's model.wv  interface
# ──────────────────────────────────────────────────────────────────────────────

class WordVectors:
    def __init__(self, vocab: dict, vectors: np.ndarray):
        """
        vocab   : { word: index }
        vectors : shape (vocab_size, vector_size) — input embeddings
        """
        self.vocab        = vocab
        self.index_to_key = list(vocab.keys())
        self.vectors      = vectors.astype(np.float32)

        # Pre-compute L2-normalised matrix for fast cosine similarity
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._normed = self.vectors / norms

    # ── dict-style access: model.wv["word"] ───────────────────────────────────
    def __getitem__(self, word):
        return self.vectors[self.vocab[word]]

    def __contains__(self, word):
        return word in self.vocab

    # ── key_to_index property (gensim compat) ─────────────────────────────────
    @property
    def key_to_index(self):
        return self.vocab

    # ── most_similar ──────────────────────────────────────────────────────────
    def most_similar(self, word=None, positive=None, negative=None, topn=5):
        """
        most_similar("student", topn=5)
        most_similar(positive=["student","campus"], negative=["leave"], topn=5)
        """
        if word is not None:
            positive = [word]
            negative = []
        if positive is None:
            positive = []
        if negative is None:
            negative = []

        # Build query vector from normalised embeddings
        query = np.zeros(self.vectors.shape[1], dtype=np.float32)
        for w in positive:
            if w in self.vocab:
                query += self._normed[self.vocab[w]]
            else:
                raise KeyError(f"Word '{w}' not in vocabulary")
        for w in negative:
            if w in self.vocab:
                query -= self._normed[self.vocab[w]]
            else:
                raise KeyError(f"Word '{w}' not in vocabulary")

        q_norm = np.linalg.norm(query)
        if q_norm == 0:
            return []
        query /= q_norm

        # Cosine similarities against all vectors
        sims = self._normed @ query   # (vocab_size,)

        exclude = set(positive) | set(negative)
        results = []
        for idx in np.argsort(sims)[::-1]:
            w = self.index_to_key[idx]
            if w not in exclude:
                results.append((w, float(sims[idx])))
                if len(results) == topn:
                    break
        return results


# ──────────────────────────────────────────────────────────────────────────────
# Word2Vec
# ──────────────────────────────────────────────────────────────────────────────

class Word2Vec:
    """
    NumPy Word2Vec with negative sampling.
    API mirrors gensim.models.Word2Vec.

    Parameters
    ----------
    sentences    : list of list of str
    vector_size  : embedding dimension  (default 100)
    window       : context window size  (default 5)
    sg           : 1 = Skip-gram, 0 = CBOW  (default 0)
    negative     : number of negative samples  (default 5)
    min_count    : ignore words with freq < min_count  (default 1)
    epochs       : training epochs  (default 5)
    learning_rate: initial SGD lr  (default 0.025)
    """

    def __init__(self, sentences=None, vector_size=100, window=5, sg=0,
                 negative=5, min_count=1, epochs=15, learning_rate=0.025):
        self.vector_size   = vector_size
        self.window        = window
        self.sg            = sg
        self.negative      = negative
        self.min_count     = min_count
        self.epochs        = epochs
        self.lr            = learning_rate
        self.wv            = None

        if sentences is not None:
            self._build_vocab(sentences)
            self._train(sentences)

    # ── Vocabulary ────────────────────────────────────────────────────────────

    def _build_vocab(self, sentences):
        counts = Counter(w for sent in sentences for w in sent)
        words  = [w for w, c in counts.most_common() if c >= self.min_count]

        self.vocab         = {w: i for i, w in enumerate(words)}
        self.vocab_size    = len(self.vocab)
        self.index_to_word = {i: w for w, i in self.vocab.items()}

        # Unigram^0.75 negative-sampling table
        freqs  = np.array([counts[w] ** 0.75 for w in words], dtype=np.float64)
        freqs /= freqs.sum()
        table_size   = 1_000_000
        counts_table = np.round(freqs * table_size).astype(np.int64)
        counts_table = np.maximum(counts_table, 1)   # every word at least once
        self._neg_table = np.repeat(np.arange(self.vocab_size, dtype=np.int32),
                                    counts_table)

    def _sample_negatives(self, exclude_set, k):
        out = []
        while len(out) < k:
            batch = self._neg_table[
                np.random.randint(0, len(self._neg_table), size=k * 4)
            ]
            for idx in batch:
                if idx not in exclude_set:
                    out.append(int(idx))
                    if len(out) == k:
                        break
        return out[:k]

    # ── Training ──────────────────────────────────────────────────────────────

    def _train(self, sentences):
        V, D = self.vocab_size, self.vector_size
        W     = (np.random.rand(V, D).astype(np.float32) - 0.5) / D
        W_out = np.zeros((V, D), dtype=np.float32)

        for epoch in range(self.epochs):
            total_loss = 0.0
            word_count = 0
            for sent in sentences:
                ids = [self.vocab[w] for w in sent if w in self.vocab]
                if len(ids) < 2:
                    continue
                if self.sg:
                    total_loss += self._train_skipgram(W, W_out, ids)
                else:
                    total_loss += self._train_cbow(W, W_out, ids)
                word_count += len(ids)

            avg = total_loss / word_count if word_count else 0.0
            print(f"  Epoch {epoch + 1}/{self.epochs}  avg_loss={avg:.4f}")

        self.wv = WordVectors(self.vocab, W)

    # ── Skip-gram update ──────────────────────────────────────────────────────

    def _train_skipgram(self, W, W_out, ids):
        total_loss = 0.0
        for i, center in enumerate(ids):
            start = max(0, i - self.window)
            end   = min(len(ids), i + self.window + 1)
            for j in range(start, end):
                if j == i:
                    continue
                ctx  = ids[j]
                negs = self._sample_negatives({center}, self.negative)

                h           = W[center]                          # (D,)
                all_targets = np.array([ctx] + negs, dtype=np.int32)
                labels      = np.array([1.0] + [0.0] * self.negative,
                                       dtype=np.float32)

                scores = _sigmoid(W_out[all_targets] @ h)       # (1+neg,)
                errors = (scores - labels) * self.lr             # (1+neg,)

                grad_h = errors @ W_out[all_targets]             # (D,)
                for k_idx, t in enumerate(all_targets):
                    W_out[t] -= errors[k_idx] * h
                W[center] -= grad_h

                total_loss += (- np.log(scores[0] + 1e-10)
                               - np.sum(np.log(1.0 - scores[1:] + 1e-10)))
        return total_loss

    # ── CBOW update ───────────────────────────────────────────────────────────

    def _train_cbow(self, W, W_out, ids):
        total_loss = 0.0
        for i, target in enumerate(ids):
            ctx_ids = [ids[j]
                       for j in range(max(0, i - self.window),
                                      min(len(ids), i + self.window + 1))
                       if j != i]
            if not ctx_ids:
                continue
            negs = self._sample_negatives({target}, self.negative)

            h           = W[ctx_ids].mean(axis=0)               # (D,)
            all_targets = np.array([target] + negs, dtype=np.int32)
            labels      = np.array([1.0] + [0.0] * self.negative,
                                   dtype=np.float32)

            scores = _sigmoid(W_out[all_targets] @ h)
            errors = (scores - labels) * self.lr

            grad_h = errors @ W_out[all_targets]                # (D,)
            for k_idx, t in enumerate(all_targets):
                W_out[t] -= errors[k_idx] * h

            grad_per = grad_h / len(ctx_ids)
            for ci in ctx_ids:
                W[ci] -= grad_per

            total_loss += (- np.log(scores[0] + 1e-10)
                           - np.sum(np.log(1.0 - scores[1:] + 1e-10)))
        return total_loss

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, path):
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        np.save(path + ".vectors.npy", self.wv.vectors)
        with open(path + ".vocab.json", "w", encoding="utf-8") as f:
            json.dump(self.wv.vocab, f, ensure_ascii=False)
        print(f"  Saved → {path}.vectors.npy  +  {path}.vocab.json")

    @classmethod
    def load(cls, path):
        vectors = np.load(path + ".vectors.npy")
        with open(path + ".vocab.json", "r", encoding="utf-8") as f:
            vocab = json.load(f)
        obj    = cls.__new__(cls)
        obj.wv = WordVectors(vocab, vectors)
        return obj
